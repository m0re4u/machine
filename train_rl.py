import argparse
import datetime
import logging
import os
import time

import gym
import torch

import babyai
from machine.trainer import ReinforcementTrainer
from machine.models import ACModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONSTANTS
LOG_FORMAT = '%(asctime)s %(name)-6s %(levelname)-6s %(message)s'


def train_model():
    # Create command line argument parser
    parser = init_argparser()
    opt = parser.parse_args()

    # Start logger first
    init_logging(opt.log_level)

    #  validate chosen options
    opt = validate_options(parser, opt)

    # Prepare logging and environment
    envs = []
    for i in range(opt.num_processes):
        env = gym.make(opt.env_name)
        env.seed(100 * opt.seed + i)
        envs.append(env)

    # Create model name
    suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    instr = opt.instr_arch if opt.instr_arch else "noinstr"
    mem = "mem" if not opt.no_mem else "nomem"
    model_name_parts = {
        'env': opt.env_name,
        'arch': opt.arch,
        'instr': instr,
        'mem': mem,
        'seed': opt.seed,
        'info': '',
        'coef': '',
        'suffix': suffix}
    model_name = "{env}_{arch}_{instr}_{mem}_seed{seed}{info}{coef}_{suffix}".format(
        **model_name_parts)

    # Prepare model
    obss_preprocessor = babyai.utils.ObssPreprocessor(
        model_name, envs[0].observation_space, None)
    acmodel = ACModel(obss_preprocessor.obs_space, envs[0].action_space,
                                   opt.image_dim, opt.memory_dim, opt.instr_dim,
                                   not opt.no_instr, opt.instr_arch, not opt.no_mem, opt.arch)

    obss_preprocessor.vocab.save()

    if torch.cuda.is_available():
        acmodel.cuda()

    def reshape_reward(_0, _1, reward, _2): return opt.reward_scale * reward
    # Prepare trainer
    from babyai.rl.utils import ParallelEnv
    trainer = ReinforcementTrainer(ParallelEnv(
        envs), opt, acmodel, obss_preprocessor, reshape_reward, 'ppo')

    # Start training
    trainer.train()


def init_argparser():
    parser = argparse.ArgumentParser()

    # Training algorithm arguments
    parser.add_argument('--env-name', required=True,
                        help='Name of the environment to use')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--lr', type=float, help='Learning rate, recommended \
                        settings.\nrecommended settings: adam=0.001 \
                        adadelta=1.0 adamax=0.002 rmsprop=0.01 sgd=0.1',
                        default=0.001)
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--batch_size', type=int, default=1280,
                        help='number of batches for ppo (default: 1280)')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False,
                        help='use a linear schedule on the ppo clipping parameter')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-env-steps', type=int, default=10e6,
                        help='number of environment steps to train (default: 10e6)')
    parser.add_argument("--frames-per-proc", type=int, default=40,
                        help="number of frames per process before update (default: 40)")
    parser.add_argument("--optim-eps", type=float, default=1e-5,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
    parser.add_argument("--reward-scale", type=float, default=20.,
                        help="Reward scale multiplier")
    parser.add_argument("--recurrence", type=int, default=20,
                        help="number of timesteps gradient is backpropagated (default: 20)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--frames", type=int, default=int(9e10),
                        help="number of frames of training (default: 9e10)")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="beta1 for Adam (default: 0.9)")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 for Adam (default: 0.999)")
    parser.add_argument('--ppo-epochs', type=int,
                        help='Number of epochs', default=6)

    # Model parameters
    parser.add_argument("--image-dim", type=int, default=128,
                        help="dimensionality of the image embedding")
    parser.add_argument("--memory-dim", type=int, default=128,
                        help="dimensionality of the memory LSTM")
    parser.add_argument("--instr-dim", type=int, default=128,
                        help="dimensionality of the memory LSTM")
    parser.add_argument("--no-instr", action="store_true", default=False,
                        help="don't use instructions in the model")
    parser.add_argument("--instr-arch", default="gru",
                        help="arch to encode instructions, possible values: gru, bigru, conv, bow (default: gru)")
    parser.add_argument("--no-mem", action="store_true", default=False,
                        help="don't use memory in the model")
    parser.add_argument("--arch", default='expert_filmcnn',
                        help="image embedding architecture")

    # Logging and model saving
    parser.add_argument('--tb',
                        help='Run tensorboard', action='store_true',)
    parser.add_argument('--load_checkpoint',
                        help='The name of the checkpoint to load, usually an encoded time string')
    parser.add_argument('--resume',
                        help='Indicates if training has to be resumed from the latest checkpoint', action='store_true',)
    parser.add_argument('--save_every', type=int,
                        help='Every how many batches the model should be saved', default=100)
    parser.add_argument('--print_every', type=int,
                        help='Every how many batches to print results', default=100)
    parser.add_argument('--output_dir', default='../models',
                        help='Path to model directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--log-level',
                        help='Logging level.', default='info', choices=['info','debug','warning','error','notset','critical'])
    parser.add_argument('--write-logs',
                        help='Specify file to write logs to after training')
    parser.add_argument('--cuda_device', default=0,
                        type=int, help='set cuda device to use')

    return parser


def validate_options(parser, opt):
    if opt.resume and not opt.load_checkpoint:
        parser.error("load_checkpoint argument is required to resume training from checkpoint")

    if torch.cuda.is_available():
        logging.info(f"CUDA device set to {opt.cuda_device}")
        torch.cuda.set_device(opt.cuda_device)
    else:
        logging.info("CUDA not available")

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    config = vars(opt)
    logging.info("Parameters:")
    for k, v in config.items():
        logging.info(f"  {k:>21} : {v}")

    return opt


def init_logging(level):
    logging.basicConfig(format=LOG_FORMAT, level=getattr(
        logging, level.upper()))

if __name__ == "__main__":
    train_model()
