import argparse
import datetime
import logging
from pathlib import Path

import gym
import numpy as np
import torch

import machine
from machine.models import ACModel, IACModel, SkillEmbedding
from machine.trainer import ReinforcementTrainer
from machine.util import ObssPreprocessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONSTANTS
LOG_FORMAT = '%(asctime)s %(name)-6s %(levelname)-6s %(message)s'
VOCAB_FILENAME = Path('vocab.json')


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
    from babyai.rl.utils import ParallelEnv
    p_envs = ParallelEnv(envs)

    # Create model name
    model_name = get_model_name(opt)

    # Observation preprocessor
    obss_preprocessor = ObssPreprocessor(
        model_name, envs[0].observation_space, load_vocab_from=opt.vocab_file, segment_level=opt.segment_level)
    obss_preprocessor.vocab.save()

    def reshape_reward(_0, _1, reward, _2): return opt.reward_scale * reward

    # Prepare agent model. If option critic, make as many as there are options
    if opt.oc:
        algo = 'ppoc'
        args = [obss_preprocessor.obs_space, envs[0].action_space, opt.image_dim, opt.memory_dim,
                opt.instr_dim, not opt.no_instr, opt.instr_arch, not opt.no_mem, opt.arch]
        model = [ACModel(*args) for _ in range(opt.n_options)]
        if torch.cuda.is_available():
            for m in model:
                m.cuda()
    elif opt.resume and (not opt.oc):
        algo = 'ppo'
        model = machine.util.RLCheckpoint.load_model(opt.load_checkpoint)
        if torch.cuda.is_available():
            model.cuda()
        model.train()
    elif opt.se:
        algo = 'ppo'
        model = SkillEmbedding(obss_preprocessor.obs_space['image'],
                               envs[0].action_space, opt.n_skills, obss_preprocessor.vocab,
                               opt.image_dim, opt.memory_dim, not opt.no_mem, opt.mapping,
                               opt.num_processes, opt.trunk_arch)
        if torch.cuda.is_available():
            model.cuda()
    else:
        algo = 'ppo'
        model = ACModel(obss_preprocessor.obs_space, envs[0].action_space,
                        opt.image_dim, opt.memory_dim, opt.instr_dim,
                        not opt.no_instr, opt.instr_arch, not opt.no_mem, opt.arch)
        if torch.cuda.is_available():
            model.cuda()

    trainer = ReinforcementTrainer(
        p_envs, opt, model, model_name, obss_preprocessor, reshape_reward, algo)

    # Start training
    trainer.train()


def init_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)

    # Training algorithm arguments
    parser.add_argument('--env-name', required=True,
                        help='Name of the environment to use')
    parser.add_argument('--lr', type=float, help='Learning rate. Recommended settings:\n  adam=0.001\n  adadelta=1.0\n  adamax=0.002\n  rmsprop=0.01\n  sgd=0.1\n  (default: 0.0001)',
                        default=0.0001)
    parser.add_argument('--num-processes', type=int, default=64,
                        help='how many training CPU processes to use (default: 64)')
    parser.add_argument('--batch_size', type=int, default=1280,
                        help='number of batches for ppo (default: 1280)')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False,
                        help='use a linear schedule on the ppo clipping parameter')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-env-steps', type=int, default=10e6,
                        help='number of environment steps to train (default: 10e6)')
    parser.add_argument("--frames-per-proc", type=int, default=40,
                        help="number of frames per process before update (default: 40)")
    parser.add_argument("--optim-eps", type=float, default=1e-5,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
    parser.add_argument("--recurrence", type=int, default=20,
                        help="number of timesteps gradient is backpropagated (default: 20)")
    parser.add_argument("--frames", type=int, default=int(9e10),
                        help="number of frames of training (default: 9e10)")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="beta1 for Adam (default: 0.9)")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 for Adam (default: 0.999)")

    # Curiosity arguments
    parser.add_argument('--explore_for', type=int, default=0,
                        help='Explore for amount of cycles (default: 0)')
    parser.add_argument('--disrupt_coef', type=float, default=1.0,
                        help='Multiply the disruptiveness metric with this value (default: 1.0)')
    parser.add_argument('--disrupt', type=int, default=0,
                        help="Add disruptiveness metric with the following semantics:\n\n\
0: No disruptiveness\n\
1: policy_loss * log(sum(binary_diff(s1,s2))), clipped to [0.01, 10]\n\
2: value_loss * log(sum(binary_diff(s1,s2))), clipped to [0.01, 10]\n\
3: Replace advantage with intrinsic reward for explore_for frames\n\n")

    # PPO arguments
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--reward-scale", type=float, default=20.,
                        help="Reward scale multiplier")
    parser.add_argument('--gae-lambda', type=float, default=0.99,
                        help='gae parameter (default: 0.99)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument('--ppo-epochs', type=int, default=4,
                        help='Number of epochs (default: 4)')

    # Option-Critic arguments
    parser.add_argument('--oc', action='store_true',
                        help='Enable option-critic version of ppo')
    parser.add_argument('--n_options', type=int, default=1,
                        help='How many options to consider')

    # Skill embedding arguments
    parser.add_argument('--se', action='store_true',
                        help='Enable skill embeddings')
    parser.add_argument('--n_skills', type=int, default=6,
                        help='How many skills to consider')
    parser.add_argument('--mapping', type=str, default='color', choices=['color', 'object', 'command', 'random', 'constant'],
                        help='What mapping to use to select the skill trunks')
    parser.add_argument('--trunk_arch', type=str, default='fcn', choices=['fcn', 'cnn'],
                        help='Skill trunk architecture')

    # Segmentation arguments
    parser.add_argument('--segment_level', type=str, default='word', choices=['word', 'segment', 'word_annotated'],
                        help='Segmentation level')

    # Model parameters
    parser.add_argument("--image-dim", type=int, default=128,
                        help="dimensionality of the image embedding")
    parser.add_argument("--memory-dim", type=int, default=128,
                        help="dimensionality of the memory LSTM")
    parser.add_argument("--instr-dim", type=int, default=128,
                        help="dimensionality of the instruction LSTM")
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
                        help='Indicates if training has to be resumed from the given checkpoint', action='store_true',)
    parser.add_argument('--save_every', type=int,
                        help='Every how many batches the model should be saved', default=100)
    parser.add_argument('--print_every', type=int,
                        help='Every how many batches to print results', default=100)
    parser.add_argument('--output_dir', default='models',
                        help='Path to output model directory to save checkpoints in.')
    parser.add_argument('--log-level',
                        help='Logging level.', default='info', choices=['info', 'debug', 'warning', 'error', 'notset', 'critical'])
    parser.add_argument('--cuda_device', default=0,
                        type=int, help='set cuda device to use')
    parser.add_argument('--slurm_id', default=0,
                        type=int, help='Get the SLURM job id if we\'re running on LISA')

    return parser


def validate_options(parser, opt):
    if opt.resume and not opt.load_checkpoint:
        parser.error(
            "load_checkpoint argument is required to resume training from checkpoint")

    if opt.disrupt == 3 and opt.explore_for == 0:
        parser.error(
            "Disrupt with intrinsic reward set but no exploration frames")
    if opt.disrupt == 0 and opt.explore_for > 0:
        parser.error("No disrupt but exploration frames are defined")

    if torch.cuda.is_available():
        logging.info(f"CUDA device set to {opt.cuda_device}")
        torch.cuda.set_device(opt.cuda_device)
    else:
        logging.info("CUDA not available")

    # Option-Critic and SkillEmbedding are mutually exclusive
    if opt.se:
        assert not opt.oc
    if opt.oc:
        logging.warning("OPTION-CRITIC IS UNFINISHED AND SHOULD NOT BE USED")
        assert not opt.se

    if opt.resume and opt.load_checkpoint:
        checkpoint = Path(opt.load_checkpoint)
        logging.info(f"Resuming from checkpoint {checkpoint}")
        assert checkpoint.exists()
        vocab_file = checkpoint.parent.joinpath(VOCAB_FILENAME)
        assert vocab_file.exists()
        opt.vocab_file = str(vocab_file)
    else:
        opt.vocab_file = None

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)

    config = vars(opt)
    logging.info("Parameters:")

    # Ignore settings for Option-Critic or SkillEmbedding if they're not enabled
    if not config['oc']:
        config.pop('n_options')
    if not config['se']:
        config.pop('n_skills')
        config.pop('mapping')
        config.pop('trunk_arch')
    if not config['resume']:
        config.pop('load_checkpoint')

    for k, v in config.items():
        logging.info(f"  {k:>21} : {v}")

    return opt


def init_logging(level):
    logging.basicConfig(format=LOG_FORMAT,
                        level=getattr(logging, level.upper()))


def get_model_name(opt):
    suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    instr = opt.instr_arch if opt.instr_arch else "noinstr"
    mem = "mem" if not opt.no_mem else "nomem"
    alg = "PPO" if not opt.oc else "PPOC"
    jobid = f'_job{opt.slurm_id}' if opt.slurm_id != 0 else ''
    if opt.resume:
        checkpoint = Path(opt.load_checkpoint)
        prev_model_name = checkpoint.relative_to(opt.output_dir).parent
        original_level = str(prev_model_name).split("_")[0]
        res = f"RESUMED-{original_level}"
    else:
        res = ""

    if opt.se:
        mod = "SE"
        model_name_parts = {
            'alg': alg,
            'mod': mod,
            'env': opt.env_name,
            'res': res,
            'mem': mem,
            'seed': opt.seed,
            'jobid': jobid,
            'suffix': suffix
        }
        return "{env}-{res}_{alg}_{mod}_{mem}_seed{seed}{jobid}_{suffix}".format(**model_name_parts)
    else:
        mod = "AC"
        model_name_parts = {
            'alg': alg,
            'mod': mod,
            'env': opt.env_name,
            'res': res,
            'arch': opt.arch,
            'instr': instr,
            'mem': mem,
            'seed': opt.seed,
            'jobid': jobid,
            'suffix': suffix
        }
        return "{env}-{res}_{alg}_{mod}_{arch}_{instr}_{mem}_seed{seed}{jobid}_{suffix}".format(**model_name_parts)


if __name__ == "__main__":
    train_model()
