import os
import argparse
import logging
import datetime
import time

import torch
import gym

from machine.trainer import ReinforcementTrainer

import babyai


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONSTANTS
IGNORE_INDEX = -1
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'


def train_model():
    # Create command line argument parser and validate chosen options
    parser = init_argparser()
    opt = parser.parse_args()
    opt = validate_options(parser, opt)

    num_updates = int(opt.num_env_steps) // opt.num_steps // opt.num_processes

    # Prepare logging and environment
    init_logging(opt)
    envs = []
    for i in range(opt.num_processes):
        env = gym.make(opt.env_name)
        env.seed(100 * opt.seed + i)
        envs.append(env)

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
    default_model_name = "{env}_{arch}_{instr}_{mem}_seed{seed}{info}{coef}_{suffix}".format(**model_name_parts)
    opt.model = opt.model.format(**model_name_parts) if opt.model else default_model_name


    # Prepare model
    obss_preprocessor = babyai.utils.ObssPreprocessor(opt.model, envs[0].observation_space, None)
    acmodel = babyai.model.ACModel(obss_preprocessor.obs_space, envs[0].action_space,
                          opt.image_dim, opt.memory_dim, opt.instr_dim,
                          not opt.no_instr, opt.instr_arch, not opt.no_mem, opt.arch)

    obss_preprocessor.vocab.save()

    if torch.cuda.is_available():
        acmodel.cuda()

    reshape_reward = lambda _0, _1, reward, _2: opt.reward_scale * reward
    algo = babyai.rl.PPOAlgo(envs, acmodel, opt.frames_per_proc, opt.gamma, opt.lr, opt.beta1, opt.beta2,
                            opt.gae_lambda,
                            opt.entropy_coef, opt.value_loss_coef, opt.max_grad_norm, opt.recurrence,
                            opt.optim_eps, opt.clip_eps, opt.ppo_epochs, opt.num_mini_batch, obss_preprocessor,
                            reshape_reward)

    status = {'i': 0,
            'num_episodes': 0,
            'num_frames': 0}
    header = (["update", "episodes", "frames", "FPS", "duration"]
          + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["success_rate"]
          + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"])


    # Train model
    total_start_time = time.time()
    best_success_rate = 0
    test_env_name = opt.env_name
    while status['num_frames'] < opt.frames:
        # Update parameters
        update_start_time = time.time()
        logs = algo.update_parameters()
        update_end_time = time.time()

        status['num_frames'] += logs["num_frames"]
        status['num_episodes'] += logs['episodes_done']
        status['i'] += 1

        # Print logs
        if status['i'] % opt.print_every == 0:
            total_ellapsed_time = int(time.time() - total_start_time)
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = datetime.timedelta(seconds=total_ellapsed_time)
            return_per_episode = babyai.utils.synthesize(logs["return_per_episode"])
            success_per_episode = babyai.utils.synthesize(
                [1 if r > 0 else 0 for r in logs["return_per_episode"]])
            num_frames_per_episode = babyai.utils.synthesize(logs["num_frames_per_episode"])

            data = [status['i'], status['num_episodes'], status['num_frames'],
                    fps, total_ellapsed_time,
                    *return_per_episode.values(),
                    success_per_episode['mean'],
                    *num_frames_per_episode.values(),
                    logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"],
                    logs["loss"], logs["grad_norm"]]

            format_str = ("U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                        "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                        "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")

            logging.info(format_str.format(*data))

def init_argparser():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument('--env-name', help='')
    parser.add_argument('--model', help='', default=None)
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
    parser.add_argument('--num-mini-batch', type=int, default=1280,
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


    parser.add_argument('--monitor', nargs='+', default=[],
                        help='Data to monitor during training')
    parser.add_argument('--output_dir', default='../models',
                        help='Path to model directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
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

    # Data management
    parser.add_argument('--load_checkpoint',
                        help='The name of the checkpoint to load, usually an encoded time string')
    parser.add_argument('--save_every', type=int,
                        help='Every how many batches the model should be saved', default=100)
    parser.add_argument('--print_every', type=int,
                        help='Every how many batches to print results', default=100)
    parser.add_argument('--resume', action='store_true',
                        help='Indicates if training has to be resumed from the latest checkpoint')
    parser.add_argument('--log-level', default='info', help='Logging level.')
    parser.add_argument('--write-logs', help='Specify file to write logs to after training')
    parser.add_argument('--cuda_device', default=0,
                        type=int, help='set cuda device to use')

    return parser


def validate_options(parser, opt):
    if opt.resume and not opt.load_checkpoint:
        parser.error(
            'load_checkpoint argument is required to resume training from checkpoint')

    if torch.cuda.is_available():
        logging.info("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)


    return opt


def init_logging(opt):
    logging.basicConfig(format=LOG_FORMAT, level=getattr(
        logging, opt.log_level.upper()))
    logging.info(opt)


def prepare_environment(opt):
    src = SourceField()
    tgt = TargetField(include_eos=use_output_eos)
    tabular_data_fields = [('src', src), ('tgt', tgt)]

    max_len = opt.max_len

    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len

    # generate training and testing data
    train = torchtext.data.TabularDataset(
        path=opt.train, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter
    )

    if opt.dev:
        dev = torchtext.data.TabularDataset(
            path=opt.dev, format='tsv',
            fields=tabular_data_fields,
            filter_pred=len_filter
        )

    else:
        dev = None

    monitor_data = OrderedDict()
    for dataset in opt.monitor:
        m = torchtext.data.TabularDataset(
            path=dataset, format='tsv',
            fields=tabular_data_fields,
            filter_pred=len_filter)
        monitor_data[dataset] = m

    return src, tgt, train, dev, monitor_data


def load_model_from_checkpoint(opt):
    logging.info("loading checkpoint from {}".format(
        os.path.join(opt.output_dir, opt.load_checkpoint)))

    return None


def initialize_model(opt):
    # build vocabulary
    src.build_vocab(train, max_size=opt.src_vocab)
    tgt.build_vocab(train, max_size=opt.tgt_vocab)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # Initialize model
    hidden_size = opt.hidden_size
    decoder_hidden_size = hidden_size * 2 if opt.bidirectional else hidden_size
    encoder = EncoderRNN(len(src.vocab), opt.max_len, hidden_size, opt.embedding_size,
                         dropout_p=opt.dropout_p_encoder,
                         n_layers=opt.n_layers,
                         bidirectional=opt.bidirectional,
                         rnn_cell=opt.rnn_cell,
                         variable_lengths=True)
    decoder = DecoderRNN(len(tgt.vocab), opt.max_len, decoder_hidden_size,
                         dropout_p=opt.dropout_p_decoder,
                         n_layers=opt.n_layers,
                         attention_method=opt.attention_method,
                         full_focus=opt.full_focus,
                         bidirectional=opt.bidirectional,
                         rnn_cell=opt.rnn_cell,
                         eos_id=tgt.eos_id, sos_id=tgt.sos_id)
    seq2seq = Seq2seq(encoder, decoder)
    seq2seq.to(device)

    return seq2seq, input_vocab, output_vocab


def prepare_losses_and_metrics(
        opt, pad, unk, sos, eos, input_vocab, output_vocab):
    use_output_eos = not opt.ignore_output_eos

    # Prepare loss and metrics
    losses = [NLLLoss(ignore_index=pad)]
    loss_weights = [1.]

    for loss in losses:
        loss.to(device)

    metrics = []

    if 'word_acc' in opt.metrics:
        metrics.append(WordAccuracy(ignore_index=pad))
    if 'seq_acc' in opt.metrics:
        metrics.append(SequenceAccuracy(ignore_index=pad))
    if 'target_acc' in opt.metrics:
        metrics.append(FinalTargetAccuracy(ignore_index=pad, eos_id=eos))
    if 'sym_rwr_acc' in opt.metrics:
        metrics.append(SymbolRewritingAccuracy(
            input_vocab=input_vocab,
            output_vocab=output_vocab,
            use_output_eos=use_output_eos,
            output_sos_symbol=sos,
            output_pad_symbol=pad,
            output_eos_symbol=eos,
            output_unk_symbol=unk))
    if 'bleu' in opt.metrics:
        metrics.append(BLEU(
            input_vocab=input_vocab,
            output_vocab=output_vocab,
            use_output_eos=use_output_eos,
            output_sos_symbol=sos,
            output_pad_symbol=pad,
            output_eos_symbol=eos,
            output_unk_symbol=unk))

    return losses, loss_weights, metrics


def create_trainer(opt, losses, loss_weights, metrics):
    return SupervisedTrainer(loss=losses, metrics=metrics, loss_weights=loss_weights, batch_size=opt.batch_size,
                             eval_batch_size=opt.eval_batch_size, checkpoint_every=opt.save_every,
                             print_every=opt.print_every, expt_dir=opt.output_dir)


if __name__ == "__main__":
    train_model()
