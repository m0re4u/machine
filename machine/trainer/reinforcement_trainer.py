import datetime
import logging
import time

import numpy
import torch

import babyai
import machine
from machine.util.callbacks import EpisodeLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReinforcementTrainer(object):
    """
    The ReinforcementTrainer class helps in setting up a training framework for
    reinforcement learning.
    """

    def __init__(self, opt, algo):
        self._trainer = "Reinforcement Trainer"
        self.status = {
            'i': 0,
            'num_episodes': 0,
            'num_frames': 0
        }
        self.frames = opt.frames
        self.env = opt.env_name
        self.print_every = opt.print_every
        self.algo = algo
        self.logger = logging.getLogger(__name__)


    def train(self, model):
        callback = EpisodeLogger()

        # Start training model
        callback.on_train_begin()
        total_start_time = time.time()
        best_success_rate = 0
        while self.status['num_frames'] < self.frames:
            # Update parameters
            update_start_time = time.time()
            logs = self.algo.update_parameters()
            update_end_time = time.time()

            self.status['num_frames'] += logs["num_frames"]
            self.status['num_episodes'] += logs['episodes_done']
            self.status['i'] += 1

            # Print logs
            if self.status['i'] % self.print_every == 0:
                total_elapsed_time = int(time.time() - total_start_time)
                fps = logs["num_frames"] / (update_end_time - update_start_time)
                duration = datetime.timedelta(seconds=total_elapsed_time)
                return_per_episode = get_stats(logs["return_per_episode"])
                success_per_episode = get_stats([1 if r > 0 else 0 for r in logs["return_per_episode"]])
                num_frames_per_episode = get_stats(logs["num_frames_per_episode"])

                data = [self.status['i'], self.status['num_episodes'], self.status['num_frames'],
                        fps, total_elapsed_time,
                        *return_per_episode.values(),
                        success_per_episode['mean'],
                        *num_frames_per_episode.values(),
                        logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"],
                        logs["loss"], logs["grad_norm"]]

                format_str = ("U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                            "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                            "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")

                logging.info(format_str.format(*data))
        callback.on_train_end()


def get_stats(arr):
    import collections
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(arr)
    d["std"] = numpy.std(arr)
    d["min"] = numpy.amin(arr)
    d["max"] = numpy.amax(arr)
    return d
