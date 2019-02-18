
import logging
import os

import numpy as np

from machine.util.callbacks import Callback


class EpisodeLogger(Callback):
    """
    Callback that is used to log information during training of a reinforcement
    agent.
    Note that this abuses the callback function names, since we do not have
    a similar setup as supervised learning. Now, we consider the following
    semantics:
        epoch: episode
        batch: step
    """

    def __init__(self, use_tensorboard=False):
        super(EpisodeLogger, self).__init__()

        self.logger = logging.getLogger("EpisodeLogger")
        if use_tensorboard:
            from tensorboardX import SummaryWriter
            self.logger.info("Using Tensorboard")
            self.writer = SummaryWriter('runs')
        else:
            self.writer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, info=None):
        self.logger.info(f"Start of epoch: {info}")

    def on_epoch_end(self, info=None):
        pass

    def on_batch_begin(self, batch, info=None):
        pass

    def on_batch_end(self, batch, info=None):
        pass

    def on_train_begin(self, info=None):
        self.logger.info("Starting training")

    def on_train_end(self, info=None):
        self.logger.info("Finished training")

    def on_cycle_start(self):
        pass

    def on_cycle_end(self, status, logs, cycle_time):
        fps = logs['num_frames'] / cycle_time
        return_per_episode = get_stats(logs["return_per_episode"])
        success_per_episode = get_stats(
            [1 if r > 0 else 0 for r in logs["return_per_episode"]])
        num_frames_per_episode = get_stats(logs["num_frames_per_episode"])
        data = [
            status['i'],
            status['num_episodes'],
            status['num_frames'],
            fps,
            cycle_time,
            *return_per_episode.values(),
            success_per_episode['mean'],
            *num_frames_per_episode.values(),
            logs["entropy"],
            logs["value"],
            logs["policy_loss"],
            logs["value_loss"],
            logs["loss"],
            logs["grad_norm"]
        ]

        format_str = ("U {} | E {} | F {:6} | FPS {:3.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                      "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                      "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")

        self.logger.info(format_str.format(*data))
        if self.writer is not None:
            self.writer.add_scalar('train/fps', fps, status['i'])
            self.writer.add_scalar('train/succes', success_per_episode['mean'], status['i'])


def get_stats(arr):
    import collections
    d = collections.OrderedDict()
    d["mean"] = np.mean(arr)
    d["std"] = np.std(arr)
    d["min"] = np.amin(arr)
    d["max"] = np.amax(arr)
    return d
