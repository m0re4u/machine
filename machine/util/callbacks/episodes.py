import logging
import os
import time

import numpy as np
from collections import defaultdict
from machine.util.callbacks import Callback
from machine.util.checkpoint import RLCheckpoint


class EpisodeLogger(Callback):
    """
    Callback that is used to log information during training of a reinforcement
    agent.

    Note that we expand on the functions, as we assume a PPO algorithm setup,
    which is different from supervised learnig.

    Now, we consider the following additional steps:
        - cycle: Outer iterations of PPO
        - epoch: Epochs in update_parameters()
        - batch: Batches in update_parameters()
    """

    def __init__(self, print_every=10, save_every=10, model_name='', use_tensorboard=False, explore_for=100, reasoning=False):
        super(EpisodeLogger, self).__init__()

        self.logger = logging.getLogger("EpisodeLogger")

        self.print_every = print_every
        self.save_every = save_every
        self.model_name = model_name

        if use_tensorboard:
            from tensorboardX import SummaryWriter
            self.logger.info("Using Tensorboard")
            self.writer = SummaryWriter(os.path.join('runs', self.model_name))
        else:
            self.writer = None

        self.cycle = 0
        self.num_frames = 0
        self.num_episodes = 0
        self.explore_for = explore_for
        self.reasoning = reasoning

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, info=None):
        self.logger.info(f"Start of epoch: {info}")
        self.scalars = defaultdict(lambda: [])

    def on_epoch_end(self, logs=None, info=None):
        if logs is not None:
            new_logs = defaultdict(lambda: 0)
            for k, v in self.scalars.items():
                new_logs[k] = np.mean(v)
            new_logs.update(logs)
            return new_logs
        else:
            pass

    def on_batch_begin(self, batch, info=None):
        batch_scalars = defaultdict(lambda: 0)
        return batch_scalars

    def on_batch_end(self, loss, info=None):
        for k, v in info.items():
            self.scalars[k].append(v)
        self.scalars['loss'].append(loss)

    def on_train_begin(self, info=None):
        self.logger.info("Starting training")
        self.cycle = 0
        self.num_frames = 0
        self.num_episodes = 0

    def on_train_end(self, info=None):
        self.logger.info("Finished training")

    def on_cycle_start(self):
        self.cycle_start_time = time.time()
        reward_origin = 'intrinsic' if self.num_frames < self.explore_for else 'advantage'
        self.logger.info(
            f"Start of cycle: {self.cycle} - Reward from: {reward_origin}")

    def on_cycle_end(self, logs=None):
        self.cycle += 1
        if logs is None:
            return
        self.num_frames += logs['num_frames']
        self.num_episodes += logs['episodes_done']
        cycle_time = time.time() - self.cycle_start_time
        fps = logs['num_frames'] / cycle_time
        return_per_episode = get_stats(logs["return_per_episode"])
        success_per_episode = get_stats(
            [1 if r > 0 else 0 for r in logs["return_per_episode"]])
        num_frames_per_episode = get_stats(logs["num_frames_per_episode"])
        data = [
            self.cycle,
            self.num_episodes,
            self.num_frames,
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

        format_str = ("U {} | E {} | F {:6} | FPS {:3.0f} | D {:1.2f} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                      "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                      "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")

        if self.cycle % self.print_every == 0:
            self.logger.info(format_str.format(*data))
        if self.cycle % self.save_every == 0:
            check = RLCheckpoint(
                self.trainer.model,
                self.trainer.optimizer,
                {'i': self.cycle, 'num_frames': self.num_frames,
                    'num_episodes': self.num_episodes},
                self.trainer.preprocess_obss,
                self.trainer.model_path
            )
            check.save()
            self.trainer.preprocess_obss.vocab.save()
        if self.writer is not None:
            self.writer.add_scalar('train/fps', fps, self.cycle)
            self.writer.add_scalar(
                'train/succes_rate', success_per_episode['mean'], self.cycle)
            self.writer.add_scalar(
                'train/episode_length', num_frames_per_episode['mean'], self.cycle)
            self.writer.add_scalar(
                'train/disrupt', logs["disrupts"], self.cycle)
            self.writer.add_scalar('train/loss', logs["loss"], self.cycle)
            self.writer.add_scalar('train/policy_loss',
                                   logs["policy_loss"], self.cycle)
            self.writer.add_scalar(
                'train/value_loss', logs["value_loss"], self.cycle)
            if self.reasoning:
                self.writer.add_scalar(
                    'train/reason_correct_rate', get_stats(logs['correct_reasons'])['mean'], self.cycle)
                self.writer.add_scalar(
                    'train/reason_loss', logs['reason_loss'], self.cycle)


def get_stats(arr):
    import collections
    d = collections.OrderedDict()
    d["mean"] = np.mean(arr)
    d["std"] = np.std(arr)
    d["min"] = np.amin(arr)
    d["max"] = np.amax(arr)
    return d
