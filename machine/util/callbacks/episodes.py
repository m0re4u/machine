import logging
import os
import time

import numpy as np

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

    def __init__(self, print_every=10, save_every=10, model_name='', use_tensorboard=False, explore_for=100):
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

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, info=None):
        self.logger.info(f"Start of epoch: {info}")
        self.log_entropies = []
        self.log_values = []
        self.log_policy_losses = []
        self.log_value_losses = []
        self.log_grad_norms = []
        self.log_losses = []
        self.log_disrupts = []

    def on_epoch_end(self, logs=None, info=None):
        if logs is not None:
            # Log some values
            logs["entropy"] = np.mean(self.log_entropies)
            logs["value"] = np.mean(self.log_values)
            logs["policy_loss"] = np.mean(self.log_policy_losses)
            logs["value_loss"] = np.mean(self.log_value_losses)
            logs["grad_norm"] = np.mean(self.log_grad_norms)
            logs["loss"] = np.mean(self.log_losses)
            logs["disrupts"] = np.mean(self.log_disrupts)
            return logs
        else:
            pass

    def on_batch_begin(self, batch, info=None):
        batch_entropy = 0
        batch_value = 0
        batch_policy_loss = 0
        batch_value_loss = 0
        batch_loss = 0
        batch_disrupt = 0
        return {
            'entropy': batch_entropy,
            'value': batch_value,
            'policy_loss': batch_policy_loss,
            'value_loss': batch_value_loss,
            'loss': batch_loss,
            'disrupt': batch_disrupt
        }

    def on_batch_end(self, loss, info=None):
        self.log_losses.append(loss.item())
        self.log_entropies.append(info['entropy'])
        self.log_values.append(info['value'])
        self.log_policy_losses.append(info['policy_loss'])
        self.log_value_losses.append(info['value_loss'])
        self.log_disrupts.append(info['disrupt'].item())
        self.log_grad_norms.append(info['grad_norm'].item())

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
        self.logger.info(f"Start of cycle: {self.cycle} - Reward from: {reward_origin}")

    def on_cycle_end(self, logs):
        self.num_frames += logs['num_frames']
        self.num_episodes += logs['episodes_done']
        self.cycle += 1
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
                {'i':self.cycle, 'num_frames':self.num_frames, 'num_episodes':self.num_episodes},
                self.trainer.preprocess_obss,
                self.trainer.model_path
            )
            check.save()
            self.trainer.preprocess_obss.vocab.save()
        if self.writer is not None:
            self.writer.add_scalar('train/fps', fps, self.cycle)
            self.writer.add_scalar(
                'train/succes_rate', success_per_episode['mean'], self.cycle)
            self.writer.add_scalar('train/episode_length', num_frames_per_episode['mean'], self.cycle)
            self.writer.add_scalar('train/disrupt', logs["disrupts"], self.cycle)
            self.writer.add_scalar('train/loss', logs["loss"], self.cycle)
            self.writer.add_scalar('train/policy_loss', logs["policy_loss"], self.cycle)
            self.writer.add_scalar('train/value_loss', logs["value_loss"], self.cycle)


def get_stats(arr):
    import collections
    d = collections.OrderedDict()
    d["mean"] = np.mean(arr)
    d["std"] = np.std(arr)
    d["min"] = np.amin(arr)
    d["max"] = np.amax(arr)
    return d
