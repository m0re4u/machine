import logging
import os

import numpy
import torch

from machine.util.callbacks import EpisodeLogger
from machine.models import SoftmaxPolicy, SigmoidTermination
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OptionTrainer(object):
    """
    The ReinforcementTrainer class helps in setting up a training framework for
    reinforcement learning.

    Largely inspired by babyAI repo code for PPOAlgo
    """

    def __init__(self, envs, opt, obs):
        self._trainer = f"Option Trainer"
        self.env = envs
        self.preprocess_obss = obs
        self.logger = logging.getLogger(__name__)

        # Mapping from instruction to particular options
        self.policy_over_options = None

        # Intra option policies
        # self.intra_policies = [SoftmaxPolicy() for _ in range(opt.n_options)]
        self.intra_policies = None
        # Termination probability per option
        # self.termination = [SigmoidTermination() for _ in range(opt.n_options)]
        self.termination = None

        # Overall critic
        self.critic = None

        # Arguments
        self.frames = opt.frames

        # Initialize callbacks
        self.callback = EpisodeLogger(
            opt.print_every, opt.save_every, "", opt.tb, opt.explore_for)
        self.callback.set_trainer(self)

    def train(self):
        """
        Perform a series on training steps as configured.
        """
        # Start training model
        self.callback.on_train_begin()
        num_frames = 0
        while num_frames < self.frames:
            self.callback.on_cycle_start()
            num_frames += 1
            self.callback.on_cycle_end(None)

        self.callback.on_train_end()
