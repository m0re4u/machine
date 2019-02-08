
import logging
import os

from collections import defaultdict
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

    def __init__(self):
        super(EpisodeLogger, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.print_loss_total = defaultdict(float)  # Reset every print_every
        self.epoch_loss_total = defaultdict(float)  # Reset every epoch
        self.epoch_loss_avg = defaultdict(float)
        self.print_loss_avg = defaultdict(float)

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, info=None):
        self.logger.info(f"Episode: {info['episode']}, Step: {info['step']}")

    def on_epoch_end(self, info=None):
        self.logger.info("End of episode")

    def on_batch_begin(self, batch, info=None):
        self.logger.info("Start of step")

    def on_batch_end(self, batch, info=None):
        self.logger.info("End of step")

    def on_train_begin(self, info=None):
        self.logger.info("Starting training")

    def on_train_end(self, info=None):
        self.logger.info("Finished training")
