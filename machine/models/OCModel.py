import torch
import torch.nn as nn

from machine.models import ACModel


class OCModel(ACModel):
    """
    """

    def __init__(self, obs_space, action_space, image_dim=128, memory_dim=128,
        instr_dim=128, use_instr=False, lang_model="gru", use_memory=False,
        arch="cnn1"):
        """
        Initialize the Option-Critic model
        """
        super().__init__()

        # Number of options
        option_space = 2

        # option_policy should choose option o in state s
        self.option_policy = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, option_space)
        )

        # Determine if we want to terminate the execution of the current option
        self.termination = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )