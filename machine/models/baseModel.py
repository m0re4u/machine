import abc
import sys
import types

import torch.nn as nn
import torch.nn.functional as F

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(types.StringType('ABC'), (), {})


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for models.
    """

    def __init__(self):
        super(BaseModel, self).__init__()

    @abc.abstractmethod
    def reset_parameters(self):
        """
        Reset the parameters of all components in the model.
        """
        pass

    @abc.abstractmethod
    def forward(self, inputs):
        """
        Inputs: inputs, input_lengths, targets, teacher_forcing_ratio
            - **inputs** (list, option): list of sequences, whose length is the batch size and within which
              each sequence is a list of token IDs. This information is passed to the encoder module.

        Outputs: decoder_outputs, decoder_hidden, ret_dict
            - **outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
              outputs of the decoder.
        """
        pass

    @property
    @abc.abstractmethod
    def model_hyperparameters(self):
        pass