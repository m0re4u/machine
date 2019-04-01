from __future__ import print_function

import logging
import os
import shutil
import time

import dill
import torch
from torch.nn import DataParallel

from machine.models import ACModel

from .base_checkpoint import BaseCheckpoint


class Checkpoint(BaseCheckpoint):
    """
    The Checkpoint class manages the saving and loading of a model during training. It allows training to be suspended
    and resumed at a later time (e.g. when running on a cluster using sequential jobs).

    To make a checkpoint, initialize a Checkpoint object with the following args; then call that object's save() method
    to write parameters to disk.

    Args:
        model (machine.Seq2seq): seq2seq model being trained
        optimizer (Optimizer): stores the state of the optimizer
        epoch (int): current epoch (an epoch is a loop through the full training data)
        step (int): number of examples seen within the current epoch
        input_vocab (Vocabulary): vocabulary for the input language
        output_vocab (Vocabulary): vocabulary for the output language

    Attributes:
        TRAINER_STATE_NAME (str): name of the file storing trainer states
        MODEL_NAME (str): name of the file storing model
        INPUT_VOCAB_FILE (str): name of the input vocab file
        OUTPUT_VOCAB_FILE (str): name of the output vocab file
    """

    def __init__(self, model, optimizer, epoch, step,
                 input_vocab, output_vocab, path=None):
        self.model = model
        self.optimizer = optimizer
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.epoch = epoch
        self.step = step
        self._path = path

    @property
    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved.")
        return self._path

    def save(self, experiment_dir, name=None):
        """
        Saves the current model and related training parameters into a subdirectory of the checkpoint directory.
        By default, the name of the subdirectory is the current local time in Y_M_D_H_M_S format, optionally a variable name can be passed to give the checkpoint a different name.
        Args:
            experiment_dir (str): path to the experiment root directory
            name (str): alternative name for the model
        Returns:
             str: path to the saved checkpoint subdirectory
        """
        name = name or time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

        self._path = os.path.join(experiment_dir, name)
        path = self._path

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        torch.save({'epoch': self.epoch,
                    'step': self.step,
                    'optimizer': self.optimizer
                    },
                   os.path.join(path, self.TRAINER_STATE_NAME))

        if isinstance(self.model, DataParallel):
            torch.save(self.model.module, os.path.join(path, self.MODEL_NAME))
        else:
            torch.save(self.model, os.path.join(path, self.MODEL_NAME))

        with open(os.path.join(path, self.INPUT_VOCAB_FILE), 'wb') as fout:
            dill.dump(self.input_vocab, fout)
        with open(os.path.join(path, self.OUTPUT_VOCAB_FILE), 'wb') as fout:
            dill.dump(self.output_vocab, fout)

        return path

    @classmethod
    def load(cls, path):
        """
        Loads a Checkpoint object that was previously saved to disk.
        Args:
            path (str): path to the checkpoint subdirectory
        Returns:
            checkpoint (Checkpoint): checkpoint object with fields copied from those stored on disk
        """
        logger = logging.getLogger(__name__)
        logger.info("Loading checkpoints from {}".format(path))

        if torch.cuda.is_available():
            resume_checkpoint = torch.load(
                os.path.join(path, cls.TRAINER_STATE_NAME))
            model = torch.load(os.path.join(path, cls.MODEL_NAME))
        else:
            resume_checkpoint = torch.load(os.path.join(
                path, cls.TRAINER_STATE_NAME), map_location=lambda storage, loc: storage)
            model = torch.load(os.path.join(path, cls.MODEL_NAME),
                               map_location=lambda storage, loc: storage)

        model.flatten_parameters()  # make RNN parameters contiguous
        with open(os.path.join(path, cls.INPUT_VOCAB_FILE), 'rb') as fin:
            input_vocab = dill.load(fin)
        with open(os.path.join(path, cls.OUTPUT_VOCAB_FILE), 'rb') as fin:
            output_vocab = dill.load(fin)
        optimizer = resume_checkpoint['optimizer']
        return Checkpoint(model=model,
                          epoch=resume_checkpoint['epoch'],
                          step=resume_checkpoint['step'],
                          input_vocab=input_vocab,
                          output_vocab=output_vocab,
                          optimizer=optimizer,
                          path=path)



class RLCheckpoint(BaseCheckpoint):
    CHECKPOINT_NAME='check.pt'

    def __init__(self, model, optimizer, status, obs, path=None):
        self.model = model
        self.optimizer = optimizer
        self.status = status
        self.obs = obs
        if path is None:
            # some default
            pass
        else:
            self._path = path

    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved.")
        return self._path

    def save(self):
        logger = logging.getLogger(__name__)

        name = f"{self.status['i']:06}_{self.CHECKPOINT_NAME}"
        path = os.path.join(self._path, name)
        state = {
            'i': self.status['i'],
            'num_episodes': self.status['num_episodes'],
            'num_frames': self.status['num_frames'],
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.state_dict(),
            'model_params': {
                'obs_space': self.model.obs_space,
                'action_space': self.model.action_space,
                'image_dim': self.model.image_dim,
                'memory_dim': self.model.memory_dim,
                'instr_dim': self.model.instr_dim,
                'use_instr': self.model.use_instr,
                'lang_model': self.model.lang_model,
                'use_memory': self.model.use_memory,
                'arch': self.model.arch,
            }
        }
        logger.info(f"Saving RLCheckpoint to {path}")
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        """
        Load entire training state to resume training
        """
        pass

    @classmethod
    def load_model(cls, path):
        """
        Only load a model to perform some evaluation runs
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Loading RLCheckpoint from {path}")
        state = torch.load(path, map_location=torch.device('cpu'))
        params = [
            state['model_params']['obs_space'],
            state['model_params']['action_space'],
            state['model_params']['image_dim'],
            state['model_params']['memory_dim'],
            state['model_params']['instr_dim'],
            state['model_params']['use_instr'],
            state['model_params']['lang_model'],
            state['model_params']['use_memory'],
            state['model_params']['arch'],
        ]
        model = ACModel(*params)
        model.load_state_dict(state['model'])
        print(model)
        model.eval()
        return model
