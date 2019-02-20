from abc import ABC, abstractmethod


class BaseCheckpoint(ABC):
    TRAINER_STATE_NAME = 'trainer_states.pt'
    MODEL_NAME = 'model.pt'
    INPUT_VOCAB_FILE = 'input_vocab.pt'
    OUTPUT_VOCAB_FILE = 'output_vocab.pt'

    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def path(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @classmethod
    @abstractmethod
    def load(self):
        pass