import abc
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseMapping(abc.ABC):
    @abc.abstractmethod
    def __call__(self):
        pass


class CommandMapping(BaseMapping):
    def __init__(self, vocab):
        self.vocab = vocab
        self.mapping = {
            'go': 0,
            'pick': 1,
            'open': 2,
            'put': 3
        }

    def __call__(self, instruction):
        if self.vocab['go'] in instruction:
            return self.mapping['go']
        elif self.vocab['pick'] in instruction:
            return self.mapping['pick']
        elif self.vocab['open'] in instruction:
            return self.mapping['open']
        elif self.vocab['put'] in instruction:
            return self.mapping['put']


class ColorMapping(BaseMapping):
    def __init__(self, vocab):
        self.vocab = vocab
        self.mapping = {
            self.vocab['red']: 0,
            self.vocab['green']: 1,
            self.vocab['blue']: 2,
            self.vocab['purple']: 3,
            self.vocab['yellow']: 4,
            self.vocab['grey']: 5
        }

    def __call__(self, instruction):
        kept = np.isin(instruction.cpu(), list(self.mapping.keys())).astype(np.uint8)
        idx = instruction[torch.from_numpy(kept)]
        x = torch.tensor([self.mapping[id.item()] for id in idx], device=device)
        return x


class ObjectMapping(BaseMapping):
    def __init__(self, vocab):
        self.vocab = vocab
        self.mapping = {
            'key': 0,
            'ball': 1,
            'box': 2,
            'door': 3
        }

    def __call__(self, instruction):
        if self.vocab['key'] in instruction:
            return self.mapping['key']
        elif self.vocab['ball'] in instruction:
            return self.mapping['ball']
        elif self.vocab['box'] in instruction:
            return self.mapping['box']
        elif self.vocab['door'] in instruction:
            return self.mapping['door']

class RandomMapping(BaseMapping):
    def __init__(self, max_num=6, seed=0):
        self.max_num = max_num

    def __call__(self, instruction):
        return torch.randint(self.max_num, size=(64,))

class ConstantMapping(BaseMapping):
    def __call__(self, instruction):
        return torch.zeros((64,))
