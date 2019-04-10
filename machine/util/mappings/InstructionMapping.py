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
            self.vocab['go']: 0,
            self.vocab['pick']: 1,
            self.vocab['open']: 2,
            self.vocab['put']: 3
        }

    def __call__(self, instruction):
        kept = np.isin(instruction.cpu(), list(self.mapping.keys())).astype(np.uint8)
        idx = instruction[torch.from_numpy(kept)]
        x = torch.tensor([self.mapping[id.item()] for id in idx], device=device)
        return x


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
            self.vocab['key']: 0,
            self.vocab['ball']: 1,
            self.vocab['box']: 2,
            self.vocab['door']: 3
        }

    def __call__(self, instruction):
        kept = np.isin(instruction.cpu(), list(self.mapping.keys())).astype(np.uint8)
        idx = instruction[torch.from_numpy(kept)]
        x = torch.tensor([self.mapping[id.item()] for id in idx], device=device)
        return x


class RandomMapping(BaseMapping):
    def __init__(self, max_num=6, seed=0):
        self.max_num = max_num

    def __call__(self, instruction):
        return torch.randint(self.max_num, size=(64,))

class ConstantMapping(BaseMapping):
    def __call__(self, instruction):
        return torch.zeros((64,))
