import abc
import torch
import numpy as np

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
        # self.keep_words = [self.vocab[k] for k in self.mapping]

    def __call__(self, instruction):
        print("vocab", self.vocab.vocab)
        kept = np.isin(instruction, list(self.mapping.keys())).astype(np.uint8)
        idx = instruction[torch.from_numpy(kept)]
        return torch.Tensor([self.mapping[id.item()] for id in idx])
        # return [self.idx_mapping[] for id in idx]
        # if self.vocab['red'] in instruction:
        #     return self.mapping['red']
        # elif self.vocab['green'] in instruction:
        #     return self.mapping['green']
        # elif self.vocab['blue'] in instruction:
        #     return self.mapping['blue']
        # elif self.vocab['purple'] in instruction:
        #     return self.mapping['purple']
        # elif self.vocab['yellow'] in instruction:
        #     return self.mapping['yellow']
        # elif self.vocab['grey'] in instruction:
        #     return self.mapping['grey']


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