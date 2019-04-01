import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from machine.models import BaseModel
from machine.util.mappings import CommandMapping, ColorMapping, ObjectMapping

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / \
            torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class SkillEmbedding(BaseModel):
    """
    """

    def __init__(self, input_size, action_space, n_skills, vocab, embedding_dim=32, memory_dim=128, use_memory=False, num_procs=64):
        super().__init__()
        self.n_skills = n_skills
        self.num_procs = num_procs
        self.use_memory = use_memory
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim
        self.skill_embeddings = []
        for i in range(n_skills):
            self.skill_embeddings.append(nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.embedding_dim)
            ))

        # Define actor's model
        self.policy = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Instruction mapping
        self.instr_mapping = ColorMapping(vocab)

    def forward(self, obs, memory):
        skill_idx = self.instr_mapping(obs.instr)
        h = torch.zeros((64, 128))
        for i in range(6):
            mask = (skill_idx == i)
            print(mask.size())
            aa = obs.image[mask]
            if aa.shape[0] == 0:
                continue
            print(aa.size())
            aa = aa.reshape(aa.shape[0], -1)
            h[mask] = self.skill_embeddings[i](aa)

        act = self.policy(h)
        dist = Categorical(logits=F.log_softmax(act, dim=1))

        crit = self.critic(h)
        value = crit.squeeze(1)

        return {'dist': dist, 'value': value, 'memory': memory}

    def reset_parameters(self):
        return self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

