import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from machine.models import BaseModel



# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / \
            torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLMedBlock(BaseModel):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_channels = in_channels
        self.imm_channels = imm_channels
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=imm_channels, kernel_size=(3, 3), padding=0)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels, out_channels=out_features, kernel_size=(3, 3), padding=0)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.reset_parameters()

    def reset_parameters(self):
        return self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        out = x * self.weight(y).unsqueeze(2).unsqueeze(3) + \
            self.bias(y).unsqueeze(2).unsqueeze(3)
        out = self.bn2(out)
        out = F.relu(out)
        return out

    def model_hyperparameters(self):
        return [
            self.in_features,
            self.out_features,
            self.in_channels,
            self.imm_channels
        ]


class MinModel(BaseModel):
    """
    """

    def __init__(self, obs_space, action_space, image_dim=128, memory_dim=128, instr_dim=128, diag=True, diag_targets=18):
        super().__init__()

        # Decide which components are enabled
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim

        self.obs_space = obs_space
        self.action_space = action_space



        self.film_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # Define instruction embedding
        self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
        gru_dim = self.instr_dim
        self.instr_rnn = nn.GRU(self.instr_dim, gru_dim, batch_first=True, bidirectional=False)
        self.final_instr_dim = self.instr_dim

        # memory
        self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size

        self.film = FiLMedBlock(
            in_features=self.image_dim, out_features=128, in_channels=3, imm_channels=128
        )

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.embedding = None

        if diag:
            self.diag_targets = diag_targets
            self.reasoning = nn.Sequential(
                nn.Linear(self.embedding_size, self.diag_targets),
            # nn.ReLU(),
            # nn.Linear(32, 18)
            )

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    @property
    def model_hyperparameters(self):
        return [
            self.obs_space,
            self.action_space,
            self.image_dim,
            self.memory_dim,
            self.instr_dim
        ]

    def reset_parameters(self):
        pass

    def forward(self, obs, memory, instr_embedding=None):
        instr_embedding = self._get_instr_embedding(obs.instr)
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        x = self.film(x, instr_embedding)
        x = F.relu(self.film_pool(x))

        x = x.reshape(x.shape[0], -1)
        hidden = (memory[:, :self.semi_memory_size],
                    memory[:, self.semi_memory_size:])
        hidden = self.memory_rnn(x, hidden)
        embedding = hidden[0]
        memory = torch.cat(hidden, dim=1)

        self.embedding = embedding
        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        reason = self.reasoning(embedding.detach())


        return {'dist': dist, 'value': value, 'memory': memory, 'reason': reason}

    def _get_instr_embedding(self, instr):
        _, hidden = self.instr_rnn(self.word_embedding(instr))
        return hidden[-1]