import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from machine.models import ACModel


class IACModel(ACModel):
    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, instr_dim=128,
                 use_instr=False, lang_model="gru", use_memory=False,
                 arch="cnn1"):

        super().__init__(obs_space, action_space, image_dim, memory_dim,
                         instr_dim, use_instr, lang_model, use_memory, arch)
        self.reasoning = nn.Sequential(
            nn.Linear(self.embedding_size, 18),
            nn.ReLU(),
            nn.LogSoftmax(dim=1)
        )
        self.embedding = None

    @property
    def model_hyperparameters(self):
        return [
            self.obs_space,
            self.action_space,
            self.image_dim,
            self.memory_dim,
            self.instr_dim,
            self.use_instr,
            self.lang_model,
            self.use_memory,
            self.arch,
        ]

    def forward(self, obs, memory, instr_embedding=None):
        if self.use_instr and instr_embedding is None:
            instr_embedding = self._get_instr_embedding(obs.instr)
        if self.use_instr and self.lang_model == "attgru":
            # outputs: B x L x D
            # memory: B x M
            mask = (obs.instr != 0).float()
            instr_embedding = instr_embedding[:, :mask.shape[1]]
            keys = self.memory2key(memory)
            pre_softmax = (keys[:, None, :] *
                           instr_embedding).sum(2) + 1000 * mask
            attention = F.softmax(pre_softmax, dim=1)
            instr_embedding = (instr_embedding * attention[:, :, None]).sum(1)

        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)

        if self.arch.startswith("expert_filmcnn"):
            x = self.image_conv(x)
            for controler in self.controllers:
                x = controler(x, instr_embedding)
            x = F.relu(self.film_pool(x))
        else:
            x = self.image_conv(x)

        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size],
                      memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_instr and not "filmcnn" in self.arch:
            embedding = torch.cat((embedding, instr_embedding), dim=1)

        self.embedding = embedding
        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        reason = self.reasoning(embedding)

        return {'dist': dist, 'value': value, 'memory': memory, 'reason': reason}
