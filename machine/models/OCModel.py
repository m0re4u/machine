import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

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
        super().__init__(obs_space, action_space, image_dim, memory_dim, instr_dim, use_instr, lang_model, use_memory, arch)

        # Number of options
        option_space = 4

        # option_policy should choose option o in state s
        self.option_policy = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, option_space),
            nn.Softmax()
        )

        # Determine if we want to terminate the execution of the current option
        self.termination = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

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
                x = nn.ReLU(self.film_pool(x))
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

            # intra policy
            x = self.actor(embedding)
            dist = Categorical(logits=F.log_softmax(x, dim=1))

            # Value function
            x = self.critic(embedding)
            value = x.squeeze(1)

            # Termination
            term = self.termination(embedding)

            # Option policy
            option = self.option_policy(embedding)

            return {'dist': dist, 'value': value, 'memory': memory}
