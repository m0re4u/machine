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
            in_channels=in_channels, out_channels=imm_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels, out_channels=out_features, kernel_size=(3, 3), padding=1)
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


class ACModel(BaseModel):
    """
    Model taken from babyAI repo
    """

    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, instr_dim=128,
                 use_instr=False, lang_model="gru", use_memory=False,
                 arch="cnn1",):
        super().__init__()

        # Decide which components are enabled
        self.use_instr = use_instr
        self.use_memory = use_memory
        self.arch = arch
        self.lang_model = lang_model
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim

        self.obs_space = obs_space
        self.action_space = action_space

        if arch == "cnn1":
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=image_dim,
                          kernel_size=(2, 2)),
                nn.ReLU()
            )
        elif arch.startswith("expert_filmcnn"):
            if not self.use_instr:
                raise ValueError(
                    "FiLM architecture can be used when instructions are enabled")

            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=128,
                          kernel_size=(2, 2), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=128, out_channels=128,
                          kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )
            self.film_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        else:
            raise ValueError("Incorrect architecture name: {}".format(arch))

        # Define instruction embedding
        if self.use_instr:
            if self.lang_model in ['gru', 'bigru', 'attgru']:
                self.word_embedding = nn.Embedding(
                    obs_space["instr"], self.instr_dim)
                if self.lang_model in ['gru', 'bigru', 'attgru']:
                    gru_dim = self.instr_dim
                    if self.lang_model in ['bigru', 'attgru']:
                        gru_dim //= 2
                    self.instr_rnn = nn.GRU(
                        self.instr_dim, gru_dim, batch_first=True,
                        bidirectional=(self.lang_model in ['bigru', 'attgru']))
                    self.final_instr_dim = self.instr_dim
                else:
                    kernel_dim = 64
                    kernel_sizes = [3, 4]
                    self.instr_convs = nn.ModuleList([
                        nn.Conv2d(1, kernel_dim, (K, self.instr_dim)) for K in kernel_sizes])
                    self.final_instr_dim = kernel_dim * len(kernel_sizes)

            if self.lang_model == 'attgru':
                self.memory2key = nn.Linear(
                    self.memory_size, self.final_instr_dim)

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_instr and not "filmcnn" in arch:
            self.embedding_size += self.final_instr_dim

        if arch.startswith("expert_filmcnn"):
            if arch == "expert_filmcnn":
                num_module = 2
            else:
                num_module = int(arch[(arch.rfind('_') + 1):])
            self.controllers = []
            for ni in range(num_module):
                if ni < num_module-1:
                    mod = FiLMedBlock(
                        in_features=self.final_instr_dim,
                        out_features=128, in_channels=128, imm_channels=128)
                else:
                    mod = FiLMedBlock(
                        in_features=self.final_instr_dim, out_features=self.image_dim,
                        in_channels=128, imm_channels=128)
                self.controllers.append(mod)
                self.add_module('FiLM_Controler_' + str(ni), mod)

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

        # Initialize parameters correctly
        self.reset_parameters()
        self.embedding = None

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
            self.instr_dim,
            self.use_instr,
            self.lang_model,
            self.use_memory,
            self.arch
        ]

    def reset_parameters(self):
        return self.apply(initialize_parameters)

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

        print(x.size())
        if self.arch.startswith("expert_filmcnn"):
            x = self.image_conv(x)
            for controler in self.controllers:
                x = controler(x, instr_embedding)
            x = F.relu(self.film_pool(x))
        else:
            x = self.image_conv(x)
        print(x.size())

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

        return {'dist': dist, 'value': value, 'memory': memory}

    def _get_instr_embedding(self, instr):
        if self.lang_model == 'gru':
            _, hidden = self.instr_rnn(self.word_embedding(instr))
            return hidden[-1]

        elif self.lang_model in ['bigru', 'attgru']:
            lengths = (instr != 0).sum(1).long()
            masks = (instr != 0).float()

            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if instr.is_cuda:
                    iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                inputs = self.word_embedding(instr)
                inputs = inputs[perm_idx]

                inputs = pack_padded_sequence(
                    inputs, seq_lengths.data.cpu().numpy(), batch_first=True)

                outputs, final_states = self.instr_rnn(inputs)
            else:
                instr = instr[:, 0:lengths[0]]
                outputs, final_states = self.instr_rnn(
                    self.word_embedding(instr))
                iperm_idx = None
            final_states = final_states.transpose(0, 1).contiguous()
            final_states = final_states.view(final_states.shape[0], -1)
            if iperm_idx is not None:
                outputs, _ = pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[iperm_idx]
                final_states = final_states[iperm_idx]

            if outputs.shape[1] < masks.shape[1]:
                masks = masks[:, :(outputs.shape[1]-masks.shape[1])]
                # the packing truncated the original length
                # so we need to change mask to fit it

            return outputs if self.lang_model == 'attgru' else final_states

        else:
            ValueError(
                "Undefined instruction architecture: {}".format(self.use_instr))
