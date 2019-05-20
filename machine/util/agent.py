from abc import ABC, abstractmethod
from random import Random

import torch

import babyai
from machine.util import RLCheckpoint


class Agent(ABC):
    """An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def on_reset(self):
        pass

    @abstractmethod
    def act(self, obs):
        """Propose an action based on observation.

        Returns a dict, with 'action` entry containing the proposed action,
        and optionaly other entries containing auxiliary information
        (e.g. value function).

        """
        pass

    @abstractmethod
    def analyze_feedback(self, reward, done):
        pass


class ModelAgent(Agent):
    """A model-based agent. This agent behaves using a model."""

    def __init__(self, model_or_name, obss_preprocessor, argmax, partial=False):
        self.partial = partial
        if obss_preprocessor is None:
            assert isinstance(model_or_name, str)
            obss_preprocessor = babyai.utils.ObssPreprocessor(model_or_name)
        self.obss_preprocessor = obss_preprocessor
        if isinstance(model_or_name, str):
            if self.partial:
                self.model = RLCheckpoint.load_partial_model(model_or_name)
            else:
                self.model = RLCheckpoint.load_model(model_or_name)
            if torch.cuda.is_available():
                self.model.cuda()
        else:
            self.model = model_or_name
        self.device = next(self.model.parameters()).device
        self.argmax = argmax
        self.memory = None

    def act_batch(self, many_obs):
        if self.memory is None:
            self.memory = torch.zeros(
                len(many_obs), self.model.memory_size, device=self.device)
        elif self.memory.shape[0] != len(many_obs):
            raise ValueError(
                "stick to one batch size for the lifetime of an agent")
        preprocessed_obs = self.obss_preprocessor(many_obs, device=self.device)

        with torch.no_grad():
            model_results = self.model(preprocessed_obs, self.memory)
            dist = model_results['dist']
            value = model_results['value']
            if self.partial:
                reason = model_results['reason']
            self.memory = model_results['memory']

        if self.argmax:
            action = dist.probs.max(1, keepdim=True)[1]
        else:
            action = dist.sample()

        if self.partial:
            return {'action': action,
                    'dist': dist,
                    'value': value,
                    'reason': reason}
        else:
            return {'action': action,
                    'dist': dist,
                    'value': value}

    def act(self, obs):
        return self.act_batch([obs])

    def analyze_feedback(self, reward, done):
        if isinstance(done, tuple):
            for i in range(len(done)):
                if done[i]:
                    self.memory[i, :] *= 0.
        else:
            self.memory *= (1 - done)


class RandomAgent:
    """A newly initialized model-based agent."""

    def __init__(self, seed=0, number_of_actions=7):
        self.rng = Random(seed)
        self.number_of_actions = number_of_actions

    def act(self, obs):
        action = self.rng.randint(0, self.number_of_actions - 1)
        # To be consistent with how a ModelAgent's output of `act`:
        return {'action': torch.tensor(action),
                'dist': None,
                'value': None}


def load_agent(env, model_name, argmax=True, env_name=None, vocab=None, partial=False):
    # env_name needs to be specified for demo agents
    obss_preprocessor = babyai.utils.ObssPreprocessor(
        model_name, env.observation_space, load_vocab_from=vocab)
    return ModelAgent(model_name, obss_preprocessor, argmax, partial=partial)
