import datetime
import logging
import time

import numpy
import torch

import machine
from babyai.rl.utils import DictList
from machine.util.callbacks import EpisodeLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReinforcementTrainer(object):
    """
    The ReinforcementTrainer class helps in setting up a training framework for
    reinforcement learning.

    Largely inspired by babyAI repo code for PPOAlgo
    """

    def __init__(self, envs, opt, model, obs, reshape_reward, algo_name='ppo'):
        self._trainer = "Reinforcement Trainer"
        self._algo = algo_name
        self.env = envs
        self.model = model
        self.model.train()
        self.preprocess_obss = obs
        self.reshape_reward = reshape_reward
        self.logger = logging.getLogger(__name__)

        # Copy command-line arguments to class
        self.frames = opt.frames
        self.frames_per_proc = opt.frames_per_proc
        self.num_procs = opt.num_processes
        self.discount = opt.gamma
        self.lr = opt.lr
        self.gae_lambda = opt.gae_lambda
        self.recurrence = opt.recurrence
        self.batch_size = opt.batch_size
        self.clip_eps = opt.clip_eps
        self.entropy_coef = opt.entropy_coef
        self.value_loss_coef = opt.value_loss_coef
        self.max_grad_norm = opt.max_grad_norm

        self.num_frames = self.frames_per_proc * self.num_procs
        assert self.frames_per_proc % opt.recurrence == 0

        # Initialize experience matrices
        shape = (self.frames_per_proc, self.num_procs)
        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])

        self.memory = torch.zeros(shape[1], self.model.memory_size, device=device)
        self.memories = torch.zeros(*shape, self.model.memory_size, device=device)

        self.mask = torch.ones(shape[1], device=device)
        self.masks = torch.zeros(*shape, device=device)
        self.actions = torch.zeros(*shape, device=device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=device)
        self.rewards = torch.zeros(*shape, device=device)
        self.advantages = torch.zeros(*shape, device=device)
        self.log_probs = torch.zeros(*shape, device=device)

        # Initialize log variables
        self.log_episode_return = torch.zeros(self.num_procs, device=device)
        self.log_episode_reshaped_return = torch.zeros(
            self.num_procs, device=device)
        self.log_episode_num_frames = torch.zeros(
            self.num_procs, device=device)

        self.callback = EpisodeLogger(opt.tb)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

        if algo_name == 'ppo':
            assert opt.batch_size % opt.recurrence == 0
            self.optimizer = torch.optim.Adam(self.model.parameters(
            ), self.lr, (opt.beta1, opt.beta2), eps=opt.optim_eps)
            self.batch_num = 0
            self.epochs = opt.ppo_epochs

    def collect_experiences(self):
        """
        Collect actions, observations and rewards over multiple concurrent
        environments.

        Taken from babyAI repo
        """
        for i in range(self.frames_per_proc):
            # Do one agent-environment interaction
            preprocessed_obs = self.preprocess_obss(self.obs, device=device)
            with torch.no_grad():
                model_results = self.model(
                    preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                dist = model_results['dist']
                value = model_results['value']
                memory = model_results['memory']
                extra_predictions = model_results['extra_predictions']

            action = dist.sample()

            obs, reward, done, env_info = self.env.step(action.cpu().numpy())

            # Update experiences values
            self.obss[i] = self.obs
            self.obs = obs

            self.memories[i] = self.memory
            self.memory = memory

            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=device)
            else:
                self.rewards[i] = torch.tensor(reward, device=device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values
            self.log_episode_return += torch.tensor(reward, device=device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(
                        self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(
                        self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences
        preprocessed_obs = self.preprocess_obss(self.obs, device=device)
        with torch.no_grad():
            next_value = self.model(
                preprocessed_obs, self.memory * self.mask.unsqueeze(1))['value']

        for i in reversed(range(self.frames_per_proc)):
            next_mask = self.masks[i +
                                   1] if i < self.frames_per_proc - 1 else self.mask
            next_value = self.values[i +
                                     1] if i < self.frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i +
                                             1] if i < self.frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * \
                next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * \
                self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.frames_per_proc)]
        # In commments below T is self.frames_per_proc, P is self.num_procs,
        # D is the dimensionality

        # T x P x D -> P x T x D -> (P * T) x D
        exps.memory = self.memories.transpose(
            0, 1).reshape(-1, *self.memories.shape[2:])
        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences
        exps.obs = self.preprocess_obss(exps.obs, device=device)

        # Log some values
        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "episodes_done": self.log_done_counter,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, log

    def update_model_parameters(self, exps, logs):
        """
        Perform gradient update on the model using the gathered experience.

        Taken from babyAI repo
        """
        e_i = 0
        for _ in range(self.epochs):
            self.callback.on_epoch_begin(e_i)
            # Initialize log values
            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []
            log_losses = []

            for inds in self._get_batches_starting_indexes():
                self.callback.on_batch_begin(None)
                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]

                    # Compute loss
                    model_results = self.model(sb.obs, memory * sb.mask)
                    dist = model_results['dist']
                    value = model_results['value']
                    memory = model_results['memory']
                    extra_predictions = model_results['extra_predictions']

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(
                        sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + \
                        torch.clamp(value - sb.value, -
                                    self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * \
                        entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values
                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic
                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(
                    2) ** 2 for p in self.model.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values
                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm.item())
                log_losses.append(batch_loss.item())
                self.callback.on_batch_end(None)

            # Log some values
            logs["entropy"] = numpy.mean(log_entropies)
            logs["value"] = numpy.mean(log_values)
            logs["policy_loss"] = numpy.mean(log_policy_losses)
            logs["value_loss"] = numpy.mean(log_value_losses)
            logs["grad_norm"] = numpy.mean(log_grad_norms)
            logs["loss"] = numpy.mean(log_losses)
            e_i += 1
            self.callback.on_epoch_end()
        return logs

    def train(self):
        """
        Perfor a series on training steps as configured.
        """
        # Start training model
        self.callback.on_train_begin()
        total_start_time = time.time()
        best_success_rate = 0
        self.status = {
            'i': 0,
            'num_frames': 0,
            'num_episodes': 0
        }
        while self.status['num_frames'] < self.frames:
            self.callback.on_cycle_start()
            update_start_time = time.time()

            # Create experiences and update the training status
            exps, logs = self.collect_experiences()
            self.status['num_frames'] += logs['num_frames']
            self.status['num_episodes'] += logs['episodes_done']
            self.status['i'] += 1

            # Use experience to update policy
            logs = self.update_model_parameters(exps, logs)
            cycle_time = int(time.time() - update_start_time)
            self.callback.on_cycle_end(self.status, logs, cycle_time)

        self.callback.on_train_end()

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch

        Taken from babyAI repo
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i + num_indexes]
                                    for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
