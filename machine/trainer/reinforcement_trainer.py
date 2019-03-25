import logging
import os

import numpy
import torch

import machine
from machine.util import DictList
from machine.models import PolicyMapping, SigmoidTermination
from machine.util.callbacks import EpisodeLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReinforcementTrainer(object):
    """
    The ReinforcementTrainer class helps in setting up a training framework for
    reinforcement learning.

    Largely inspired by babyAI repo code for PPOAlgo
    """

    def __init__(self, envs, opt, model, model_name, obs, reshape_reward, algo_name='ppo'):
        self._trainer = f"Reinforcement Trainer - algorithm: {algo_name}"
        self._algo = algo_name
        self.env = envs
        self.preprocess_obss = obs
        self.reshape_reward = reshape_reward
        self.logger = logging.getLogger(__name__)
        self.model_path = os.path.join(opt.output_dir, model_name)

        # Copy command-line arguments to class
        self.frames = opt.frames
        self.frames_per_proc = opt.frames_per_proc
        self.num_procs = opt.num_processes
        self.discount = opt.gamma
        self.lr = opt.lr
        self.gae_lambda = opt.gae_lambda
        self.recurrence = opt.recurrence
        self.batch_size = opt.batch_size
        assert opt.batch_size % opt.recurrence == 0

        self.clip_eps = opt.clip_eps
        self.entropy_coef = opt.entropy_coef
        self.value_loss_coef = opt.value_loss_coef
        self.max_grad_norm = opt.max_grad_norm

        self.num_frames = self.frames_per_proc * self.num_procs
        assert self.frames_per_proc % opt.recurrence == 0

        # Arguments for disruptiveness
        self.explore_for = opt.explore_for
        self.disrupt_mode = opt.disrupt
        self.disrupt_coef = opt.disrupt_coef

        # Initialize observations
        self.obs = self.env.reset()
        self.obss = [None] * self.frames_per_proc

        # Initialize log variables
        self.init_log_vars()

        # Initialize callbacks
        self.callback = EpisodeLogger(
            opt.print_every, opt.save_every, model_name, opt.tb, opt.explore_for)
        self.callback.set_trainer(self)

        # Set parameters for specific algorithms
        if algo_name == 'ppo':
            self.epochs = opt.ppo_epochs
            self.model = model
            self.model.train()
            self.optimizer = torch.optim.Adam(self.model.parameters(
            ), self.lr, (opt.beta1, opt.beta2), eps=opt.optim_eps)
        elif algo_name == 'ppoc':
            self.epochs = opt.ppo_epochs
            self.models = model
            for model in self.models:
                model.train()
            self.optimizer = [torch.optim.Adam(
                m.parameters(), self.lr) for m in self.models]
            self.policy_over_options = PolicyMapping()
            rng = numpy.random.RandomState(42)
            self.policy_terminations = [SigmoidTermination(
                rng, 0.001, 1) for _ in range(len(self.models))]
            self.eta = 0.025
            assert opt.n_options == 1
        else:
            raise ValueError("Not a valid implemented RL algorithm!")

        # Initialize experience matrices
        self.init_experience_matrices()

        self.logger.info(
            f"Setup {self._trainer}, with model_name: {model_name}")

    def collect_experiences(self, intrinsic_reward=False, use_options=False):
        """
        Collect actions, observations and rewards over multiple concurrent
        environments.

        Taken from babyAI repo

        Args:
            intrinsic_reward (bool): Whether to use intrinsic motivation, in
                the form of the disruptiveness metric. If False, get reward
                from the environment and compute advantage.
        """
        for i in range(self.frames_per_proc):
            # Do one agent-environment interaction
            preprocessed_obs = self.preprocess_obss(self.obs, device=device)
            with torch.no_grad():
                if use_options:
                    # Check which option to take
                    option_index = self.policy_over_options.pick_option()
                    policy = self.models[option_index]
                    # Get actions based on option
                    model_results = policy(
                        preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    model_results = self.model(
                        preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                dist = model_results['dist']
                value = model_results['value']
                memory = model_results['memory']

            action = dist.sample()

            obs, reward, done, env_info = self.env.step(action.cpu().numpy())
            # TODO: reward - c_t

            # Update experiences values
            self.update_memory(i, action, value, obs, reward, done)
            self.obs = obs
            self.memory = memory
            self.mask = 1 - \
                torch.tensor(done, device=device, dtype=torch.float)

            # Check if option terminates in new state: self.obs

            # Update log values
            self.update_log_values(i, dist, action, reward, done)

        # Add advantage and return to experiences
        if intrinsic_reward:
            self.compute_disruptiveness()
        else:
            self.compute_advantage()

        # Flatten the data correctly, making sure that each episode's data is
        # a continuous chunk.
        exps = self.flatten_data()

        # Log some values
        log = self.log_output()

        return exps, log

    def update_model_parameters(self, exps, logs):
        """
        Perform gradient update on the model using the gathered experience.

        Taken from babyAI repo
        """
        for e_i in range(self.epochs):
            self.callback.on_epoch_begin(e_i)

            for inds in self._get_batches_starting_indexes():
                batch_logs = self.callback.on_batch_begin(None)
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

                    disrupt_val = torch.tensor(1)
                    if i < self.recurrence - 1:
                        if self.disrupt_mode > 0:
                            s1 = sb.obs.image
                            s2 = exps[inds + i + 1].obs.image
                            disrupt_val = torch.sum(
                                s1 != s2, dtype=torch.float)
                            disrupt_val = torch.log(disrupt_val)
                            disrupt_val = torch.clamp(
                                disrupt_val, min=.01, max=10)
                            disrupt_val *= self.disrupt_coef

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

                    if self.disrupt_mode == 1:
                        loss = (policy_loss * disrupt_val) - self.entropy_coef * \
                            entropy + (self.value_loss_coef * value_loss)
                    elif self.disrupt_mode == 2:
                        loss = policy_loss - self.entropy_coef * \
                            entropy + (self.value_loss_coef *
                                       (value_loss * disrupt_val))
                    else:
                        loss = policy_loss - self.entropy_coef * \
                            entropy + (self.value_loss_coef * value_loss)

                    # Update loss
                    batch_loss += loss

                    # Update batch logging values
                    batch_logs['entropy'] += entropy.item()
                    batch_logs['value'] += value.mean().item()
                    batch_logs['policy_loss'] += policy_loss.item()
                    batch_logs['value_loss'] += value_loss.item()
                    batch_logs['disrupt'] += disrupt_val

                    # Update memories for next epoch
                    if i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update loss
                batch_loss /= self.recurrence

                # Update batch logging values
                batch_logs['entropy'] /= self.recurrence
                batch_logs['value'] /= self.recurrence
                batch_logs['policy_loss'] /= self.recurrence
                batch_logs['value_loss'] /= self.recurrence
                batch_logs['disrupt'] /= self.recurrence

                # Update actor-critic
                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(
                    2) ** 2 for p in self.model.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values
                batch_logs['grad_norm'] = grad_norm
                self.callback.on_batch_end(batch_loss, batch_logs)

            logs = self.callback.on_epoch_end(logs)
        return logs

    def train(self):
        """
        Perform a series on training steps as configured.
        """
        # Start training model
        self.callback.on_train_begin()
        num_frames = 0
        while num_frames < self.frames:
            self.callback.on_cycle_start()

            # Create experiences and update the training status
            exps, logs = self.collect_experiences(
                intrinsic_reward=(num_frames < self.explore_for),
                use_options=(self._algo == 'ppoc')
            )

            # Use experience to update policy
            logs = self.update_model_parameters(exps, logs)

            num_frames += logs['num_frames']
            self.callback.on_cycle_end(logs)

        self.callback.on_train_end()

    def init_experience_matrices(self):
        """
        Initialize matrices used in the storing of observations.
        """
        shape = (self.frames_per_proc, self.num_procs)
        if self._algo == 'ppo':
            memsize = self.model.memory_size
        elif self._algo == 'ppoc':
            memsize = self.models[0].memory_size
        self.memory = torch.zeros(shape[1], memsize, device=device)
        self.memories = torch.zeros(*shape, memsize, device=device)
        self.mask = torch.ones(shape[1], device=device)
        self.masks = torch.zeros(*shape, device=device)
        self.actions = torch.zeros(*shape, device=device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=device)
        self.rewards = torch.zeros(*shape, device=device)
        self.advantages = torch.zeros(*shape, device=device)
        self.log_probs = torch.zeros(*shape, device=device)

        if self._algo == 'ppoc':
            self.c_t = torch.zeros(shape[1], device=device)

    def init_log_vars(self):
        """
        Initialize the variables used for logging training progress.
        """
        self.log_episode_return = torch.zeros(self.num_procs, device=device)
        self.log_episode_reshaped_return = torch.zeros(
            self.num_procs, device=device)
        self.log_episode_num_frames = torch.zeros(
            self.num_procs, device=device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def update_memory(self, i, action, value, obs, reward, done):
        """
        Update the memory matrices based on agent-environment interaction.
        """
        self.obss[i] = self.obs
        self.memories[i] = self.memory
        self.masks[i] = self.mask
        self.actions[i] = action
        self.values[i] = value
        if self.reshape_reward is not None:
            self.rewards[i] = torch.tensor([
                self.reshape_reward(obs_, action_, reward_, done_)
                for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
            ], device=device)
        else:
            self.rewards[i] = torch.tensor(reward, device=device)

    def update_log_values(self, i, dist, action, reward, done):
        """
        Update the logging values used for keeping track of training progress.
        """
        self.log_probs[i] = dist.log_prob(action)
        self.log_episode_return += torch.tensor(
            reward, device=device, dtype=torch.float)
        self.log_episode_reshaped_return += self.rewards[i]
        self.log_episode_num_frames += torch.ones(
            self.num_procs, device=device)

        for j, done_ in enumerate(done):
            if done_:
                self.log_done_counter += 1
                self.log_return.append(self.log_episode_return[j].item())
                self.log_reshaped_return.append(
                    self.log_episode_reshaped_return[j].item())
                self.log_num_frames.append(
                    self.log_episode_num_frames[j].item())

        self.log_episode_return *= self.mask
        self.log_episode_reshaped_return *= self.mask
        self.log_episode_num_frames *= self.mask

    def compute_advantage(self, option_idx=None):
        """
        Run the advantage estimation from [1].

        A_t = delta_t + (gamma * lambda) delta_(t+1) ...
            with
        delta_t = reward_t + gamma V(s_(t+1)) - V(s_t)

        [1]: Mnih et al. (2016) "Asynchronous methods for deep reinforcement learning"
        """
        preprocessed_obs = self.preprocess_obss(self.obs, device=device)
        with torch.no_grad():
            if option_idx is not None:
                next_value = self.models[option_idx](
                    preprocessed_obs, self.memory * self.mask.unsqueeze(1))['value']
            else:
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

    def compute_disruptiveness(self):
        """
        Compute an intrinsic reward based on the disruptiveness metric.
        """
        preprocessed_obs = self.preprocess_obss(self.obs, device=device)
        with torch.no_grad():
            next_value = self.model(
                preprocessed_obs, self.memory * self.mask.unsqueeze(1))['value']

        for i in range(self.frames_per_proc):
            s_t = self.obss[i]
            s_t1 = self.obss[i +
                             1] if i < (self.frames_per_proc - 1) else self.obs

            # Binary difference
            state_t = torch.Tensor([s['image'] for s in s_t])
            state_t1 = torch.Tensor([s['image'] for s in s_t1])
            val = torch.nonzero(state_t - state_t1).size()[0]
            # Normalize over max change
            self.advantages[i] = val / (7 * 7 * self.num_procs)

    def flatten_data(self):
        """
        Flatten the memory such that it is a continuous chunk of data. This is
        required by the PyTorch optimization step.
        """
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
        return exps

    def log_output(self):
        """
        Create logging output based on the observed training progress.
        """
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
        return log

    def _get_batches_starting_indexes(self):
        """
        Gives, for each batch, the indexes of the observations given to
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
