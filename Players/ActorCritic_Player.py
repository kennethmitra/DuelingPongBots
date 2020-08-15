import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from itertools import count
from Players.Buffer import Buffer
from Players.Logger import Logger
import os
from pathlib import Path
import numpy as np
import time
from Players.GenAlg import GenAlg


class ActorCritic_Player(GenAlg):
    def __init__(self, env, run_name, frameskip, isLeftPlayer, model, train_mode=True):
        """
        Construct neural network(s) for actor and critic
        :param obs_cnt: Number of components in an observation
        :param action_cnt: Number of possible actions
        """

        super(ActorCritic_Player, self).__init__(frameskip, isLeftPlayer)
        output_dim = env.action_space[0].n

        # Hyperparameters --------------------------
        self.ENVIRONMENT = 'Pong-v0'
        self.SEED = 543
        self.LEARNING_RATE = 6e-4
        self.DISCOUNT_FACTOR = 0.93
        self.ENTROPY_COEFF = 0.0
        self.TIMESTEPS_PER_EPOCH = 10000
        self.ACTIVATION_FUNC = torch.relu
        self.NORMALIZE_RETURNS = False
        self.NORMALIZE_ADVANTAGES = False
        self.CLIP_GRAD = False
        self.NUM_PROCESSES = 1
        self.RUN_NAME = run_name
        self.NOTES = "Episodes per epoch is now 10"
        # -----------------------------------------

        self.episode_rewards = [] # Rewards for each timestep in current episode

        self.model = model

        print("-------------------------------GPU INFO--------------------------------------------")
        print('Available devices ', torch.cuda.device_count())
        self.model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.model.device = "cpu"
        print('Current cuda device ', self.model.device)
        if self.model.device != "cpu":
            print('Current CUDA device name ', torch.cuda.get_device_name(self.model.device))
        print("-----------------------------------------------------------------------------------")

        self.model.to(self.model.device)


        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.LEARNING_RATE)


        self.buf = Buffer()

        # Choose save directory
        if run_name is None:
            run_name = Path(__file__).stem

        if train_mode:
            # Ex) If run_name is "dog", and dog-1, dog-2 are taken, save at dog-3
            for i in count():
                if not os.path.exists(f'./saves/{run_name}-{i}'):
                    run_name = f'{run_name}-{i}'
                    break
            Path(f'./saves/{run_name}').mkdir(parents=True, exist_ok=True)
            self.save_path = f'./saves/{run_name}'

            # Choose Tensorboard directory
            for i in count():
                if not os.path.exists(f'./runs/{run_name}-{i}'):
                    run_name = f'{run_name}-{i}'
                    break
            Path(f'./runs/{run_name}').mkdir(parents=True, exist_ok=True)
            self.log_dir = f'./runs/{run_name}'

            self.log = Logger(log_dir=self.log_dir, refresh_secs=30)
            self.log.log_hparams(ENVIRONMENT=self.ENVIRONMENT,
                                 SEED=self.SEED,
                                 model=self.model,
                                 optimizer=self.optimizer,
                                 LEARNING_RATE=self.LEARNING_RATE,
                                 DISCOUNT_FACTOR=self.DISCOUNT_FACTOR,
                                 ENTROPY_COEFF=self.ENTROPY_COEFF,
                                 activation_func=self.ACTIVATION_FUNC,
                                 tsteps_per_epoch=self.TIMESTEPS_PER_EPOCH,
                                 normalize_returns=self.NORMALIZE_RETURNS,
                                 normalize_advantages=self.NORMALIZE_ADVANTAGES,
                                 clip_grad=self.CLIP_GRAD, notes=self.NOTES, display=True)

        print("Num params:", sum(p.numel() for p in self.model.parameters()))
    def predict(self, obs):
        """
        Compute action distribution and value from an observation
        :param obs: observation with len obs_cnt
        :return: Action distribution (Categorical) and value (tensor)
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        obs = obs.to(self.model.device)

        actionLogits, value = self.model.forward(obs)

        action_dist = Categorical(logits=actionLogits)
        return action_dist, value

    def get_action(self, obs, timestep, train_mode=True):
        """
        Given an observation, predict action distribution and value and sample action
        Store log prob of sampled action, value calculated by critic, entropy of action prob dist
        :param obs: observation from env.step() or env.reset()
        :return: sampled action
        """
        action_dist, value = self.predict(obs)
        action = action_dist.sample()
        entropy = action_dist.entropy()

        if train_mode:  # Buffer is only used in training
            self.buf.record(timestep=timestep, obs=obs, act=action, logp=action_dist.log_prob(action), val=value,
                            entropy=entropy)

        return action.item()


    def save(self, epoch):
        try:
            torch.save({'epoch': epoch,
                        'optimizer_params': self.optimizer.state_dict(),
                        'model_state': self.model.state_dict()}, f'{self.save_path}/epo{epoch}.save')
        except:
            print('ERROR calling model.save()')

    def load(self, path, load_optim=True):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        if load_optim:
            self.optimizer.load_state_dict(checkpoint['optimizer_params'])
        else:
            print("Skipping saved optimizer")
        return checkpoint['epoch']

    def discount_rewards_to_go(self, episode_rewards, gamma):
        # Calculate discounted Rewards-To-Go (returns)
        returns = []
        running_sum = 0
        for r in episode_rewards[::-1]:
            running_sum = r + gamma * running_sum
            returns.insert(0, running_sum)

        del running_sum
        return returns

    def end_tstep(self, reward, end_episode=False):
        # Record timestep reward
        self.episode_rewards.append(reward)
        self.buf.record(rew=reward)

        # Handle end of episode
        if (end_episode):
            # Calculate discounted rewards to go
            ep_disc_rtg = self.discount_rewards_to_go(episode_rewards=self.episode_rewards, gamma=self.DISCOUNT_FACTOR)
            # End episode in buffer
            self.buf.store_episode_stats(episode_rews=self.episode_rewards, episode_disc_rtg_rews=ep_disc_rtg)
            self.episode_rewards.clear()

    def train_batch(self, epoch):
        print("Training epoch", epoch)
        data = self.buf.get()
        normalize_returns = self.NORMALIZE_RETURNS
        normalize_advantages = self.NORMALIZE_ADVANTAGES
        entropy_coeff = self.ENTROPY_COEFF
        clip_grad = self.CLIP_GRAD

        update_start_time = time.perf_counter()

        # Sanity Check
        assert len(data['tstep']) == len(data['obs']) == len(data['act']) == len(data['logp']) == len(data['val']) \
               == len(data['rew']) == len(data['entropy']) == len(data['disc_rtg_rews']) == len(data['disc_rtg_rews'])
        assert len(data['per_episode_rews']) == len(data['per_episode_length'])

        # Don't need to backprop through returns
        returns = torch.tensor(data['disc_rtg_rews']).to(self.model.device)

        if normalize_returns:
            returns = (returns) / returns.std()

        values = torch.squeeze(torch.stack(data['val']))

        advantages = returns - values


        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / advantages.std()

        assert len(advantages) == len(data['tstep'])

        # Zero out gradients before calculating loss
        self.optimizer.zero_grad()

        # Calculate actor and critic loss (Using torch functions - hopefully faster)
        actor_loss = -torch.stack(data['logp']) * advantages
        critic_loss = F.smooth_l1_loss(returns, values)

        actor_loss = actor_loss.mean()
        critic_loss = 0.5 * critic_loss.mean()
        entropy_avg = torch.stack(data['entropy']).mean()
        entropy_loss = -(entropy_coeff * entropy_avg)
        total_loss = actor_loss + critic_loss + entropy_loss

        # Perform backprop step
        total_loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        # Compute info for logging
        avg_ep_len = torch.tensor(data['per_episode_length'], requires_grad=False, dtype=torch.float).mean().item()
        avg_ep_raw_rew = torch.tensor(data['per_episode_rews'], requires_grad=False, dtype=torch.float).mean().item()
        avg_ep_disc_rew = torch.tensor(data['per_episode_disc_rews'], requires_grad=False, dtype=torch.float).mean().item()
        raw_rews = torch.tensor(data['rew'], requires_grad=False, dtype=torch.float)
        epoch_timesteps = data['tstep'][-1]
        num_episodes = len(data['per_episode_length'])

        # Package logging info
        epoch_info = dict(actor_loss=actor_loss, critic_loss=critic_loss, entropy_loss=entropy_loss,
                          entropy_avg=entropy_avg,
                          total_loss=total_loss, avg_ep_len=avg_ep_len, avg_ep_raw_rew=avg_ep_raw_rew,
                          epoch_timesteps=epoch_timesteps, num_episodes=num_episodes, advantages=advantages,
                          pred_values=data['val'], disc_rews=returns, raw_rew=raw_rews,
                          update_time=(time.perf_counter() - update_start_time),
                          avg_ep_disc_rew=avg_ep_disc_rew)
        # Log
        self.log.log_epoch(epoch, epoch_info)

        # Delete contents of buffer
        self.buf.clear()

        # Delete variables to save memory (Since python garbage collection is super sketch)
        del epoch_info
        del avg_ep_len
        del avg_ep_raw_rew
        del avg_ep_disc_rew
        del raw_rews
        del epoch_timesteps
        del num_episodes
        del total_loss
        del actor_loss
        del critic_loss
        del entropy_avg
        del entropy_loss
        del values
        del returns
        del advantages
        del data
        del normalize_returns
        del normalize_advantages
        del entropy_coeff
        del clip_grad
        del update_start_time
