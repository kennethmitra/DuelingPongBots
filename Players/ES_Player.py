from .GenAlg import GenAlg
import torch
from .Logger import Logger
from .Buffer import Buffer
from torch.distributions import Categorical
from itertools import count
from pathlib import Path
import time
import os
import numpy as np


class VPG_Player(GenAlg):
    """
    A Simple Vanilla Policy Gradient Algorithm for debugging purposes
    """

    def __init__(self, env, run_name, frameskip, isLeftPlayer, model, train_mode=True):
        super(VPG_Player, self).__init__(frameskip=frameskip, isLeftPlayer=isLeftPlayer)

        # Takes in a simple neural network that only predicts actions
        self.model = model

        # Hyperparameters --------------------------
        self.ENVIRONMENT = 'Pong'
        self.SEED = 543
        self.LEARNING_RATE = 6e-4
        self.ACTIVATION_FUNC = torch.relu
        self.NORMALIZE_RETURNS = False
        self.npop = 50  # population size
        self.sigma = 0.1  # noise standard deviation
        self.alpha = 0.001  # learning rate        self.RUN_NAME = run_name
        self.NOTES = "Vanilla Policy Gradient"
        # -----------------------------------------

        self.episode_rewards = []  # Rewards for each timestep in current episode
        self.noise_matrix = torch.randn((self.npop, [torch.size(self.model.weight)]))
        self.ctr = 0

        print("-------------------------------GPU INFO--------------------------------------------")
        print('Available devices ', torch.cuda.device_count())
        self.model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.model.device = "cpu"
        print('Current cuda device ', self.model.device)
        if self.model.device != "cpu":
            print('Current CUDA device name ', torch.cuda.get_device_name(self.model.device))
        print("-----------------------------------------------------------------------------------")

        self.model.to(self.model.device)

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

            # Setup logging
            self.log = Logger(log_dir=self.log_dir, refresh_secs=30)
            self.log.log_hparams(ENVIRONMENT=self.ENVIRONMENT,
                                 SEED=self.SEED,
                                 model=self.model,
                                 LEARNING_RATE=self.LEARNING_RATE,
                                 DISCOUNT_FACTOR=self.DISCOUNT_FACTOR,
                                 activation_func=self.ACTIVATION_FUNC,
                                 normalize_returns=self.NORMALIZE_RETURNS,
                                 notes=self.NOTES, display=True)

    def get_action(self, obs, timestep, train_mode=True):
        action_dist = self.predict(obs)
        action = action_dist.sample()
        entropy = action_dist.entropy()

        if train_mode:  # Buffer is only used in training
            # Set val to placeholder value since it isn't used
            self.buf.record(timestep=timestep, obs=obs, act=action, logp=action_dist.log_prob(action), val=None,
                            entropy=entropy)

        return action.item()

    def predict(self, obs):

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        obs = obs.to(self.model.device)

        actionLogits, _ = self.model.forward(obs)

        action_dist = Categorical(logits=actionLogits)
        return action_dist

    def load(self, path, load_optim=False):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        return checkpoint['epoch']

    def save(self, epoch):
        try:
            torch.save({'epoch': epoch,
                        'model_state': self.model.state_dict()}, f'{self.save_path}/epo{epoch}.save')
        except:
            print('ERROR calling model.save()')

    def end_tstep(self, reward, end_episode=False):
        # Record timestep reward
        self.episode_rewards.append(reward)
        self.buf.record(rew=reward)

        self.model.weight = self.model.weight + self.noise_matrix[self.ctr]

        # Handle end of episode
        if end_episode:
            # Calculate discounted rewards to go
            ep_disc_rtg = max(self.episode_rewards)
            self.ctr+=1
            # End episode in buffer
            self.buf.store_episode_stats(episode_rews=self.episode_rewards, episode_disc_rtg_rews=ep_disc_rtg)
            self.episode_rewards.clear()


    def train_batch(self, epoch):
        print("Training epoch", epoch)
        data = self.buf.get()
        normalize_returns = self.NORMALIZE_RETURNS

        update_start_time = time.perf_counter()

        # Don't need to backprop through returns
        returns = torch.tensor(data['per_episode_rews']).to(self.model.device)

        if normalize_returns:
            returns = (returns) / returns.std()

        # Calculate actor and critic loss (Using torch functions - hopefully faster)
        entropy_avg = torch.stack(data['entropy']).mean()
        self.model.weight = self.model.weight + self.LEARNING_RATE/(self.npop*self.sigma)\
                            + torch.dot(torch.transpose(self.noise_matrix), returns)
        self.noise_matrix = torch.randn((self.npop, [torch.size(self.model.weight)]))

        # Compute info for logging
        avg_ep_len = torch.tensor(data['per_episode_length'], requires_grad=False, dtype=torch.float).mean().item()
        avg_ep_raw_rew = torch.tensor(data['per_episode_rews'], requires_grad=False, dtype=torch.float).mean().item()
        avg_ep_disc_rew = torch.tensor(data['per_episode_disc_rews'], requires_grad=False,
                                       dtype=torch.float).mean().item()
        raw_rews = torch.tensor(data['rew'], requires_grad=False, dtype=torch.float)
        epoch_timesteps = data['tstep'][-1]
        num_episodes = len(data['per_episode_length'])

        # Package logging info
        epoch_info = dict( entropy_avg=entropy_avg,
                           avg_ep_len=avg_ep_len, avg_ep_raw_rew=avg_ep_raw_rew,
                          epoch_timesteps=epoch_timesteps, num_episodes=num_episodes, disc_rews=returns,
                          raw_rew=raw_rews, update_time=(time.perf_counter() - update_start_time),
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
        del entropy_avg
        del returns
        del data
        del normalize_returns
        del update_start_time
