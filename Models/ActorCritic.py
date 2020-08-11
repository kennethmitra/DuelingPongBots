import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from itertools import count
import gym
from Buffer import Buffer
from Logger import Logger
import os
from pathlib import Path
import numpy as np
import time
from GenAlg import GenAlg


class ActorCritic(GenAlg):

    class Model(torch.nn.Module):

        def __init__(self, output_dim):
            # Shared conv layers for feature extraction
            super(ActorCritic.Model, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 64, 3)
            self.max_pool = torch.nn.MaxPool2d(2)
            self.conv2 = torch.nn.Conv2d(64, 128, 5)

            # Actor Specific
            self.actor_layer1 = torch.nn.Linear(128 * 61 * 61, 64)
            self.actor_layer2 = torch.nn.Linear(64, output_dim)
            torch.nn.init.xavier_uniform_(self.actor_layer1.weight)
            torch.nn.init.xavier_uniform_(self.actor_layer2.weight)

            # Critic Specific
            self.critic_layer1 = torch.nn.Linear(128 * 61 * 61, 64)
            self.critic_layer2 = torch.nn.Linear(64, output_dim)
            torch.nn.init.xavier_uniform_(self.critic_layer1.weight)
            torch.nn.init.xavier_uniform_(self.critic_layer2.weight)

        def forward(self, obs):
            """
            Compute action distribution and value from an observation
            :param obs: observation with len obs_cnt
            :return: Action distribution (Categorical) and value (tensor)
            """

            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).float()

            # Add batch dimension and channel dimension (256, 256) -> (1, 1, 256, 256) | (n, c, h, w)
            obs = torch.unsqueeze(obs, 0)
            obs = torch.unsqueeze(obs, 0)

            # Separate Actor and Critic Networks
            obs = self.conv1(obs)
            obs = F.relu(obs)
            obs = self.max_pool(obs)
            obs = self.conv2(obs)
            obs = self.max_pool(obs)
            obs = F.relu(obs)
            obs = obs.view(-1, 7808 * 61)

            # Actor Specific
            actor_intermed = self.actor_layer1(obs)
            actor_intermed = torch.nn.Tanh()(actor_intermed)
            actor_Logits = self.actor_layer2(actor_intermed)

            # Critic Logits
            critic_intermed = self.critic_layer1(obs)
            critic_intermed = torch.nn.Tanh()(critic_intermed)
            value = self.critic_layer2(critic_intermed)

            return actor_Logits, value

    def __init__(self, env, run_name, frameskip, isLeftPlayer):
        """
        Construct neural network(s) for actor and critic
        :param obs_cnt: Number of components in an observation
        :param action_cnt: Number of possible actions1
        """

        super(ActorCritic, self).__init__(frameskip, isLeftPlayer, True)
        output_dim = env.action_space[0].n

        # Hyperparameters --------------------------
        self.ENVIRONMENT = 'Pong-v0'
        self.SEED = 543
        self.LEARNING_RATE = 0.0011
        self.DISCOUNT_FACTOR = 0.997
        self.ENTROPY_COEFF = 0.0
        self.TIMESTEPS_PER_EPOCH = 10000
        self.ACTIVATION_FUNC = torch.relu
        self.NORMALIZE_REWARDS = False
        self.NORMALIZE_ADVANTAGES = True
        self.CLIP_GRAD = False
        self.NUM_PROCESSES = 1
        self.RUN_NAME = "Pong-A2C"
        self.NOTES = ""
        # -----------------------------------------

        self.episode_rewards = []

        self.model = self.Model(output_dim=output_dim)


        print("-------------------------------GPU INFO--------------------------------------------")
        print('Available devices ', torch.cuda.device_count())
        self.model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Current cuda device ', self.model.device)
        print('Current CUDA device name ', torch.cuda.get_device_name(self.model.device))
        print("-----------------------------------------------------------------------------------")

        self.model.to(self.model.device)

        self.log = Logger(run_name=None, refresh_secs=30)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.LEARNING_RATE)

        self.log.log_hparams(ENVIRONMENT=self.ENVIRONMENT,
                             SEED=self.SEED,
                             model=self.model,
                             optimizer=self.optimizer,
                             LEARNING_RATE=self.LEARNING_RATE,
                             DISCOUNT_FACTOR=self.DISCOUNT_FACTOR,
                             ENTROPY_COEFF=self.ENTROPY_COEFF,
                             activation_func=self.ACTIVATION_FUNC,
                             tsteps_per_epoch=self.TIMESTEPS_PER_EPOCH,
                             normalize_rewards=self.NORMALIZE_REWARDS,
                             normalize_advantages=self.NORMALIZE_ADVANTAGES,
                             clip_grad=self.CLIP_GRAD, notes=self.NOTES, display=True)

        self.buf = Buffer()


        if run_name is None:
            run_name = Path(__file__).stem

        # Ex) If run_name is "dog", and dog-1, dog-2 are taken, save at dog-3
        for i in count():
            if not os.path.exists(f'./saves/{run_name}-{i}'):
                run_name = f'{run_name}-{i}'
                break
        Path(f'./saves/{run_name}').mkdir(parents=True, exist_ok=True)
        self.save_path = f'./saves/{run_name}'

    def predict(self, obs):
        """
        Compute action distribution and value from an observation
        :param obs: observation with len obs_cnt
        :return: Action distrition (Categorical) and value (tensor)
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
            self.buf.store_episode_stats(episode_disc_rtg_rews=ep_disc_rtg)
            self.episode_rewards.clear()

    def train_batch(self, epoch):

        data = self.buf.get()
        normalize_returns = self.NORMALIZE_REWARDS
        normalize_advantages = self.NORMALIZE_ADVANTAGES
        entropy_coeff = self.ENTROPY_COEFF
        clip_grad = self.CLIP_GRAD

        update_start_time = time.perf_counter()

        # Sanity Check
        assert len(data['tstep']) == len(data['obs']) == len(data['act']) == len(data['logp']) == len(data['val']) \
               == len(data['rew']) == len(data['entropy']) == len(data['disc_rtg_rews']) == len(data['disc_rtg_rews'])
        assert len(data['per_episode_rews']) == len(data['per_episode_length'])

        # Don't need to backprop through returns
        returns = torch.tensor(data['disc_rtg_rews']).to(self.device)

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
        raw_rews = torch.tensor(data['rew'], requires_grad=False, dtype=torch.float)
        epoch_timesteps = data['tstep'][-1]
        num_episodes = len(data['per_episode_length'])

        # Package logging info
        epoch_info = dict(actor_loss=actor_loss, critic_loss=critic_loss, entropy_loss=entropy_loss,
                          entropy_avg=entropy_avg,
                          total_loss=total_loss, avg_ep_len=avg_ep_len, avg_ep_raw_rew=avg_ep_raw_rew,
                          epoch_timesteps=epoch_timesteps, num_episodes=num_episodes, advantages=advantages,
                          pred_values=data['val'], disc_rews=returns, raw_rew=raw_rews,
                          update_time=(time.perf_counter() - update_start_time))
        # Log
        self.log(epoch, epoch_info)
        # Clear buffer
        self.buf.clear()

# if __name__ == '__main__':
#     print("-------------------------------GPU INFO--------------------------------------------")
#     print('Available devices ', torch.cuda.device_count())

#     print('Current cuda device ', device)
#     print('Current CUDA device name ', torch.cuda.get_device_name(device))
#     print("-----------------------------------------------------------------------------------")


#     TRAIN_MODE = False

#     torch.manual_seed(SEED)
#     env = gym.make(ENVIRONMENT)
#     env.seed(SEED)

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = ActorCritic (env = env, run_name=RUN_NAME)
#     model.device = device
#     model = model.to(model.device)

#     model.optimizer = torch.optim.Adam(params=model.parameters(),lr = DEFAULT_LR)

#     buf = Buffer()
#     self.log = Logger(run_name=None, refresh_secs=30)

#     # Load saved weights
#     start_epoch = model.load("./saves/epo5350.save", load_optim=True)
#     # Override epoch
#     # start_epoch = 0

#     self.log.log_hparams(ENVIRONMENT=ENVIRONMENT, SEED=SEED, model=model, ACTOR_LEARNING_RATE=ACTOR_LEARNING_RATE,
#                     CRITIC_LEARNING_RATE=CRITIC_LEARNING_RATE, DISCOUNT_FACTOR=DISCOUNT_FACTOR,
#                     ENTROPY_COEFF=ENTROPY_COEFF, activation_func=ACTIVATION_FUNC,
#                     tsteps_per_epoch=TIMESTEPS_PER_EPOCH, normalize_rewards=NORMALIZE_REWARDS,
#                     normalize_advantages=NORMALIZE_ADVANTAGES, clip_grad=CLIP_GRAD, notes=NOTES, display=True)

#     # Setup env for first episode
#     obs = env.reset()

#     episode_rewards = []
#     episode = 0
#     epoch = 0

#     # Iterate over epochs
#     for epoch in range(start_epoch, NUM_EPOCHS):

#         # Render first episode of every Nth epoch
#         render = ((epoch % 1) == 0) or (not TRAIN_MODE)

#         # Continue getting timestep data until reach TIMESTEPS_PER_EPOCH
#         for timestep in count():

#             # Get action prediction from model
#             action, logprob, value, entropy = model.get_action(obs)

#             # Perform action in environment and get new observation and rewards
#             new_obs, reward, done, _ = env.step(action.item())

#             # Store state-action information for updating model
#             buf.record(timestep=timestep, obs=obs, act=action, logp=logprob, val=value, entropy=entropy, rew=reward)

#             obs = new_obs
#             if render: env.render()

#             if done:
#                 render = False

#                 # Store discounted Rewards-To-Go™
#                 ep_disc_rtg = model.discount_rewards_to_go(episode_rewards=episode_rewards, gamma=DISCOUNT_FACTOR)
#                 buf.store_episode_stats(episode_rewards=episode_rewards, episode_disc_rtg_rews=ep_disc_rtg,
#                                         episode_length=timestep)

#                 # Initialize env after end of episode
#                 obs = env.reset()
#                 episode_rewards.clear()
#                 episode += 1

#                 if timestep >= TIMESTEPS_PER_EPOCH:
#                     break
#                 else:
#                     print(f"episode: {episode}, timestep {timestep} of {TIMESTEPS_PER_EPOCH}")

#         # Save model
#         if epoch % 50 == 0:
#             model.save(epoch=epoch)

#         # Train model on epoch data
#         if TRAIN_MODE:
#             epoch_info = model.learn_from_experience(data=buf.get(), normalize_returns=NORMALIZE_REWARDS,
#                                                      entropy_coeff=ENTROPY_COEFF, clip_grad=CLIP_GRAD)

#             # Log epoch statistics
#             log.log_epoch(epoch, epoch_info)

#         # Clear buffer
#         buf.clear()

#     # After Training
#     model.save(epoch=epoch)
#     env.close()
