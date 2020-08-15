from Models.Dummy_Model import DummyModel
from Players.HardcodedOpponent import HardcodedOpponent
from PongEnv.PongEnv import PongEnv
import gym
import numpy as np
from stable_baselines3 import A2C
import torch


# Using stable-baselines to debug

# Wraps PongEnv to make it a single player environment
class PongEnvWrapper(gym.Env):
     def __init__(self):
         super(PongEnvWrapper, self).__init__()
         self.env = PongEnv(framerate=-1, player_speeds=(10.0,10.0))
         self.action_space = gym.spaces.Discrete(3)
         self.observation_space = gym.spaces.Box(low=np.array([-256, -256, -30, -30, -256, -256, -256, -256]), high=np.array([256, 256, 30, 30, 256, 256, 256, 256]), dtype=np.float)

         self.RightPlayer = HardcodedOpponent(isLeftPlayer=False, frameskip=1, model=DummyModel())
         self.epo = 0

     def step(self, action):
         obs, rew, done, info = self.env.step((action, self.R_action))
         self.R_action = self.RightPlayer.get_action(obs=obs[1], timestep=None)
         obs = obs[1]
         obs = np.asarray([obs[k] for k in obs.keys()])
         obs = obs.flatten()
         return obs, rew[0], done, dict()

     def reset(self):
         self.epo += 1
         obs = self.env.reset()
         self.R_action = self.RightPlayer.get_action(obs=obs[1], timestep=None)
         obs = obs[1]
         obs = np.asarray([obs[k] for k in obs.keys()])
         obs = obs.flatten()
         return obs

     def render(self):
         self.env.render()

     def close(self):
         self.env.close()

if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env
    env = PongEnvWrapper()
    check_env(env)

    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[64])
    model = A2C('MlpPolicy', env=env, verbose=1, policy_kwargs=policy_kwargs,tensorboard_log='sb_runs')
    model.learn(total_timesteps=250000)
    model.save(f"StableBaselinesPolicyEpochs{env.epo}.save")
    print(f"Trained for {env.epo} epochs")

    #model.load('StableBaselinesPolicy.save')
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            obs = env.reset()


