import pygame
import random
import numpy as np
import gym
from gym import spaces
import sys
from PIL import Image
import matplotlib.pyplot as plt

from .pongGame import Game

CANVAS_WIDTH = 256
CANVAS_HEIGHT = 256
AGENT_ACT_SPACE = 3


class PongEnv(gym.Env):
    def __init__(self, framerate,player_speeds=(10.0, 10.0)):
        super(PongEnv, self).__init__()

        # Initialize pygame
        pygame.init()

        # Create game window
        self.game = Game(CANVAS_WIDTH, CANVAS_HEIGHT, framerate=framerate, player_speeds=player_speeds)

        self.board_state = None  # Will get set when render() is called

        self.action_space = spaces.Tuple([spaces.Discrete(AGENT_ACT_SPACE), spaces.Discrete(AGENT_ACT_SPACE)])
        self.observation_space = spaces.Box(low=0, high=1, shape=(CANVAS_WIDTH, CANVAS_HEIGHT), dtype=np.float32)

    def reset(self):
        self.game.reset()
        self.board_state = self.game.getScreenBlackWhite()
        obs = (self.board_state, self.game.nonVisualObs())
        return obs

    def step(self, action_vec):
        """
        Performs action_vec on environment and returns new state and reward
        :param action_vec: Tuple (<Left Player Action>, <Right Player Action>)
        :return: obs: Tuple (<Board Image>, <Observation Dictionary>)
                rew: Tuple(<Left player reward>, <Right player reward>
                done: is episode done?
                info: extra info
        """
        assert self.action_space.contains(action_vec)
        assert len(action_vec) == 2
        timestep_info = self.game.step(action_vec[0], action_vec[1])
        reward = (timestep_info['leftPlayerRew'], timestep_info['rightPlayerRew'])
        done = timestep_info['done']
        self.board_state = self.game.getScreenBlackWhite()
        obs = (self.board_state, self.game.nonVisualObs())
        info = None
        return obs, reward, done, info

    def seed(self, seed):
        self.seed = seed
        random.seed(seed)

    def render(self):
        self.game.render()

    def close(self):
        pass



if __name__ == "__main__":
    print("Making env")
    env = PongEnv()
    print("Resetting env")
    env.reset()

    while True:
        print("Stepping")
        print("Ball velocity", env.game.Ball.velocityX, env.game.Ball.velocityY)
        obs, rew, done, info = env.step((1, 2))
        frame = Image.fromarray(obs)
        plt.imshow(frame, cmap='Greys')
        plt.pause(0.0001)
        print(info)
        env.render()
        if done:
            obs = env.reset()
            done = False
