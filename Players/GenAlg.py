from abc import ABC, abstractmethod
import torch

class GenAlg(ABC):

    def __init__(self, frameskip, isLeftPlayer):
        assert(frameskip > 0)
        self.frameskip = frameskip
        self.isLeftPlayer = isLeftPlayer


    @abstractmethod
    def get_action(self, obs, timestep, train_mode=True):
        pass

    def predict(self, X):
        raise NotImplementedError()

    def train_batch(self, epoch):
        pass

    def save(self, epoch):
        pass

    def load(self, path, load_optim=True):
        raise NotImplementedError()

    def end_tstep(self, reward, end_episode=False):
        pass