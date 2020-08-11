from abc import ABC, abstractmethod


class GenAlg():

    def __init__(self, frameskip, isLeftPlayer, obsIsImage):
        assert(frameskip > 0)
        self.frameskip = frameskip
        self.isLeftPlayer = isLeftPlayer
        self.obsIsImage = obsIsImage

    @abstractmethod
    def get_action(self, X):
        pass

    def predict(self, X):
        raise NotImplementedError()

    def train_batch(self, min_tsteps):
        pass

    def save(self, save_dir):
        pass

    def load(self, save_path):
        raise NotImplementedError()