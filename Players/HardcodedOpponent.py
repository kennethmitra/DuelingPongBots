from abc import ABC

from Players.GenAlg import GenAlg


class HardcodedOpponent(GenAlg, ABC):
    def __init__(self, isLeftPlayer, frameskip,model):
        super(HardcodedOpponent, self).__init__(frameskip=frameskip, isLeftPlayer=isLeftPlayer)
        self.model = model

    def get_action(self, obs, timestep, train_mode=True):
        if obs is None:
            return 0

        ballPos = (obs[0], obs[1])
        playerPos = (obs[4], obs[5]) if self.isLeftPlayer else (obs[6], obs[7])

        if ballPos[1] < playerPos[1]:  # If ball is above paddle go up
            return 1
        elif ballPos[1] > playerPos[1]:  # If ball is below paddle go down
            return 2
        else:   # If paddle is lined up perfectly with ball then do nothing
            return 0
