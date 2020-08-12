from abc import ABC

from Players.GenAlg import GenAlg


class HardcodedOpponent(GenAlg, ABC):
    def __init__(self, isLeftPlayer, frameskip):
        super(HardcodedOpponent, self).__init__(frameskip=frameskip, isLeftPlayer=isLeftPlayer, obsIsImage=False)

    def get_action(self, obs, timestep, train_mode=True):
        if obs is None:
            return 0

        ballPos = obs['ballCenter']
        playerPos = obs['leftPlayerCenter'] if self.isLeftPlayer else obs['rightPlayerCenter']

        if ballPos[1] < playerPos[1]:  # If ball is above paddle go up
            return 1
        elif ballPos[1] > playerPos[1]:  # If ball is below paddle go down
            return 2
        else:   # If paddle is lined up perfectly with ball then do nothing
            return 0
