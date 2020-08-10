from abc import ABC

from GenAlg import GenAlg


class HardcodedOpponent(GenAlg, ABC):
    def __init__(self, isLeftPlayer, frameskip):
        super(HardcodedOpponent, self).__init__(frameskip=frameskip, isLeftPlayer=isLeftPlayer, obsIsImage=False)

    def get_action(self, info):
        if info is None:
            return 0

        ballPos = info['ballCenter']
        playerPos = info['leftPlayerCenter'] if self.isLeftPlayer else info['rightPlayerCenter']

        if ballPos[1] < playerPos[1]:  # If ball is above paddle go up
            return 1
        elif ballPos[1] > playerPos[1]:  # If ball is below paddle go down
            return 2
        else:   # If paddle is lined up perfectly with ball then do nothing
            return 0
