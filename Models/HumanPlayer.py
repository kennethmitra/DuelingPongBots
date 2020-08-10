from abc import ABC

import keyboard
from GenAlg import *


class HumanPlayer(GenAlg, ABC):

    def __init__(self, frameskip, isLeftPlayer):
        super(HumanPlayer, self).__init__(frameskip=frameskip, isLeftPlayer=isLeftPlayer, obsIsImage=False)

    def get_action(self, unused):
        action = 0
        try:
            if keyboard.is_pressed('up'):
                action = 1
            elif keyboard.is_pressed('down'):
                action = 2
        except:
            print("Not a valid Action")
        return action
