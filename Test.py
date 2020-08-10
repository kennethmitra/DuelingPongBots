from Models.HumanPlayer import HumanPlayer
from Models.HardcodedOpponent import HardcodedOpponent
from Models.GenAlg import GenAlg
from PongEnv import PongEnv


def test(LeftPlayer, RightPlayer, framerate=-1):
    """
    Takes in two players. Feeds players observations and gets actions from player after player.frameskip frames
    Repeats previous action until a new action is obtained
    """
    env = PongEnv(framerate=framerate)

    obs = env.reset()

    info = None
    LeftPlayer_action = 0
    RightPlayer_action = 0

    frame = 0
    while True:
        if (frame % LeftPlayer.frameskip) == 0:
            LeftPlayer_action = LeftPlayer.get_action(obs[0] if LeftPlayer.obsIsImage else obs[1])

        if (frame % RightPlayer.frameskip) == 0:
            RightPlayer_action = RightPlayer.get_action(obs[0] if RightPlayer.obsIsImage else obs[1])

        obs, rew, done, info = env.step((LeftPlayer_action, RightPlayer_action))

        env.render()

        frame += 1
        if done:
            frame = 0
            env.reset()


if __name__ == '__main__':
    test(HumanPlayer(frameskip=1, isLeftPlayer=True), HardcodedOpponent(isLeftPlayer=False, frameskip=5), framerate=60)
