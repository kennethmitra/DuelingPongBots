from Players.HumanPlayer import HumanPlayer
from Players.HardcodedOpponent import HardcodedOpponent
from PongEnv import PongEnv
from Players.HardcodedOpponent import HardcodedOpponent
from Players.ActorCritic_Player import ActorCritic_Player
from Players.VPG_Player import VPG_Player
from PongEnv.PongEnv import PongEnv
from PIL import Image
import numpy as np
import time
from Players.GenAlg import GenAlg
from utils.GIF_Recorder import GIF_Recorder
from Models.Gen_FC import Gen_FC
from Models.Dummy_Model import DummyModel

def test(env, LeftPlayer, RightPlayer, framerate=-1, epochs=10, episodes_per_epoch=3, L_start_epoch=0, R_start_epoch=0):
    """
    Takes in two players. Feeds players observations and gets actions from player after player.frameskip frames
    Repeats previous action until a new action is obtained
    If framerate is -1, runs with unlimited framerate. If framerate is > 0
    """
    # Modulus that defines saving freq
    obs = env.reset()

    render = True

    L_timestep = 0
    R_timestep = 0
    L_obs = None
    R_obs = None

    L_action = None
    R_action = None

    L_reward = 0
    R_reward = 0

    frame = 0
    episode = 0
    episode_counter = 0

    # Get epoch data
    while True:

        # Get action from models if needed
        if frame % LeftPlayer.frameskip == 0:
            # Store timestep
            if frame != 0:
                L_reward = 0

            # Downsample obs if needed
            L_obs = None
            if LeftPlayer.model.obsIsImage:
                L_obs = Image.fromarray(obs[0])
                L_obs = L_obs.resize((32, 32), resample=Image.LANCZOS)
                # plt.imshow(L_obs)
                # plt.pause(0.000000001)

                L_obs = np.asarray(L_obs)
                last_obs = L_obs

            else:
                L_obs = obs[1]
                L_obs = np.asarray([L_obs[k] for k in L_obs.keys()])
                L_obs = L_obs.flatten()

            # Get new action
            L_action = LeftPlayer.get_action(L_obs, timestep=L_timestep, train_mode=False)  # get_action stores obs, logprobs, and any other intermediary stuff
            L_timestep += 1
        if frame % RightPlayer.frameskip == 0:

            # Downsample obs if needed
            R_obs = None
            if RightPlayer.model.obsIsImage:
                R_obs = Image.fromarray(obs[0])
                R_obs = R_obs.resize((32, 32), resample=Image.LANCZOS)
                R_obs = np.asarray(R_obs)
            else:
                R_obs = obs[1]

            R_action = RightPlayer.get_action(R_obs, timestep=R_timestep, train_mode=False)
            R_timestep += 1


        # Perform actions
        obs, rew, done, info = env.step((L_action, R_action))

        # Reward for a timestep is sum of rewards that happen during timestep
        L_reward += rew[0]
        R_reward += rew[1]

        frame += 1

        env.render()

        if done:
            L_reward = 0
            R_reward = 0
            episode += 1
            episode_counter += 1
            render = False
            frame = 0
            env.reset()


if __name__ == '__main__':
    # Setup Environment
    FRAME_RATE = 60
    env = PongEnv(framerate=FRAME_RATE)

    # Create Left Player
    # LeftPlayer = ActorCritic_Player(env=env, run_name="ActorCritic_vs_Hardcoded_Simple_FS1_EPE50", frameskip=1, isLeftPlayer=True, model=Gen_FC(8, env.action_space[0].n))
    # L_start_epoch = LeftPlayer.load('./saves/ActorCritic_vs_Hardcoded_FS1_EPE10-3/epo1200.save', load_optim=True)

    LeftPlayer = VPG_Player(env=env, run_name="VPG", frameskip=1, isLeftPlayer=True, model=Gen_FC(input_dim=8, output_dim=env.action_space[0].n, isValNet=False))

    # Create Right Player
    RightPlayer = HardcodedOpponent(isLeftPlayer=False, frameskip=1, model=DummyModel())
    R_start_epoch = 0

    test(env=env, LeftPlayer=LeftPlayer, RightPlayer=RightPlayer, framerate=FRAME_RATE, epochs=100000,
          episodes_per_epoch=50, R_start_epoch=R_start_epoch)
