from Models.HumanPlayer import HumanPlayer
from Models.HardcodedOpponent import HardcodedOpponent
from Models.GenAlg import GenAlg
from Models.ActorCritic import ActorCritic
from PongEnv import PongEnv
from PIL import  Image
import matplotlib.pyplot as plt
import numpy as np
import time
from utils.GIF_Recorder import GIF_Recorder

def train(env, LeftPlayer, RightPlayer, framerate=-1, epochs=10, episodes_per_epoch=3, L_start_epoch=0, R_start_epoch=0):
    """
    Takes in two players. Feeds players observations and gets actions from player after player.frameskip frames
    Repeats previous action until a new action is obtained
    If framerate is -1, runs with unlimited framerate. If framerate is > 0
    """
    #Modoulus that defines saving freq
    obs = env.reset()
    SAVE_MOD = 50
    L_recorder = GIF_Recorder(LeftPlayer.save_path)

    for epoch in range(epochs):

        # Render if first episode of epoch
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


        last_time = time.perf_counter()

        last_obs = None

        print(f"============Epoch {epoch} out of {epochs}============")
        episode_counter = 0

        # Get epoch data
        while episode_counter < episodes_per_epoch:

            curTime = time.perf_counter()
            #print("Frame Time: ", curTime - last_time)
            last_time = curTime

            # Get action from models if needed
            if frame % LeftPlayer.frameskip == 0:
                # Store timestep
                if frame != 0:
                    LeftPlayer.end_tstep(reward=L_reward)  # Store reward for last timestep
                    L_reward = 0

                # Downsample obs if needed
                L_obs = None
                if LeftPlayer.obsIsImage:
                    L_obs = Image.fromarray(obs[0])
                    L_obs = L_obs.resize((32, 32), resample=Image.LANCZOS)
                    #plt.imshow(L_obs)
                    #plt.pause(0.000000001)


                    L_obs = np.asarray(L_obs)
                    last_obs = L_obs

                else:
                    L_obs = obs[1]

                # Get new action
                L_action = LeftPlayer.get_action(L_obs, timestep=L_timestep, train_mode=True) # get_action stores obs, logprobs, and any other intermediary stuff
                L_timestep += 1
            if frame % RightPlayer.frameskip == 0:
                # Store timestep
                if frame != 0:
                    RightPlayer.end_tstep(reward=R_reward)
                    R_reward = 0

                # Downsample obs if needed
                R_obs = None
                if RightPlayer.obsIsImage:
                    R_obs = Image.fromarray(obs[0])
                    R_obs = R_obs.resize((32, 32), resample=Image.LANCZOS)
                    R_obs = np.asarray(R_obs)
                else:
                    R_obs = obs[1]

                R_action = RightPlayer.get_action(R_obs, timestep=R_timestep, train_mode=True)
                R_timestep += 1


            # Record obs to GIF
            if (epoch + L_start_epoch) % SAVE_MOD == 0:
                L_recorder.add_image(env.game.getScreenRGB())
            if (epoch + R_start_epoch) % SAVE_MOD == 0:
                pass

            # Perform actions
            obs, rew, done, info = env.step((L_action, R_action))

            # Reward for a timestep is sum of rewards that happen during timestep
            L_reward += rew[0]
            R_reward += rew[1]


            frame += 1

            # if render:
            #     env.render()
            env.render()

            if done:
                LeftPlayer.end_tstep(reward=L_reward, end_episode=True) # Store reward for last timestep
                RightPlayer.end_tstep(reward=R_reward, end_episode=True)
                episode += 1
                episode_counter += 1
                print(f"---------Episode {episode_counter} out of {episodes_per_epoch} in epoch {epoch} out of {epochs}---------")
                render = False
                frame = 0
                env.reset()

        if epoch % SAVE_MOD == 0:
            L_recorder.make_gif(f"{epoch + L_start_epoch}.gif")
            LeftPlayer.save(epoch + L_start_epoch)
            RightPlayer.save(epoch + R_start_epoch)

        # Train on epoch data
        LeftPlayer.train_batch(epoch + L_start_epoch)
        RightPlayer.train_batch(epoch + R_start_epoch)


if __name__ == '__main__':

    # Setup Environment
    FRAME_RATE = -1
    env = PongEnv(framerate=FRAME_RATE)

    # Create Left Player
    LeftPlayer = ActorCritic(env=env, run_name="ActorCritic_vs_Hardcoded_FS1_EPE10", frameskip=1, isLeftPlayer=True)
    L_start_epoch = LeftPlayer.load('./saves/ActorCritic_vs_Hardcoded_FS1_EPE10-3/epo1200.save', load_optim=True)

    # Create Right Player
    RightPlayer = HardcodedOpponent(isLeftPlayer=False, frameskip=1)
    R_start_epoch = 0

    train(env=env, LeftPlayer=LeftPlayer, RightPlayer=RightPlayer, framerate=FRAME_RATE, epochs=100000, episodes_per_epoch=10, L_start_epoch=L_start_epoch, R_start_epoch=R_start_epoch)
