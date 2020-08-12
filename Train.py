from Models.HumanPlayer import HumanPlayer
from Models.HardcodedOpponent import HardcodedOpponent
from Models.GenAlg import GenAlg
from Models.ActorCritic import ActorCritic
from PongEnv import PongEnv
from PIL import  Image
import matplotlib.pyplot as plt
import numpy as np

def train(env, LeftPlayer, RightPlayer, framerate=-1, epochs=10, episodes_per_epoch=3):
    """
    Takes in two players. Feeds players observations and gets actions from player after player.frameskip frames
    Repeats previous action until a new action is obtained
    If framerate is -1, runs with unlimited framerate. If framerate is > 0
    """
    #Modoulus that defines saving freq
    obs = env.reset()
    SAVE_MOD = 50

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
        while epoch < epochs:
            
            # Get epoch data
            while episode < episodes_per_epoch:

                # Get action from models if needed
                if frame % LeftPlayer.frameskip == 0:
                    # Store timestep
                    if frame != 0:
                        LeftPlayer.end_tstep(reward=L_reward)  # Store reward for last timestep

                    # Downsample obs if needed
                    L_obs = None
                    if LeftPlayer.obsIsImage:
                        L_obs = Image.fromarray(obs[0])
                        L_obs = L_obs.resize((32, 32), resample=Image.LANCZOS)
                        plt.imshow(L_obs)
                        plt.pause(0.001)
                        L_obs = np.asarray(L_obs)
                    else:
                        L_obs = obs[1]

                    # Get new action
                    L_action = LeftPlayer.get_action(L_obs, timestep=L_timestep, train_mode=True) # get_action stores obs, logprobs, and any other intermediary stuff
                    L_timestep += 1
                if frame % RightPlayer.frameskip == 0:
                    # Store timestep
                    if frame != 0:
                        RightPlayer.end_tstep(reward=R_reward)

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

                # Perform actions
                obs, rew, done, info = env.step((L_action, R_action))

                # Reward for a timestep is sum of rewards that happen during timestep
                L_reward += rew[0]
                R_reward += rew[1]


                frame += 1

                if render:
                    env.render()
                
                if done:
                    LeftPlayer.end_tstep(reward=L_reward, end_episode=True) # Store reward for last timestep
                    RightPlayer.end_tstep(reward=R_reward, end_episode=True)
                    episode += 1
                    render = False
                    frame = 0
                    env.reset()

                if epoch % SAVE_MOD:
                    LeftPlayer.save()
                    RightPlayer.save()
            
            # Train on epoch data
            LeftPlayer.train_batch(epoch)
            RightPlayer.train_batch(epoch)


if __name__ == '__main__':
    # m = MemoryGrabber()
    # m.clear_mem(0.85)

    FRAME_RATE = -1
    env = PongEnv(framerate=FRAME_RATE)
    train(env=env, LeftPlayer=ActorCritic(env=env, run_name="A2C_Algo", frameskip=1, isLeftPlayer=True), RightPlayer=HardcodedOpponent(isLeftPlayer=False, frameskip=5), framerate=FRAME_RATE, epochs=10, episodes_per_epoch=5)
