from torch.utils.tensorboard import SummaryWriter
import torch
import time
import numpy as np

class Logger:
    def __init__(self, run_name=None, refresh_secs=30):
        log_loc = f"runs/{run_name}" if run_name is not None else None
        self.writer = SummaryWriter(log_dir=log_loc, flush_secs=refresh_secs)
        self.last_episode_time = time.perf_counter()
        self.loggedModel=False

    def log_hparams(self, ENVIRONMENT=None, SEED=None, model=None, optimizer=None, LEARNING_RATE=None, DISCOUNT_FACTOR=None, ENTROPY_COEFF=None, activation_func=None,
                    tsteps_per_epoch=None, normalize_rewards=None, normalize_advantages=None, clip_grad=None, notes=None, display=True):
        self.writer.add_text("Hyperparams/Environment", ENVIRONMENT, 0)
        self.writer.add_text("Hyperparams/Seed", str(SEED), 0)
        self.writer.add_text("Hyperparams/Model", str(model), 0)
        self.writer.add_text("Hyperparams/Optimizer", str(optimizer), 0)
        self.writer.add_text("Hyperparams/Learning_Rate", str(LEARNING_RATE), 0)
        self.writer.add_text("Hyperparams/Discount_Factor", str(DISCOUNT_FACTOR), 0)
        self.writer.add_text("Hyperparams/Entropy_coefficient", str(ENTROPY_COEFF), 0)
        self.writer.add_text("Hyperparams/Activation_Function", str(activation_func), 0)
        self.writer.add_text("Hyperparams/Timesteps_per_epoch", str(tsteps_per_epoch), 0)
        self.writer.add_text("Hyperparams/Normalize_rewards", str(normalize_rewards), 0)
        self.writer.add_text("Hyperparams/Normalize_advantages", str(normalize_advantages), 0)
        self.writer.add_text("Hyperparams/clip_grad", str(clip_grad), 0)
        self.writer.add_text("Hyperparams/Notes", notes, 0)

        # self.writer.add_graph(model, torch.randn(1, 6400).to(model.device))

        if display:
            print('------------------------------Hyperparameters--------------------------------------------------')
            print(f'ENVIRONMENT: {ENVIRONMENT}')
            print(f'SEED: {SEED}')
            print(f'MODEL: {model}')
            print(f'OPTIMIZER: {optimizer}')
            print(f'LEARNING_RATE: {LEARNING_RATE}')
            print(f'DISCOUNT_FACTOR: {DISCOUNT_FACTOR}')
            print(f'ENTROPY_COEFF: {ENTROPY_COEFF}')
            print(f'ACTIVATION_FUNC: {activation_func}')
            print(f'NORMALIZE_REWARDS: {normalize_rewards}')
            print(f'NORMALIZE_ADVANTAGES: {normalize_advantages}')
            print(f'CLIP_GRAD: {clip_grad}')
            print(f'NOTES: {notes}')
            print('-----------------------------------------------------------------------------------------------')

    def log_epoch(self, epoch_no, epoch_info):

        if 'actor_loss' in epoch_info:
            self.writer.add_scalar("Loss/Actor_Loss", epoch_info['actor_loss'], epoch_no)
        if 'critic_loss' in epoch_info:
            self.writer.add_scalar("Loss/Critic_Loss", epoch_info['critic_loss'], epoch_no)
        if 'entropy_loss' in epoch_info:
            self.writer.add_scalar("Loss/Entropy_Loss", epoch_info['entropy_loss'], epoch_no)
        if 'entropy_avg' in epoch_info:
            self.writer.add_scalar("Loss/Entropy", epoch_info['entropy_avg'], epoch_no)
        if 'total_loss' in epoch_info:
            self.writer.add_scalar("Loss/Total_Loss", epoch_info['total_loss'], epoch_no)

        if 'disc_rews' in epoch_info:
            self.writer.add_histogram("Train/disc_rews", epoch_info['disc_rews'], epoch_no)
        if 'pred_values' in epoch_info:
            self.writer.add_histogram("Train/pred_values", torch.stack(epoch_info['pred_values']), epoch_no)
        if 'advantages' in epoch_info:
            self.writer.add_histogram("Train/Advantages", epoch_info['advantages'], epoch_no)
        if 'raw_rew' in epoch_info:
            self.writer.add_histogram("Train/Raw_Rewards", epoch_info['raw_rew'], epoch_no)

        if 'avg_ep_len' in epoch_info:
            self.writer.add_scalar("Metrics/Episode_Length", epoch_info['avg_ep_len'], epoch_no)
        if 'epoch_timesteps' in epoch_info:
            self.writer.add_scalar("Metrics/Actual_Timesteps_per_Epoch", epoch_info['epoch_timesteps'], epoch_no)
        if 'num_episodes' in epoch_info:
            self.writer.add_scalar("Metrics/Episodes_per_Epoch", epoch_info['num_episodes'], epoch_no)

        if 'update_time' in epoch_info:
            self.writer.add_scalar('Time/Update_Time', epoch_info['update_time'], epoch_no)
        if 'avg_ep_raw_rew' in epoch_info:
            self.writer.add_scalar('Metrics/Avg_Raw_Reward', epoch_info['avg_ep_raw_rew'], epoch_no)
        if 'avg_ep_disc_rew' in epoch_info:
            self.writer.add_scalar('Metrics/Avg_Disc_Reward', epoch_info['avg_ep_disc_rew'], epoch_no)

        elapsed_time = time.perf_counter() - self.last_episode_time
        self.writer.add_scalar('Time/Time_per_Epoch', elapsed_time, epoch_no)
        self.last_episode_time = time.perf_counter()

        print(f"Epoch: {epoch_no}, Entropy: {epoch_info['entropy_avg'] if ('entropy_avg' in epoch_info) else 'NA'}, Raw Rew: {epoch_info['avg_ep_raw_rew'] if ('avg_ep_raw_rew' in epoch_info) else 'NA'}, "
              f"Disc Rew: {epoch_info['avg_ep_disc_rew'] if ('avg_ep_disc_rew' in epoch_info) else 'NA'}, Time: {elapsed_time}")
