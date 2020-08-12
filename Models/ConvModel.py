import torch
import torch.nn.functional as F
import numpy as np

class ConvModel(torch.nn.Module):
    def __init__(self, output_dim, side_length):
        # Shared conv layers for feature extraction
        super(ConvModel, self).__init__()
        f1 = 64
        w1 = 5
        f2 = 32
        self.conv1 = torch.nn.Conv2d(1, f1, w1)
        size1 = (side_length - w1) + 1
        self.conv2 = torch.nn.Conv2d(f1, f2, w1)
        size1 = (size1 - w1) + 1
        self.fc_size = f2 * size1 * size1

        # Actor Specific
        self.actor_layer1 = torch.nn.Linear(self.fc_size, 64)
        self.actor_layer2 = torch.nn.Linear(64, output_dim)
        torch.nn.init.xavier_uniform_(self.actor_layer1.weight)
        torch.nn.init.xavier_uniform_(self.actor_layer2.weight)

        # Critic Specific
        self.critic_layer1 = torch.nn.Linear(self.fc_size, 64)
        self.critic_layer2 = torch.nn.Linear(64, 1)
        torch.nn.init.xavier_uniform_(self.critic_layer1.weight)
        torch.nn.init.xavier_uniform_(self.critic_layer2.weight)

    def forward(self, obs):
        """
        Compute action distribution and value from an observation
        :param obs: observation with len obs_cnt
        :return: Action distribution (Categorical) and value (tensor)
        """

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        # Add batch dimension and channel dimension (256, 256) -> (1, 1, 256, 256) | (n, c, h, w)
        obs = torch.unsqueeze(obs, 0)
        obs = torch.unsqueeze(obs, 0)

        # Separate Actor and Critic Networks
        obs = self.conv1(obs)
        obs = F.relu(obs)
        obs = self.conv2(obs)
        obs = F.relu(obs)
        obs = obs.view(-1, self.fc_size)

        # Actor Specific
        actor_intermed = self.actor_layer1(obs)
        actor_intermed = torch.nn.Tanh()(actor_intermed)
        actor_Logits = self.actor_layer2(actor_intermed)

        # Critic Logits
        critic_intermed = self.critic_layer1(obs)
        critic_intermed = torch.nn.Tanh()(critic_intermed)
        value = self.critic_layer2(critic_intermed)

        return actor_Logits, value