import torch
import numpy as np

class Gen_FC(torch.nn.Module):

    def __init__(self, input_dim, output_dim, isValNet):

        # Note input type for Train.py
        super().__init__()
        self.obsIsImage = False
        self.isValNet = isValNet

        # Actor Specific
        self.actor_layer1 = torch.nn.Linear(input_dim, 64)
        self.actor_layer2 = torch.nn.Linear(64, output_dim)
        torch.nn.init.xavier_uniform_(self.actor_layer1.weight)
        torch.nn.init.xavier_uniform_(self.actor_layer2.weight)

        if self.isValNet:
            # Critic Specific
            self.critic_layer1 = torch.nn.Linear(input_dim, 64)
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

        # Actor Specific
        actor_intermed = self.actor_layer1(obs)
        actor_intermed = torch.nn.Tanh()(actor_intermed)
        actor_Logits = self.actor_layer2(actor_intermed)

        value = None
        if self.isValNet:
            # Critic Logits
            critic_intermed = self.critic_layer1(obs)
            critic_intermed = torch.nn.Tanh()(critic_intermed)
            value = self.critic_layer2(critic_intermed)

        return actor_Logits, value