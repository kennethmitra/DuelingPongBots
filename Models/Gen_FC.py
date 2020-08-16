import torch
import numpy as np

class Gen_FC(torch.nn.Module):

    def __init__(self, input_dim, output_dim, isValNet,useBias=True):

        # Note input type for Train.py
        super().__init__()
        self.obsIsImage = False
        self.isValNet = isValNet

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.useBias =useBias

        # Actor Specific
        self.actor_layer1 = torch.nn.Linear(input_dim, 64, bias=useBias)
        self.actor_layer2 = torch.nn.Linear(64, output_dim, bias=useBias)
        torch.nn.init.xavier_uniform_(self.actor_layer1.weight)
        torch.nn.init.xavier_uniform_(self.actor_layer2.weight)
        self.weight = torch.cat([self.actor_layer1.weight.data.view(input_dim, 64), self.actor_layer2.weight.data], 0)


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
        if not self.useBias:
            self.actor_layer1.weight = torch.nn.Parameter(self.weight[0:self.input_dim].view(64, self.input_dim))
            self.actor_layer2.weight = torch.nn.Parameter(self.weight[self.input_dim:self.input_dim + self.output_dim])
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
        print

        return actor_Logits, value