# Dueling Pong Bots

Pong AI's trained with various reinforcement learning methods compete each other in a multiagent framework 
that allows RL Agent vs RL Agent, Human vs RL Agent, and Human vs Hardcoded AI interactions

RL Algorithms implemented: Advantage-Actor-Critic (A2C), Vanilla Policy Gradient (VPG)

Model Types: Convolutional networks, Feedforward networks

----------------------------------------------------
# Results

Running VPG (Red) vs ActorCritic (Blue) after training for 100 epochs with rewards for scoring and hitting the ball:

![VPG vs A2C with reward for hitting the ball](VPG_vs_A2C-rally-reward.gif "*Interestingly, both agents learn to cooperate with each other, rallying the ball to farm the reward for hitting the ball*")


Tensorboard VPG(Green) vs ActorCritic(Gray):
![VPG vs A2C with reward for hitting the ball Training Graph](VPG_vs_A2C-rally-reward-tensorboard.png)

When the reward for hitting the ball is removed, the agents learn to score against each other.

![VPG vs A2C with no reward for hitting the ball](VPG_vs_A2C-150epochs-after-rally-reward.gif "*150 epochs after removing reward for touching the ball*")


The agents now play a zero sum game as seen in the rewards graph:
![VPG vs A2C no reward for hitting the ball training graph](VPG_vs_A2C-150epochs-after-rally-reward-tensorboard.gif)