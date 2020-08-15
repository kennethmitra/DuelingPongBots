# Dueling Pong Bots

Pong AI's trained with various reinforcement learning methods compete each other in a multiagent framework 
that allows RL Agent vs RL Agent, Human vs RL Agent, and Human vs Hardcoded AI interactions

RL Algorithms implemented: Advantage-Actor-Critic (A2C), Vanilla Policy Gradient (VPG)

Model Types: Convolutional networks, Feedforward networks

----------------------------------------------------
# Results

Running VPG (Red) vs ActorCritic (Blue) after training for 100 epochs with rewards for scoring and hitting the ball:

![VPG vs A2C with reward for hitting the ball](VPG_vs_A2C-rally-reward.gif)

Interestingly, both agents learn to cooperate with each other, rallying the ball to farm the reward for hitting the ball

Tensorboard VPG(Green) vs ActorCritic(Gray):
![VPG vs A2C with reward for hitting the ball Training Graph](VPG_vs_A2C-rally-reward-tensorboard.png)