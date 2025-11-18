import torch
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import torch.nn as nn
from Q1 import frozenlake_agent
from numpy import random

class q_model3(nn.Module):
    def __init__(self, state_size, action_size):
        super(q_model3, self).__init__()
        self.fc1 = nn.Linear(state_size, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class q_model5(nn.Module):
    def __init__(self, state_size, action_size):
        super(q_model5, self).__init__()
        self.fc1 = nn.Linear(state_size, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    

class carpole_env(gym.Env):
    def __init__(self,env,model,learn_rate:float,epsilon:float,epsilon_decay,min_epsilon,gamma:float,episodes:int=1000):
        """Initialize the Q-learning agent for FrozenLake environment.
        Args:
            env (gym.Env): The FrozenLake environment.
            learn_rate (float): Learning rate for Q-learning updates.
            epsilon (float): Exploration rate (epsilon) for epsilon-greedy policy.
            gamma (float): Discount factor for future rewards.
            episode (int): Number of episodes for training.
        """
        self.env=env
        self.model=model
        self.lr=learn_rate
        self.epsilon=epsilon
        self.epsilon_decay=epsilon_decay
        self.min_epsilon=min_epsilon
        self.gamma=gamma
        self.episodes = episodes

    def choose_action(self,state):
        """Choose an action based on epsilon-greedy policy.
        Args:
            state (int): Current state of the environment.
        Returns:
            int: Chosen action.
        """
        if random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()  # Explore: random action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()  # Exploit: best action from Q-network
        return action
    
    def update_q_value(self,state,action,reward,next_state,terminal,optimizer,loss_fn):
        """Update the Q-value for a given state-action pair.
        Args:
            state (int): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (int): Next state after taking the action.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        q_values = self.model(state_tensor)
        next_q_values = self.model(next_state_tensor)

        target = reward if terminal else reward + self.gamma * torch.max(next_q_values).item()

        target_f = q_values.clone().detach()
        target_f[0][action] = target

        optimizer.zero_grad()
        loss = loss_fn(q_values, target_f)
        loss.backward()
        optimizer.step()