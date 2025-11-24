import torch
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from collections import deque
import random as rand
from numpy import random

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 96)  # Shared layer

        # Separate streams for Advantage and Value
        self.advantage = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

        self.value = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        # Compute Advantage and Value streams
        advantage = self.advantage(x)
        value = self.value(x)

        # Combine Advantage and Value to get Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    

# Include the ReplayMemory class from above or define it here
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Samples a batch of transitions."""
        return rand.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class carpole_env(gym.Env):
    def __init__(self, env, model, learn_rate:float, epsilon:float, epsilon_decay, min_epsilon, gamma:float,memory_capacity = 10000,batch_size=64,target_update_freq=10,episodes:int=1000):
        # Initialization as in your prompt
        self.env = env
        self.model = model
        self.lr = learn_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.episodes = episodes
        # Added attributes for training
        self.memory_capacity = memory_capacity 
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.loss_tracking = [] #Track loss over training steps
        self.episode_rewards = [] # Track rewards over episodes
    def choose_action(self, state):
        # choose_action method as in your prompt
        if random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()  # Explore: random action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad(): # Disable gradient calculations for inference
                q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()  # Exploit: best action from Q-network
        return action
    
    # Updated update_q_value to use target network for stability
    def update_q_value(self, target_net, optimizer, loss_fn, transitions):
        """
        Updates the Q-network using a minibatch of transitions.
        """
        # Unpack the batch of transitions
        states, actions, rewards, next_states, terminals = zip(*transitions)

        # Convert to PyTorch Tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        terminals = torch.BoolTensor(terminals)
        
        # Compute Q(s_t, a) - the Q-values for the states and actions taken
        # We only care about the Q-value for the action that was actually taken
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute V(s_{t+1}) = max_a' Q'(s_{t+1}, a') for all non-terminal next states
        # The use of target_net for the next state value is the core of DQN
        # Target Q-values: Q'(s_{t+1}, a')
        next_q_values = target_net(next_states).max(1)[0].detach()

        # Compute the target Q-value: y_j = r_j + gamma * max_a' Q'(s_{t+1}, a')
        target_q_values = rewards + (self.gamma * next_q_values * (~terminals))
        
        # Perform gradient descent step on (y_j - Q(s_j, a_j))^2
        optimizer.zero_grad()
        loss = loss_fn(current_q_values, target_q_values)
        self.loss_tracking.append(loss.item())
        loss.backward()
        
        # Optional: Clip gradients
        # for param in self.model.parameters():
        #     param.grad.data.clamp_(-1, 1)
            
        optimizer.step()
        return loss.item()

    def train(self):
        """
        Main training loop for the DQN agent.
        """
        print("Starting training...")
        # 1. Initialize replay memory D
        memory = ReplayMemory(self.memory_capacity)
        
        # 2. Initialize action-value network Q (self.model)
        # 2. Initialize target network Q' (target_net) with the same weights
        target_net = type(self.model)(
            self.env.observation_space.shape[0], 
            self.env.action_space.n
        ).eval() # Set target network to evaluation mode
        target_net.load_state_dict(self.model.state_dict()) 

        # Setup optimizer and loss function
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        
        

        # 3. Loop for episode 1 to M do
        for episode in range(self.episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            steps_done = 0

            # Loop for t from 1 to T do
            while not done:
                # 4. Select action a_t with epsilon-greedy policy
                action = self.choose_action(state)

                # 5. Execute action a_t, observe reward r_t, state s_{t+1}, and termination d_t
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # 6. Store transition in D
                memory.push(state, action, reward, next_state, done)

                # 7. Update current state
                state = next_state
                total_reward += reward
                steps_done += 1


                # 8. Sample minibatch of transitions from D
                if len(memory) > self.batch_size:
                    transitions = memory.sample(self.batch_size)
                    
                    # 9. Compute target y_j and perform gradient descent step
                    self.update_q_value(target_net, optimizer, loss_fn, transitions)

                # 10. Update target network every C steps
                if steps_done % self.target_update_freq == 0:
                    target_net.load_state_dict(self.model.state_dict())

            self.episode_rewards.append(total_reward)  # Average reward per step

            
            if np.mean(self.episode_rewards[-100:]) >= 475:  # Solved condition for CartPole-v1
                print(f"\nSolved in episode {episode}!\n")
                self.episode_rewards.append(total_reward)
                print(f"Episode {episode+1}/{self.episodes} | Avg Reward: {total_reward/steps_done:.2f} | avg Loss: {np.mean(self.loss_tracking[-100:]):.4f}\n")
                print("Training complete.\n")

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{self.episodes} | Last 100 Avg Reward: {np.mean(self.episode_rewards[-100:]):.2f} | Last 100 avg Loss: {np.mean(self.loss_tracking[-100:]):.4f}")
            

            # Epsilon decay
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return self.episode_rewards

# Hyperparameters
LEARN_RATE = 1e-3
GAMMA = 0.99
EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.995
EPISODES = 2000

# Environment setup (CartPole has continuous observation space, discrete action space)
env_name = 'CartPole-v1'
env = gym.make(env_name,render_mode='human')
state_size = env.observation_space.shape[0] # 4 for CartPole: position, velocity, angle, angular velocity
action_size = env.action_space.n           # 2 for CartPole: push left or right

# Model instantiation
q_net = DuelingQNetwork(state_size, action_size) # Or q_model5

# Agent instantiation and training
agent = carpole_env(
    env=env,
    model=q_net,
    learn_rate=LEARN_RATE,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    min_epsilon=MIN_EPSILON,
    gamma=GAMMA,
    episodes=EPISODES
)

rewards = agent.train()