import torch
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import torch.nn as nn
from Q1 import frozenlake_agent
import random as rand
from collections import deque
from torch.optim import Adam
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_size, fc2_size, fc3_size, fc_adv_size, fc_val_size):
        super(DuelingQNetwork, self).__init__()
        # Save the provided layer sizes so external code (e.g. agents) can access them
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.fc3_size = fc3_size
        self.fc_adv_size = fc_adv_size
        self.fc_val_size = fc_val_size

        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)  # Shared layer

        # Separate streams for Advantage and Value
        self.advantage = nn.Sequential(
            nn.Linear(fc3_size, fc_adv_size),
            nn.ReLU(),
            nn.Linear(fc_adv_size, action_size)
        )

        self.value = nn.Sequential(
            nn.Linear(fc3_size, fc_val_size),
            nn.ReLU(),
            nn.Linear(fc_val_size, 1)
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

class cartpole_agent(gym.Env):
    def __init__(self, env, model, learn_rate:float=None, epsilon:float=None, epsilon_decay=None, min_epsilon=None, gamma:float=None,memory_capacity = 10000,batch_size=64,target_update_freq=10,episodes:int=1000,params=None):
        # Initialization as in your prompt
        if params is None:
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
        else:
            self.lr = params['learn_rate']
            self.epsilon = params['epsilon']
            self.epsilon_decay = params['epsilon_decay']
            self.min_epsilon = params['min_epsilon']
            self.gamma = params['gamma']
            self.episodes = params['episodes']
            # Added attributes for training
            self.memory_capacity = params['memory_capacity'] 
            self.batch_size = params['batch_size']
            self.target_update_freq = params['target_update_freq']

        self.env = env
        self.model = model
        self.model_name="dualing_model"
        self.file_name = 'plot_Q3_dualing_model'
        self.loss_tracking = [] #Track loss over training steps
        self.episode_rewards = [] # Track rewards over episodes
        self.training_log = []

    def choose_action(self, state):
        # choose_action method as in your prompt
        if rand.random() < self.epsilon:
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
            self.env.action_space.n,
            self.model.fc1_size,
            self.model.fc2_size,
            self.model.fc3_size,
            self.model.fc_adv_size,
            self.model.fc_val_size
        ).eval() # Set target network to evaluation mode

        target_net.load_state_dict(self.model.state_dict()) 

        # Setup optimizer and loss function
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        steps_done = 0
        

        # 3. Loop for episode 1 to M do
        for episode in range(self.episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            

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

            self.episode_rewards.append(total_reward)  # Total reward per epeisode

            

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{self.episodes} | Last 100 Avg Reward: {np.mean(self.episode_rewards[-100:]):.2f} | Last 100 avg Loss: {np.mean(self.loss_tracking[-100:]):.4f}")
                self.training_log.append({
                            "episode": episode + 1,
                            "avg_reward_last100": float(np.mean(self.episode_rewards[-100:])),
                            "avg_loss_last100": float(np.mean(self.loss_tracking[-100:])),
                            "solved (>=475)": np.mean(self.episode_rewards[-100:]) >= 475
                        })
                
                self.df = pd.DataFrame(self.training_log)

                if np.mean(self.episode_rewards[-100:]) >= 475:
                    self.filename = get_unique_filename(base_name=f"{self.model_name}_training_log", extension="csv", output_path='.')
                    # Save Dataframe of training log
                    self.df.to_csv(self.filename, index=False)

            # Epsilon decay
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return self.episode_rewards
    
    def plot_results(self, save_dir="plots_Q3", model_name="dualing_model"):
        #filename = get_unique_filename(base_name=f"{model_name}_training_log", extension="csv", output_path=save_dir)
        # Save Dataframe of training log
        self.df.to_csv(self.filename, index=False)
        # Create the folder if it doesn't exist
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)


        if len(self.loss_tracking) > 0:
            x_100 = np.arange(1,len(self.loss_tracking) + 1)

            plt.figure()
            plt.plot(x_100,self.loss_tracking)
            plt.xlabel("Training Steps")
            plt.ylabel("Loss")
            plt.title("Learning curve: loss over training steps")
            plt.grid(True)

            plt.savefig(save_dir / "loss_over_training_steps.png", dpi=300, bbox_inches='tight')
            plt.show()

        assert len(self.episode_rewards) == self.episodes, "Episode rewards length mismatch."
        
        if len(self.episode_rewards) > 0:
            episodes = np.arange(1, len(self.episode_rewards) + 1)

            plt.figure()
            plt.scatter(episodes, self.episode_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(f"Reward per Episode [{model_name}]")
            plt.grid(True)

            plt.savefig(save_dir / "episode_rewards.png", dpi=300, bbox_inches='tight')
            plt.show()

def evaluate_cartpole_agent_Q3(env, agent, eval_episodes=10):
    total_rewards = []
    for episode in range(eval_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        agent.epsilon = 0.0  # Disable exploration for evaluation
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
        total_rewards.append(total_reward)
    mean_reward = np.mean(total_rewards)
    return mean_reward

    
def get_unique_filename(base_name="training_log", extension="csv", output_path="."):
    """
    Generate a unique filename by appending a number if the file already exists.
    """
    counter = 0
    while True:
        filename = os.path.join(output_path, f"{base_name}{counter if counter > 0 else ''}.{extension}")
        if not os.path.exists(filename):
            return filename
        counter += 1
        
        
if __name__ == "__main__":
    # Hyperparameters
    learn_rate = 0.0001
    gamma = 0.99
    epsilon = 1
    min_epsilon = 0.1
    epsilon_decay = 0.99
    episodes = 2000
    memory_capacity = 50000
    batch_size = 128
    target_update_freq = 50
    fc1_size = 64
    fc2_size = 64
    fc3_size = 64
    fc_adv_size = 32
    fc_val_size = 32

    # Environment setup (CartPole has continuous observation space, discrete action space)
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0] # 4 for CartPole: position, velocity, angle, angular velocity
    action_size = env.action_space.n           # 2 for CartPole: push left or right

    # Model instantiation
    q_net = DuelingQNetwork(state_size, action_size,fc1_size=fc1_size,fc2_size=fc2_size,fc3_size=fc3_size,fc_adv_size=fc_adv_size,fc_val_size=fc_val_size) 

    # Agent instantiation and training
    agent = cartpole_agent(
        env=env,
        model=q_net,
        learn_rate=learn_rate,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        gamma=gamma,
        episodes=episodes,
        memory_capacity=memory_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq
    )

    rewards = agent.train() 
    agent.plot_results(save_dir="plots_Q3_dueling_model-1",model_name="dueling_model-1")
