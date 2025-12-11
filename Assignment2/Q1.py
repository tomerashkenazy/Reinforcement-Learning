import torch
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os

# Policy Network - basic reinforce
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

# Value Network - advantage astimete
class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReinforceAgent:
    def __init__(self, env, lr=0.001, gamma=0.99, advantage=False, episodes=1000, early_stop=True):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.advantage = advantage
        self.episodes = episodes
        self.early_stop = early_stop
        
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        
        # Initialize Policy Network
        self.policy_net = PolicyNetwork(self.state_size, self.action_size)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Initialize Value Network if using advantage
        if self.advantage:
            self.value_net = ValueNetwork(self.state_size)
            self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr)
            
        self.episode_rewards = [] # Track rewards over episodes
        self.training_log = []

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state_tensor)
        
        # Create a categorical distribution to sample an action
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action)

    def calculate_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns

    def update_basic(self, rewards, log_probs):
        returns = self.calculate_returns(rewards)
        
        policy_loss = []
        for log_prob, G_t in zip(log_probs, returns):
            policy_loss.append(-log_prob * G_t)
        
        self.optimizer_policy.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer_policy.step()

    def update_with_baseline(self, rewards, log_probs, states):
        returns = self.calculate_returns(rewards)
        states_tensor = torch.FloatTensor(np.array(states))
        
        # Get baseline values V(s)
        values = self.value_net(states_tensor).squeeze()
        
        # Calculate Advantage: A_t = G_t - V(s_t)
        advantages = returns - values.detach()
        
        policy_loss = []
        for log_prob, adv in zip(log_probs, advantages):
            policy_loss.append(-log_prob * adv)
            
        self.optimizer_policy.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer_policy.step()
        
        # Update Value Network
        value_loss = nn.MSELoss()(values, returns)
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()

    def train(self):
        print(f"Starting training (Baseline: {self.advantage})...")
        
        for episode in range(self.episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            
            log_probs = []
            rewards = []
            states = []

            while not done:
                # Store state for baseline calculation
                states.append(state)
                
                # Select action
                action, log_prob = self.select_action(state)
                
                # Step environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                log_probs.append(log_prob)
                rewards.append(reward)
                total_reward += reward
                
                state = next_state

            # Update policy (and value function) at the end of the episode
            if self.advantage:
                self.update_with_baseline(rewards, log_probs, states)
            else:
                self.update_basic(rewards, log_probs)
            
            self.episode_rewards.append(total_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode+1}/{self.episodes} | Last 100 Avg Reward: {avg_reward:.2f}")
                self.training_log.append({
                            "episode": episode + 1,
                            "avg_reward_last100": float(avg_reward),
                            "solved (>=475)": avg_reward >= 475
                        })
            
                self.df = pd.DataFrame(self.training_log)
                
                if avg_reward >= 475:
                    print(f"Solved in episode {episode+1}!")
                    if self.early_stop:
                        break # Optional: stop if solved

        return self.episode_rewards
    
    def plot_results(self, save_dir="Assignment2/plots_A2_Q1", model_name="reinforce"):
        # Correctly construct the save directory path
        save_dir = Path(save_dir) / model_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save Dataframe of training log
        self.df.to_csv(save_dir / "training_log.csv", index=False)
        
        if len(self.episode_rewards) > 0:
            episodes = np.arange(1, len(self.episode_rewards) + 1)

            plt.figure()
            plt.plot(episodes, self.episode_rewards, linewidth=0.6)  # Adjusted line width
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(f"Reward per Episode [{model_name}]")
            plt.grid(True)

            plt.savefig(save_dir / "episode_rewards.png", dpi=300, bbox_inches='tight')
            plt.show()

def evaluate_agent(env, agent, eval_episodes=10):
    total_rewards = []
    for episode in range(eval_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            # For evaluation, we can still sample or take argmax. 
            # Standard REINFORCE usually samples, but argmax is often used for eval.
            # Here we stick to sampling as the policy is stochastic.
            action, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

if __name__ == "__main__":
    # Hyperparameters
    learn_rate = 0.001
    gamma = 0.99
    episodes = 1250
    expiriment_name = "1250_episodes"

    # Environment setup
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    
    # --- 1. Basic REINFORCE ---
    print("=== Training Basic REINFORCE ===")
    agent_basic = ReinforceAgent(
        env=env,
        lr=learn_rate,
        gamma=gamma,
        episodes=episodes,
        advantage=False
    )
    rewards_basic = agent_basic.train()
    agent_basic.plot_results(save_dir="Assignment2/plots_A2_Q1", model_name=f"{expiriment_name}/Basic_REINFORCE")
    
    # --- 2. REINFORCE with Baseline (Advantage) ---
    print("\n=== Training REINFORCE with Baseline ===")
    agent_baseline = ReinforceAgent(
        env=env,
        lr=learn_rate,
        gamma=gamma,
        episodes=episodes,
        advantage=True
    )
    rewards_baseline = agent_baseline.train()
    agent_baseline.plot_results(save_dir="Assignment2/plots_A2_Q1", model_name=f"{expiriment_name}/REINFORCE_with_Baseline")
    
    # Compare
    # Ensure the directory exists before saving the comparison plot
    comparison_dir = Path(f"Assignment2/plots_A2_Q1/{expiriment_name}")
    comparison_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(pd.Series(rewards_basic).rolling(50).mean(), label='Basic REINFORCE')
    plt.plot(pd.Series(rewards_baseline).rolling(50).mean(), label='REINFORCE + Baseline')
    plt.xlabel('Episode')
    plt.ylabel('Rolling Avg Reward (50)')
    plt.title('Comparison of REINFORCE versions')
    plt.legend()
    plt.savefig(comparison_dir / "comparison.png")
    plt.show()
