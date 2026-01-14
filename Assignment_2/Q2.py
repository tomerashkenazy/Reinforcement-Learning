import torch
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os
import time
from Q1 import ReinforceAgent

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    
    
class ActorCriticAgent:
    def __init__(self, env, alpha_theta=1e-3, alpha_w=1e-2, gamma=0.99, episodes=1000, early_stop=True):
        self.env = env
        self.gamma = gamma
        self.episodes = episodes
        self.early_stop = early_stop

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        # Policy & Value networks
        self.policy_net = PolicyNetwork(self.state_size, self.action_size).to(device)
        self.value_net  = ValueNetwork(self.state_size).to(device)

        self.optim_policy = optim.Adam(self.policy_net.parameters(), lr=alpha_theta)
        self.optim_value  = optim.Adam(self.value_net.parameters(), lr=alpha_w)

        self.episode_rewards = []


    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.policy_net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


    def train(self):
        print("Starting Actor–Critic training...")

        self.training_log = []   # store progress logs
        self.episode_times = []   # track time for each episode

        for episode in range(self.episodes):
            episode_start_time = time.time()

            state, _ = self.env.reset()
            done = False
            total_reward = 0

            I = 1.0   # episodic importance factor

            while not done:

                # Choose action
                action, log_prob = self.select_action(state)

                # Step in environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                # Prepare tensors
                s  = torch.FloatTensor(state).to(device)
                s_next = torch.FloatTensor(next_state).to(device)

                # Value(s)
                value_s = self.value_net(s)

                # Value(s')
                if done:
                    value_s_next = torch.tensor(0.0).to(device)
                else:
                    value_s_next = self.value_net(s_next)

                # -------- TD Error (δ) --------
                # delta must be a PYTORCH tensor
                delta = reward + self.gamma * value_s_next.detach() - value_s.detach()

                # -------- POLICY UPDATE --------
                # Objective:  maximize I * δ * logπ  → minimize negative 
                policy_loss = -(I * delta * log_prob)

                self.optim_policy.zero_grad()
                policy_loss.backward()
                self.optim_policy.step()

                # -------- VALUE UPDATE --------
                # critic uses δ = R + γ v(S') − v(S) 
                delta_value = reward + self.gamma * value_s_next.detach() - value_s
                value_loss = (I * delta_value).pow(2)   # MSE form

                self.optim_value.zero_grad()
                value_loss.backward()
                self.optim_value.step()

                # Update episodic weighting factor I ← γI
                I *= self.gamma

                state = next_state

            # Record episode time and reward
            episode_time = time.time() - episode_start_time
            self.episode_times.append(episode_time)
            self.episode_rewards.append(total_reward)

            # -------- LOGGING (unchanged) --------
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                total_time_100 = np.sum(self.episode_times[-100:])
                print(f"Episode {episode+1}/{self.episodes} | Last 100 Avg Reward: {avg_reward:.2f} | Time for 100 Episodes: {total_time_100:.2f}s")

                self.training_log.append({
                    "episode": episode + 1,
                    "avg_reward_last100": float(avg_reward),
                    "time_last100_episodes": float(total_time_100),
                    "solved (>=475)": avg_reward >= 475
                })

                self.df = pd.DataFrame(self.training_log)

                if avg_reward >= 475:
                    print(f"Solved in episode {episode+1}!")
                    if self.early_stop:
                        break 

        return self.episode_rewards
    
    def plot_results(self, save_dir="plots_A2_Q2", model_name="actor_critic"):
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
            plt.close()

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
    learn_rate_b = [0.001]#[0.01, 0.001, 0.0001]
    learn_rate = [0.0001]#[0.01, 0.001, 0.0001]
    gamma = [0.99]#[0.99, 0.95]
    episodes = 1500
    early_stop = True
    

    # Add a DataFrame to store grid search results
    grid_search_results = []

    for lr in learn_rate:
        for gamma_val in gamma:
            for lrb in learn_rate_b:
                expiriment_name = f"{episodes}_episodes,_lr_{lr}_lrb_{lrb}_gamma_{gamma_val}"
                base_dir = Path("Assignment2/plots_A2_Q2")

                # Environment setup
                env_name = 'CartPole-v1'
                env = gym.make(env_name)

                # ----- Actor Critic -----
                print("=== Training Actor-Critic ===")
                agent_actor_critic = ActorCriticAgent(
                    env=env,
                    alpha_theta=lr,
                    alpha_w=lrb,  # Using the same learning rate for simplicity
                    gamma=gamma_val,
                    episodes=episodes,
                    early_stop=early_stop
                )
                rewards_actor_critic = agent_actor_critic.train()
                agent_actor_critic.plot_results(save_dir=base_dir, model_name=f"{expiriment_name}/actor_critic")

                # --- Basic REINFORCE ---
                print("=== Training Basic REINFORCE ===")
                agent_basic = ReinforceAgent(
                    env=env,
                    lr=lr,
                    gamma=gamma_val,
                    episodes=episodes,
                    advantage=False,
                    early_stop=early_stop
                )
                rewards_basic = agent_basic.train()

                # --- REINFORCE with Baseline (Advantage) ---
                print("\n=== Training REINFORCE with Baseline ===")
                agent_baseline = ReinforceAgent(
                    env=env,
                    lr=lr,
                    gamma=gamma_val,
                    episodes=episodes,
                    advantage=True,
                    early_stop=early_stop
                )
                rewards_baseline = agent_baseline.train()

                # Compare and log results
                comparison_dir = Path(base_dir / expiriment_name)
                comparison_dir.mkdir(parents=True, exist_ok=True)

                plt.figure(figsize=(10, 6))
                plt.plot(pd.Series(rewards_basic).rolling(50).mean(), label='Basic REINFORCE')
                plt.plot(pd.Series(rewards_baseline).rolling(50).mean(), label='REINFORCE + Baseline')
                plt.plot(pd.Series(rewards_actor_critic).rolling(50).mean(), label='Actor-Critic')
                plt.xlabel('Episode')
                plt.ylabel('Rolling Avg Reward')
                plt.title('Comparison of REINFORCE versions')
                plt.legend()
                plt.savefig(comparison_dir / "comparison.png")
                plt.close()

                # Log results for grid search
                for agent_name, rewards in zip(
                    ['Basic REINFORCE', 'REINFORCE + Baseline', 'Actor-Critic'],
                    [rewards_basic, rewards_baseline, rewards_actor_critic]
                ):
                    solved_episode = len(rewards)
                    grid_search_results.append({
                        "Agent": agent_name,
                        "Learning Rate": lr,
                        "Learning rate b": lrb,
                        "Gamma": gamma_val,
                        "Episodes": episodes,
                        "Solved Episode": solved_episode
                    })

    # Save grid search results to CSV
    grid_search_df = pd.DataFrame(grid_search_results)
    grid_search_df.sort_values(by="Solved Episode", inplace=True, na_position='last')
    grid_search_df.to_csv("Assignment2/plots_A2_Q2/grid_search_results.csv", index=False)
    print("Grid search results saved to Assignment2/plots_A2_Q2/grid_search_results.csv")