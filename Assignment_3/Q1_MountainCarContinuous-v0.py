import torch
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SOLVE_THRESHOLDS = {
    "CartPole-v1": 475,
    "Acrobot-v1": -100,
    "MountainCarContinuous-v0": 90,
}


# actor Network - basic reinforce
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128,hidden_size2=32): # Modified hidden size to 32 from 128
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.fc3 = nn.Linear(hidden_size2, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        #x = self.fc3(x)
        return torch.softmax(x, dim=-1)
    
# critic Network - value function approximator
class CriticNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=128,hidden_size2=32): # Modified hidden size to 32 from 128
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        #self.fc3 = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        #x = self.fc3(x)
        return x
    
    
class ActorCriticAgent:
    def __init__(self, env, alpha_theta=1e-3, alpha_w=1e-2, gamma=0.99, episodes=1000, early_stop=True,env_name='CartPole-v1'):
        self.env = env
        self.gamma = gamma
        self.episodes = episodes
        self.early_stop = early_stop
        self.df = pd.DataFrame()

        self.state_size = 6 # Modified for Acrobot-v1 the biggest state space
        self.action_size = 3 # Modified for Acrobot-v1 the biggest action space

        # Policy & Value networks
        self.actor_network = ActorNetwork(self.state_size, self.action_size).to(device)
        self.critic_network  = CriticNetwork(self.state_size).to(device)

        self.optim_policy = optim.Adam(self.actor_network.parameters(), lr=alpha_theta)
        self.optim_value  = optim.Adam(self.critic_network.parameters(), lr=alpha_w)
        self.episode_rewards = []
        self.env_name = env_name


    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.actor_network(state_tensor)
        
        if self.env_name == 'CartPole-v1':
            mask = torch.tensor([1.0, 1.0, 0.0]).to(device)
            probs = probs * mask
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
            
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


    def train(self):
        print(f"Starting {self.env_name} training...")

        self.training_log = []   # store progress logs

        for episode in range(self.episodes):

            state, _ = self.env.reset()
            if self.env_name == 'CartPole-v1':
                state = np.pad(state, (0, 2), mode='constant')
            if self.env_name == 'MountainCarContinuous-v0':
                state = np.pad(state, (0, 4), mode='constant')
                state[1] *=100
                
            done = False
            total_reward = 0

            I = 1.0   # episodic importance factor

            while not done:

                # Choose action
                action, log_prob = self.select_action(state)

                # Step in environment
                next_state, reward, terminated, truncated, _ = self.env.step([action])
                done = terminated or truncated
                total_reward += reward
                
                if self.env_name == 'CartPole-v1':
                    next_state = np.pad(next_state, (0, 2), mode='constant')
                if self.env_name == 'MountainCarContinuous-v0':
                    next_state = np.pad(next_state, (0, 4), mode='constant')
                    next_state[1] *=100
                if self.env_name == 'MountainCarContinuous-v0':
                    # Give a bonus for velocity. This encourages the car to rock back and forth.
                    # We multiply by 10 or 100 because velocity is very small (~0.07)
                    velocity_bonus = abs(next_state[1]) * 100 
                    
                    # Give a bonus for height (position). 
                    # -0.5 is roughly the bottom. > 0.5 is the goal.
                    height_bonus = (next_state[0])
                    
                    # Create a "shaped" reward for training
                    # We keep the original 'reward' (-1) to punish wasted time, 
                    # but add bonuses to guide it.
                    shaped_reward = reward + velocity_bonus + height_bonus
                else:
                    shaped_reward = reward
                # Prepare tensors
                s  = torch.FloatTensor(state).to(device)
                s_next = torch.FloatTensor(next_state).to(device)

                # Value(s)
                value_s = self.critic_network(s)

                # Value(s')
                if done:
                    value_s_next = torch.tensor(0.0).to(device)
                else:
                    value_s_next = self.critic_network(s_next)

                # -------- TD Error (δ) --------
                # delta must be a PYTORCH tensor
                delta = shaped_reward + self.gamma * value_s_next.detach() - value_s.detach()

                # -------- actor UPDATE --------
                # Objective:  maximize I * δ * logπ  → minimize negative 
                actor_loss = -(I * delta * log_prob)

                self.optim_policy.zero_grad()
                actor_loss.backward()
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

            # Store reward for this episode
            self.episode_rewards.append(total_reward)

            # -------- LOGGING (unchanged) --------
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode+1}/{self.episodes} | Last 100 Avg Reward: {avg_reward:.2f}")

                threshold = SOLVE_THRESHOLDS[self.env_name]
                solved = avg_reward >= threshold

                self.training_log.append({
                    "episode": episode + 1,
                    "avg_reward_last100": float(avg_reward),
                    "solved": solved,
                    "solve_threshold": threshold,
                    'episode_rewards': self.episode_rewards
                })

                if solved:
                    print(f"{self.env_name} solved in episode {episode+1}!")
                    if self.early_stop:
                        break

        return self.episode_rewards
    
    def plot_results(self, save_dir="plots_A2_Q2", model_name="actor_critic"):
        # Correctly construct the save directory path
        save_dir = Path(save_dir) / model_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save Dataframe of training log
        self.df = pd.DataFrame(self.training_log)
        self.df.to_csv(save_dir / "training_log.csv", index=False)
        print(f"Training log saved to {save_dir / 'training_log.csv'}")
        
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
            
    def save_model(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor_network.state_dict(), save_dir / "actor_net.pth")
        torch.save(self.critic_network.state_dict(), save_dir / "critic_net.pth")


if __name__ == "__main__":
    # Hyperparameters
    learn_rate_b =  [0.01,0.0001]
    learn_rate = [0.001,0.0001]
    gamma = [1,0.99,0.999]
    episodes = 1000
    early_stop = True
    env_names = ['MountainCarContinuous-v0']#['Acrobot-v1', 'MountainCar-v0']
    grid_acrobot_results = []
    grid_mountaincar_results = []
    


    for lr in learn_rate:
        for gamma_val in gamma:
            for lrb in learn_rate_b:
                for env_name in env_names:
                    expiriment_name = f"{episodes}_episodes,_lr_{lr}_lrb_{lrb}_gamma_{gamma_val}_env_{env_name}"
                    base_dir = Path("Assignment3/plots_Q1")

                    # Environment setup
                    
                    env = gym.make(env_name)

                    
                    agent_actor_critic = ActorCriticAgent(
                        env=env,
                        alpha_theta=lr,
                        alpha_w=lrb,  # Using the same learning rate for simplicity
                        gamma=gamma_val,
                        episodes=episodes,
                        early_stop=early_stop,
                        env_name=env_name
                    )
                    rewards = agent_actor_critic.train()
                    agent_actor_critic.plot_results(save_dir=base_dir, model_name=f"{expiriment_name}/actor_critic")
                    agent_actor_critic.save_model(save_dir=base_dir / expiriment_name / f"actor_critic_{env_name}")
                    # if env_name == 'CartPole-v1':
                    #     grid_carpole_results.append({
                    #         "Learning Rate": lr,
                    #         "Learning rate b": lrb,
                    #         "Gamma": gamma_val,
                    #         "Episodes": episodes,
                    #         "Solved Episode": len(rewards)
                    #     })
                    if env_name == 'Acrobot-v1':
                        grid_acrobot_results.append({
                            "Learning Rate": lr,
                            "Learning rate b": lrb,
                            "Gamma": gamma_val,
                            "Episodes": episodes,
                            "Solved Episode": len(rewards)
                        })
                    elif env_name == 'MountainCarContinuous-v0':
                        grid_mountaincar_results.append({
                            "Learning Rate": lr,
                            "Learning rate b": lrb,
                            "Gamma": gamma_val,
                            "Episodes": episodes,
                            "Solved Episode": len(rewards)
                        })
                    
                    

    grids = {
        "Acrobot-v1": grid_acrobot_results,
        "MountainCarContinuous-v0": grid_mountaincar_results
    }

    for env_name, grid_results in grids.items():
        grid_search_df = pd.DataFrame(grid_results)
        grid_search_df.sort_values(by="Solved Episode", inplace=True, na_position='last')
        save_path = Path("Assignment3/plots_Q1") / f"{env_name}_grid_search_results.csv"
        grid_search_df.to_csv(save_path, index=False)
        print(f"Grid search results for {env_name} saved to {save_path}")
