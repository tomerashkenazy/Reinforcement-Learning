import torch
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import time

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- CONFIGURATION ---
SOLVE_THRESHOLDS = {
    "CartPole-v1": 475,
    "Acrobot-v1": -100,
    'MountainCarContinuous-v0': 90,
}

# Mapping of max state dimensions to ensure compatibility
# We assume the "Universal" model input size is 6 (size of Acrobot)
MAX_STATE_DIM = 6 

learn_rate_b =[0.01] #[0.01, 0.001]      # Critic LR multiplier
learn_rate = [0.0001] #[0.01, 0.001, 0.0001]    # Actor LR
gamma_grid  = [1]#[1,0.99, 0.95]
episodes = 1000
early_stop = True     # Already built into your training


# --- MODEL DEFINITIONS (Must match Q1 exactly for loading) ---
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128, hidden_size2=32):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        # Note: Even if unused in forward, this must exist to load state_dict if it was in Q1
        self.fc3 = nn.Linear(hidden_size2, action_size) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=128, hidden_size2=32):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.fc3 = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_and_modify_model(model_class, path, input_size, old_action_size, new_action_size):

    # Build old model (to match checkpoint)
    if model_class is ActorNetwork:
        model = ActorNetwork(input_size, old_action_size, hidden_size=128).to(device)
    else:
        model = CriticNetwork(input_size, hidden_size=128).to(device)

    checkpoint = torch.load(path, map_location=device)

    # LOAD WITH strict=False  ← THIS IS THE KEY
    model.load_state_dict(checkpoint, strict=False)

    hidden_size = model.fc2.in_features

    # Replace output layer AFTER loading
    if isinstance(model, ActorNetwork):
        model.fc2 = nn.Linear(hidden_size, new_action_size).to(device)
    else:
        model.fc2 = nn.Linear(hidden_size, 1).to(device)

    return model

# --- AGENT CLASS (Adapted for Fine-Tuning) ---
class FineTunedAgent:
    def __init__(self, env, actor_path, critic_path, source_action_size, 
                 lr=1e-3,lrb=1e-2,
 gamma=0.99, episodes=1000, env_name='CartPole-v1'):
        self.env = env
        self.env_name = env_name
        self.gamma = gamma
        self.episodes = episodes
        
        # Determine dimensions
        self.state_size = MAX_STATE_DIM # Fixed to 6 for transfer compatibility
        # Current environment action size
        try:
            self.action_size = env.action_space.n
        except:
             self.action_size = 3 # Fallback if direct access fails

        # Load and Modify Actor
        self.actor_network = load_and_modify_model(
            ActorNetwork, actor_path, self.state_size, source_action_size, self.action_size
        )
        
        # Load and Modify Critic
        self.critic_network = load_and_modify_model(
            CriticNetwork, critic_path, self.state_size, source_action_size, self.action_size
        )

        # Optimizers (Standard finetuning often uses slightly lower LR, but we stick to Q1 values or passed args)
        self.optim_policy = optim.Adam(self.actor_network.parameters(), lr=lr)
        self.optim_value = optim.Adam(self.critic_network.parameters(), lr=lrb) # Critic usually higher LR

        self.episode_rewards = []

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.actor_network(state_tensor)
        
        # Masking for CartPole if needed (optional if network learns fast)
        # But generally fine-tuning relies on the network learning the new valid moves.
        if self.env_name == 'CartPole-v1' and self.action_size == 2:
            # If the network accidentally outputs 3 probs (shouldn't happen due to surgery), handle it.
            pass 

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def preprocess_state(self, state):
        """Pads state to match the MAX_STATE_DIM (6) expected by the network."""
        current_dim = state.shape[0]
        if current_dim < MAX_STATE_DIM:
            padding = MAX_STATE_DIM - current_dim
            return np.pad(state, (0, padding), mode='constant')
        return state
    
    def save(self, save_dir, prefix):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.actor_network.state_dict(), save_dir / f"{prefix}_actor.pth")
        torch.save(self.critic_network.state_dict(), save_dir / f"{prefix}_critic.pth")

        # Save training statistics
        stats_path = save_dir / f"{prefix}_stats.npz"
        np.savez(stats_path,
                 rewards=np.array(self.episode_rewards),
                 env=self.env_name)

        print(f"[Saved] {prefix} models + stats to {save_dir}")

    def train(self):
        start_time = time.time()
        print(f"Starting Fine-Tuning on {self.env_name}...")
        
        solved_episode = -1
        self.episode_times = []   # track time for each episode

        for episode in range(self.episodes):
            episode_start_time = time.time()
            
            state, _ = self.env.reset()
            state = self.preprocess_state(state) # PAD STATE
            
            #MountainCar specific initialization hack from Q1
            if self.env_name == 'MountainCarContinuous-v0':
                state[1] *= 100

            done = False
            total_reward = 0
            I = 1.0

            while not done:
                action, log_prob = self.select_action(state)
                action = [action] if self.env_name == 'MountainCarContinuous-v0' else action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                # Preprocess Next State
                next_state = self.preprocess_state(next_state)
                if self.env_name == 'MountainCarContinuous-v0':
                    next_state[1] *= 100
                    # Shaped Reward for MountainCar
                    velocity_bonus = abs(next_state[1]) * 100 
                    height_bonus = (next_state[0])
                    shaped_reward = reward + velocity_bonus + height_bonus
                else:
                    shaped_reward = reward

                # Tensors
                s = torch.FloatTensor(state).to(device)
                s_next = torch.FloatTensor(next_state).to(device)

                # Critic Update Logic
                value_s = self.critic_network(s)
                value_s_next = torch.tensor(0.0).to(device) if done else self.critic_network(s_next)

                # TD Error
                delta = shaped_reward + self.gamma * value_s_next.detach() - value_s.detach()

                # Actor Loss
                actor_loss = -(I * delta * log_prob)
                self.optim_policy.zero_grad()
                actor_loss.backward()
                self.optim_policy.step()

                # Critic Loss
                delta_value = reward + self.gamma * value_s_next.detach() - value_s
                value_loss = (I * delta_value).pow(2)
                self.optim_value.zero_grad()
                value_loss.backward()
                self.optim_value.step()

                I *= self.gamma
                state = next_state

            # Record episode time and reward
            episode_time = time.time() - episode_start_time
            self.episode_times.append(episode_time)
            self.episode_rewards.append(total_reward)

            # Logging & Solved Check
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                total_time_100 = np.sum(self.episode_times[-100:])
                print(f"Episode {episode+1} | Avg Reward: {avg_reward:.2f} | Time for 100 Episodes: {total_time_100:.2f}s")
                
                if avg_reward >= SOLVE_THRESHOLDS[self.env_name]:
                    print(f"--- {self.env_name} SOLVED in {episode+1} episodes! ---")
                    solved_episode = episode + 1
                    break
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "env": self.env_name,
            "episodes_to_solve": solved_episode if solved_episode != -1 else self.episodes,
            "solved": solved_episode != -1,
            "duration_seconds": duration,
            "final_avg_reward": np.mean(self.episode_rewards[-100:]),
            "total_episodes": len(self.episode_rewards) 
        }, self.episode_rewards

def run_fine_tuning_experiment():

    grid_results = []

    base_save_path = Path("Assignment_3/plots_Q2")
    base_save_path.mkdir(parents=True, exist_ok=True)

    experiments = [
        # ("Acrobot-v1", "CartPole-v1",
        #  "Assignment_3/transfer_models/actor_critic_Acrobot-v1/actor_net.pth",
        #  "Assignment_3/transfer_models/actor_critic_Acrobot-v1/critic_net.pth",
        #  3),

        ("CartPole-v1", "MountainCarContinuous-v0",
         "Assignment_3/transfer_models/actor_critic_CartPole-v1/actor_net.pth",
         "Assignment_3/transfer_models/actor_critic_CartPole-v1/critic_net.pth",
         3)
    ]

    for lr in learn_rate:
        for gamma_val in gamma_grid:
            for lrb in learn_rate_b:
                for src, tgt, actor_path, critic_path, src_action_size in experiments:

                    exp_name = f"{src}_to_{tgt}_lr_{lr}_lrb_{lrb}_gamma_{gamma_val}"
                    print(f"\n=== {exp_name} ===")

                    env = gym.make(tgt)

                    agent = FineTunedAgent(
                        env=env,
                        actor_path=actor_path,
                        critic_path=critic_path,
                        source_action_size=src_action_size,
                        lr=lr,
                        gamma=gamma_val,
                        episodes=episodes,
                        env_name=tgt
                    )

                    stats, rewards = agent.train()

                    agent.save("Assignment_3/finetuned_models", exp_name)

                    # Plot
                    plt.figure()
                    plt.plot(rewards)
                    plt.title(exp_name)
                    plt.savefig(base_save_path / f"{exp_name}.png")
                    plt.close()

                    grid_results.append({
                        "Source": src,
                        "Target": tgt,
                        "Actor LR": lr,
                        "Critic LR Mult": lrb,
                        "Gamma": gamma_val,
                        "Episodes to Solve": stats["episodes_to_solve"],
                        "Solved": stats["solved"]
                    })

    df = pd.DataFrame(grid_results)
    df.sort_values(by="Episodes to Solve", inplace=True)
    df.to_csv(base_save_path / "Q2_grid_search_results.csv", index=False)
    print("Grid search saved.")

if __name__ == "__main__":
    run_fine_tuning_experiment()