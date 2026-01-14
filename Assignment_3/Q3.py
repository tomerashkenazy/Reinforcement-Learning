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
    "MountainCarContinuous-v0": 90,
}

learn_rate = [0.01, 0.001]
learn_rate_b = [0.01, 0.001]
gamma_grid = [0.99, 0.95]

# Universal input size (max state dim among all envs)
MAX_STATE_DIM = 6 

# --- ORIGINAL MODEL DEFINITIONS (Must match Q1/Q2 exactly) ---
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128, hidden_size2=32):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
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

# --- PROGRESSIVE NETWORK WRAPPERS ---
class ProgressiveBase(nn.Module):
    """
    Base class that handles the connection of Source Hidden Layers 
    to the Target Output Layer.
    """
    def __init__(self, target_net, source_nets, action_size):
        super(ProgressiveBase, self).__init__()
        self.target_net = target_net
        self.source_nets = nn.ModuleList(source_nets)
        
        # 1. Freeze Source Networks
        for net in self.source_nets:
            for param in net.parameters():
                param.requires_grad = False
            net.eval() # Set to eval mode

        # 2. Calculate Combined Hidden Size
        # Target Hidden + Source 1 Hidden + Source 2 Hidden ...
        target_hidden = target_net.fc2.in_features
        source_hidden_sum = sum(net.fc2.in_features for net in source_nets)
        total_hidden_size = target_hidden + source_hidden_sum
        
        # 3. Re-initialize Target Output Layer
        # This replaces the old fc2 with a new larger one
        self.target_net.fc2 = nn.Linear(total_hidden_size, action_size)
        
    def get_combined_features(self, x):
        # Get features from the new Target network
        h_target = torch.relu(self.target_net.fc1(x))
        
        # Get features from frozen Source networks
        source_features = []
        for net in self.source_nets:
            # We assume 'x' is padded to MAX_STATE_DIM, which sources expect
            h_s = torch.relu(net.fc1(x))
            source_features.append(h_s)
        
        # Concatenate all feature vectors
        combined = torch.cat([h_target] + source_features, dim=-1)
        return combined

class ProgressiveActor(ProgressiveBase):
    def forward(self, x):
        combined = self.get_combined_features(x)
        out = self.target_net.fc2(combined)
        return torch.softmax(out, dim=-1)

class ProgressiveCritic(ProgressiveBase):
    def forward(self, x):
        combined = self.get_combined_features(x)
        out = self.target_net.fc2(combined)
        return out

# --- HELPER: Load Model ---
def load_and_modify_model(model_class, path, input_size, old_action_size, new_action_size):

    if model_class is ActorNetwork:
        model = ActorNetwork(input_size, old_action_size).to(device)
    else:
        model = CriticNetwork(input_size).to(device)

    checkpoint = torch.load(path, map_location=device)

    # 🚨 REMOVE old output layer from checkpoint BEFORE loading
    checkpoint.pop("fc2.weight", None)
    checkpoint.pop("fc2.bias", None)

    model.load_state_dict(checkpoint, strict=False)

    hidden_size = model.fc1.out_features

    # Attach new output head
    if isinstance(model, ActorNetwork):
        model.fc2 = nn.Linear(hidden_size, new_action_size).to(device)
    else:
        model.fc2 = nn.Linear(hidden_size, 1).to(device)

    return model


# --- PROGRESSIVE AGENT ---
class ProgressiveAgent:
    def __init__(self, env, source_actors, source_critics, lr=1e-3, gamma=0.99, episodes=1000, env_name='CartPole-v1'):
        self.env = env
        self.env_name = env_name
        self.gamma = gamma
        self.episodes = episodes
        
        # Dimensions
        self.state_size = MAX_STATE_DIM
        try:
            self.action_size = env.action_space.n
        except:
            self.action_size = 3

        # 1. Create FRESH Target Networks
        target_actor = ActorNetwork(self.state_size, self.action_size).to(device)
        target_critic = CriticNetwork(self.state_size).to(device)

        # 2. Wrap them in Progressive Architecture
        self.actor_network = ProgressiveActor(target_actor, source_actors, self.action_size).to(device)
        self.critic_network = ProgressiveCritic(target_critic, source_critics, 1).to(device) # Critic output is always 1

        # 3. Optimizers (Only optimize parameters that require grad)
        self.optim_policy = optim.Adam(filter(lambda p: p.requires_grad, self.actor_network.parameters()), lr=lr)
        self.optim_value = optim.Adam(filter(lambda p: p.requires_grad, self.critic_network.parameters()), lr=lr * 10)

        self.episode_rewards = []

    def preprocess_state(self, state):
        """Pads state to match MAX_STATE_DIM (6)"""
        current_dim = state.shape[0]
        if current_dim < MAX_STATE_DIM:
            return np.pad(state, (0, MAX_STATE_DIM - current_dim), mode='constant')
        return state

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.actor_network(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def train(self):
        start_time = time.time()
        print(f"Starting Progressive Training on {self.env_name}...")
        
        solved_episode = -1
        self.episode_times = []   # track time for each episode

        for episode in range(self.episodes):
            episode_start_time = time.time()
            
            state, _ = self.env.reset()
            state = self.preprocess_state(state) 
            
            if self.env_name == 'MountainCarContinuous-v0':
                state[1] *= 100 # Scaling hack

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
                    # Shaping
                    shaped_reward = reward + abs(next_state[1])*100 + next_state[0]
                else:
                    shaped_reward = reward

                s = torch.FloatTensor(state).to(device)
                s_next = torch.FloatTensor(next_state).to(device)

                # Critic Update
                value_s = self.critic_network(s)
                value_s_next = torch.tensor(0.0).to(device) if done else self.critic_network(s_next)

                delta = shaped_reward + self.gamma * value_s_next.detach() - value_s.detach() # TD Error

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

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                total_time_100 = np.sum(self.episode_times[-100:])
                print(f"Episode {episode+1} | Avg Reward: {avg_reward:.2f} | Time for 100 Episodes: {total_time_100:.2f}s")
                
                if avg_reward >= SOLVE_THRESHOLDS[self.env_name]:
                    print(f"--- {self.env_name} SOLVED in {episode+1} episodes! ---")
                    solved_episode = episode + 1
                    break
        
        end_time = time.time()
        return {
            "env": self.env_name,
            "episodes_to_solve": solved_episode if solved_episode != -1 else self.episodes,
            "duration": end_time - start_time,
            "final_reward": np.mean(self.episode_rewards[-100:])
        }, self.episode_rewards

def run_progressive_experiment():

    base_save_path = Path("Assignment3/plots_Q3")
    base_save_path.mkdir(parents=True, exist_ok=True)
    grid_results = []

    experiments = [
    ("{Acrobot,MC}->CartPole", "CartPole-v1",
     ["Assignment_3/transfer_models/actor_critic_Acrobot-v1/actor_net.pth",
      "Assignment_3/transfer_models/actor_critic_MountainCarContinuous-v0/actor_net.pth"],
     ["Assignment_3/transfer_models/actor_critic_Acrobot-v1/critic_net.pth",
      "Assignment_3/transfer_models/actor_critic_MountainCarContinuous-v0/critic_net.pth"],
     [3,3]),

    ("{CartPole,Acrobot}->MC", "MountainCarContinuous-v0",
     ["Assignment_3/transfer_models/actor_critic_CartPole-v1/actor_net.pth",
      "Assignment_3/transfer_models/actor_critic_Acrobot-v1/actor_net.pth"],
     ["Assignment_3/transfer_models/actor_critic_CartPole-v1/critic_net.pth",
      "Assignment_3/transfer_models/actor_critic_Acrobot-v1/critic_net.pth"],
     [2,3])
]


    for lr in learn_rate:
        for gamma in gamma_grid:
            for lrb in learn_rate_b:

                for name, tgt, actor_paths, critic_paths, src_actions in experiments:

                    print(f"\n=== {name} | lr={lr} lrb={lrb} gamma={gamma} ===")
                    env = gym.make(tgt)

                    src_actors, src_critics = [], []

                    for ap, cp, sa in zip(actor_paths, critic_paths, src_actions):
                        action_size = env.action_space.n if tgt != 'MountainCarContinuous-v0' else env.action_space.shape[0]
                        src_actors.append(load_and_modify_model(ActorNetwork, ap, MAX_STATE_DIM, sa, action_size))
                        src_critics.append(load_and_modify_model(CriticNetwork, cp, MAX_STATE_DIM, sa, action_size))

                    agent = ProgressiveAgent(env, src_actors, src_critics, lr=lr, gamma=gamma, env_name=tgt)
                    stats, rewards = agent.train()

                    tag = f"{name}_lr{lr}_lrb{lrb}_g{gamma}"
                    plt.figure()
                    plt.plot(rewards)
                    plt.savefig(base_save_path / f"{tag}.png")
                    plt.close()

                    grid_results.append({
                        "Experiment": name,
                        "LR": lr,
                        "Critic LR": lrb,
                        "Gamma": gamma,
                        "Solved Episodes": stats["episodes_to_solve"],
                        "Final Reward": stats["final_reward"]
                    })

    pd.DataFrame(grid_results).to_csv(base_save_path / "Q3_grid_search_results.csv", index=False)


if __name__ == "__main__":
    run_progressive_experiment()