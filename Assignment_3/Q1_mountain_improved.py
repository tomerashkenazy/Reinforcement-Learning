import torch
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SOLVE_THRESHOLDS = {"MountainCarContinuous-v0": -110}


# ============ NETWORKS ============

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, h1=128, h2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, action_size)
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class CriticNetwork(nn.Module):
    def __init__(self, state_size, h1=128, h2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============ AGENT ============

class ActorCriticAgent:
    def __init__(self, env, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, episodes=1000):
        self.env = env
        self.gamma = gamma
        self.episodes = episodes

        self.state_size = 6     # padded universal size
        self.action_size = 3

        self.actor = ActorNetwork(self.state_size, self.action_size).to(device)
        self.critic = CriticNetwork(self.state_size).to(device)

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.rewards = []


    def pad_state(self, s):
        s = np.pad(s, (0, self.state_size - len(s)))
        if self.env == "MountainCarContinuous-v0":
            s[1] *= 100    # amplify velocity
        return s


    def select_action(self, state):
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.actor(s)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        return a.item(), dist.log_prob(a), dist.entropy()


    def train(self):
        print("Starting padded MountainCarContinuous-v0 training...")
        for ep in range(self.episodes):
            s,_ = self.env.reset()
            s = self.pad_state(s)
            done = False
            total = 0

            while not done:
                a, logp, entropy = self.select_action(s)
                s2, r, term, trunc, _ = self.env.step([a])
                done = term or trunc
                s2 = self.pad_state(s2)

                # reward shaping
                velocity_bonus = abs(s2[1]) * 10
                height_bonus = s2[0]
                r_shaped = r + velocity_bonus + height_bonus

                s_t = torch.FloatTensor(s).to(device)
                s2_t = torch.FloatTensor(s2).to(device)

                v = self.critic(s_t)
                v2 = self.critic(s2_t).detach() if not done else torch.tensor(0.).to(device)

                delta = r_shaped + self.gamma * v2 - v
                ##advantage = (delta - delta.mean()) / (delta.std() + 1e-8)

                # Actor
                actor_loss = -(logp * delta.detach() + 0.01 * entropy)
                self.opt_actor.zero_grad()
                actor_loss.backward()
                self.opt_actor.step()

                # Critic
                critic_loss = (delta).pow(2)
                self.opt_critic.zero_grad()
                critic_loss.backward()
                self.opt_critic.step()

                s = s2
                total += r

            self.rewards.append(total)

            if (ep+1) % 100 == 0:
                avg = np.mean(self.rewards[-100:])
                print(f"Episode {ep+1} | Avg100: {avg:.2f}")
                if avg >= 85:
                    print("🏁 MountainCarContinuous solved!")
                    break

        return self.rewards
    
    def plot_results(self, save_dir="plots_A3_Q3", model_name="actor_critic"):
        # Correctly construct the save directory path
        save_dir = Path(save_dir) / model_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save Dataframe of training log
        print(f"Training log saved to {save_dir / 'training_log.csv'}")
        
        if len(self.rewards) > 0:
            episodes = np.arange(1, len(self.rewards) + 1)

            plt.figure()
            plt.plot(episodes, self.rewards, linewidth=0.6)  # Adjusted line width
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(f"Reward per Episode [{model_name}]")
            plt.grid(True)

            plt.savefig(save_dir / "episode_rewards.png", dpi=300, bbox_inches='tight')
            plt.close()
            
    def save_model(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), save_dir / "actor_net.pth")
        torch.save(self.critic.state_dict(), save_dir / "critic_net.pth")



if __name__ == "__main__":
    # env = gym.make("MountainCarContinuous-v0")
    # agent = ActorCriticAgent(env)
    # agent.train()
    # agent.plot_results()
    # agent.save_model(save_dir="/home/ohadshee/Desktop/RL/Reinforcement-Learning/plots_A3_Q3")


    
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
                    base_dir = Path("/home/ohadshee/Desktop/RL/Reinforcement-Learning/Assignment_3/plots_improved_continuous")

                    # Environment setup
                    
                    env = gym.make(env_name)

                    
                    agent_actor_critic = ActorCriticAgent(
                        env=env,
                        lr_actor=lr,
                        lr_critic=lrb,  # Using the same learning rate for simplicity
                        gamma=gamma_val,
                        episodes=episodes,
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

                    grid_mountaincar_results.append({
                        "Learning Rate": lr,
                        "Learning rate b": lrb,
                        "Gamma": gamma_val,
                        "Episodes": episodes,
                        "Solved Episode": len(rewards)
                    })
                    
                    

    grids = {
        "MountainCarContinuous-v0": grid_mountaincar_results
    }

    for env_name, grid_results in grids.items():
        grid_search_df = pd.DataFrame(grid_results)
        grid_search_df.sort_values(by="Solved Episode", inplace=True, na_position='last')
        save_path = Path("/home/ohadshee/Desktop/RL/Reinforcement-Learning/Assignment_3/plots_improved_continuous") / f"{env_name}_grid_search_results.csv"
        grid_search_df.to_csv(save_path, index=False)
        print(f"Grid search results for {env_name} saved to {save_path}")