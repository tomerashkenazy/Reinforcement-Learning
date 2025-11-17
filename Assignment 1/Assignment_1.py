import numpy as np
import torch
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from numpy import random



class frozenlake_agent(gym.Env):
    def __init__(self,env,learn_rate:float,epsilon:float,epsilon_decay,min_epsilon,gamma:float,episodes:int=1000):
        """Initialize the Q-learning agent for FrozenLake environment.
        Args:
            env (gym.Env): The FrozenLake environment.
            learn_rate (float): Learning rate for Q-learning updates.
            epsilon (float): Exploration rate (epsilon) for epsilon-greedy policy.
            gamma (float): Discount factor for future rewards.
            episode (int): Number of episodes for training.
        """
        self.env=env
        self.lr=learn_rate
        self.epsilon=epsilon
        self.epsilon_decay=epsilon_decay
        self.min_epsilon=min_epsilon
        self.gamma=gamma
        self.q_table=np.zeros((self.env.observation_space.n,self.env.action_space.n)) #(16,4)
        self.learning_error = []
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
            action = np.argmax(self.q_table[state, :])  # Exploit: best action from Q-table
        return action

    def update_q_value(self,state,action,reward,next_state,terminal):
        """Update the Q-value for a given state-action pair.
        Args:
            state (int): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (int): Next state after taking the action.
        """
        old_value = self.q_table[state, action]
        next_max_state = np.max(self.q_table[next_state, :])

        target = reward if terminal else reward + self.gamma * next_max_state

        self.q_table[state, action] = (1-self.lr)*old_value + self.lr*(target)



    def train(self):
        """Train the agent using Q-learning algorithm."""
        for episode in range(self.episodes):
            state, _ = self.env.reset()
            done = False
            truncated = False
            total_reward = 0

            while not (done or truncated):
                action = self.choose_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                terminal = done or truncated

                self.update_q_value(state, action, reward, next_state, terminal)

                state = next_state
                total_reward += reward

            self.learning_error.append(total_reward)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)  # Decay epsilon



def evaluate(env,agent, episodes=10):
    total_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total = 0
        step = 0


        
        while not (done or truncated):
            action = np.argmax(agent.q_table[state])
            state, reward, done, truncated, info = env.step(action)
            total += reward
            step += 1
        env.render()
        total_rewards.append(total)
        print(f'Episode {ep + 1}: Reward = {total}')
    env.close()
            

    return total_rewards


if __name__ == "__main__":
    # Easier environment to verify learning:
    train_env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=True,  # deterministic first!
    )

    test_env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=True,  # stochastic environment for testing
        render_mode='human'
    )

    agent = frozenlake_agent(
        env=train_env,
        learn_rate=0.1,
        epsilon=1.0,
        epsilon_decay=0.999,
        min_epsilon=0.1,
        gamma=0.99,
        episodes=5000
    )

    agent.train()

    scores = evaluate(test_env, agent, episodes=50)
    print("Evaluation rewards:", scores)
    print("Average:", np.mean(scores))