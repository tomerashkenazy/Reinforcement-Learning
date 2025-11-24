import torch
import numpy as np 
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from numpy import random
from Q1 import frozenlake_agent, evaluate
from functools import partial


class GridSearch:
    def __init__(self,env,param_grid,evaluate_function = None,train_function = None):
        """
        param_grid: dict, keys are parameter names and values are lists of parameter settings to try
        train_function: function, a function that takes parameters as input and returns a performance metric
        """
        self.param_grid = param_grid
        self.model = frozenlake_agent
        self.train_function = train_function if train_function is not None else self.model.train
        self.best_params = None
        self.best_score = -np.inf
        self.evaluate_function = evaluate_function
        self.env = env
        

    def _generate_param_combinations(self):
        from itertools import product
        keys = self.param_grid.keys()
        values = (self.param_grid[key] for key in keys)
        for combination in product(*values):
            yield dict(zip(keys, combination))

    def check_params(self, params):
        print(f"Testing parameters: {params}")
        model = self.model(self.env,params=params)
        model.train()
        score = self.evaluate_function(agent = model)
        print(f"Score with parameters {params}: {score}")
        print("==================================================\n")
        if np.mean(score) > self.best_score:
            self.best_score = np.mean(score)
            self.best_params = params

    def train(self):
        for params in self._generate_param_combinations():
            self.check_params(params)

    def get_best_params(self):
        return self.best_params, self.best_score
    


def Q1_grid_search(param_grid, train_function, evaluate_function):
    grid_search = GridSearch(param_grid, train_function, evaluate_function)
    grid_search.train()
    return grid_search.get_best_params()


if __name__ == "__main__":
    params = {
        'learn_rate': [0.1, 0.5, 0.9],
        'epsilon': [0.5,0.7,1],
        'epsilon_decay': [0.99, 0.95,0.8],
        'min_epsilon': [0.01, 0.1],
        'gamma': [0.8, 0.9, 0.99],
        'episodes': [5000]
    }

    train_env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=True,  # deterministic first!
    )
    model = frozenlake_agent
    eval_function = partial(evaluate, train_env)
    best_params, best_score = Q1_grid_search(train_env, params,evaluate_function=eval_function)
