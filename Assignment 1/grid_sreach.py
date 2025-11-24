import torch
import numpy as np 
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from numpy import random
from Q1 import frozenlake_agent, evaluate
from itertools import product
from functools import partial
import pandas as pd

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
        self.results_table = pd.DataFrame(columns=['episodes','learn_rate','epsilon', 'epsilon_decay', 'min_epsilon', 'gamma', "mean score"])
        

    def _generate_param_combinations(self):
        keys = self.param_grid.keys()
        values = (self.param_grid[key] for key in keys)
        for combination in product(*values):
            yield dict(zip(keys, combination))

    def check_params(self, params):
        print("==================================================\n")
        print(f"Testing parameters: {params}")
        model = self.model(self.env,params=params)
        model.train()
        score = self.evaluate_function(agent = model)
        print(f"Score with parameters {params}: {np.mean(score)}")
        if np.mean(score) > self.best_score:
            print(f"\nNew best score: {np.mean(score)} with parameters: {params}")
            self.best_score = np.mean(score)
            self.best_params = params

            # ---- Add to results table ----
        new_row = params.copy()
        new_row["mean score"] = np.mean(score)

        self.results_table = pd.concat([self.results_table, pd.DataFrame([new_row])],
                                    ignore_index=True)

        self.results_table = self.results_table.sort_values(
            by="mean score", ascending=False
        ).reset_index(drop=True)

        print("\nCurrent parameter table:")
        print(self.results_table)
        self.results_table.to_csv("results_table.csv", index=False)

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
        'learn_rate': [0.1, 0.2],
        'epsilon': [0.5,0.7,1],
        'epsilon_decay': [0.99, 0.95],
        'min_epsilon': [0.05, 0.1],
        'gamma': [0.9, 0.99],
        'episodes': [5000]
    }

    #Q1
    train_env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=True,  # deterministic first!
    )
    model = frozenlake_agent
    eval_function = partial(evaluate, train_env)
    best_params, best_score = Q1_grid_search(train_env, params,evaluate_function=eval_function)

    #===================================================

    #Q2
