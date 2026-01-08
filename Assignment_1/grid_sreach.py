import torch
import numpy as np 
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from numpy import random
from Q1 import frozenlake_agent, evaluate
from itertools import product
from functools import partial
from Q2 import cartpole_agent, q_model3, q_model5,evaluate_cartpole_agent
import pandas as pd
from Q3 import cartpole_agent,DuelingQNetwork,evaluate_cartpole_agent_Q3
class GridSearch_Q1:
    def _init_(self,env,param_grid = None,evaluate_function = None,train_function = None):
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
        model = self.model(self.env,params=params) if params else self.model.copy()
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
    



class GridSearch_Q2(GridSearch_Q1):
    def _init_(self,env,qmodel,param_grid = None,evaluate_function = None,train_function = None):
        super()._init_(env,param_grid,evaluate_function,train_function)
        self.agent = cartpole_agent
        self.results_table = pd.DataFrame(columns=['episodes','learn_rate','epsilon', 'epsilon_decay', 
        'min_epsilon', 'gamma', 'memory_capacity','batch_size','target_update_freq','fc1_size','fc2_size','fc3_size','fc4_size',"mean score"])
        self.qmodel = qmodel

    def check_params(self, params):
        print("==================================================\n")
        print(f"Testing parameters: {params}")

        if self.qmodel is q_model3:
            model = self.qmodel(
                self.env.observation_space.shape[0], 
                self.env.action_space.n,
                params['fc1_size'],
                params['fc2_size'],
            )
        else:
            model = self.qmodel(
                self.env.observation_space.shape[0], 
                self.env.action_space.n,
                params['fc1_size'],
                params['fc2_size'],
                params['fc3_size'],
                params['fc4_size']
            )
        agent = self.agent(self.env, model=model, params=params) 
        agent.train()
        score = self.evaluate_function(agent = agent)
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
        self.results_table.to_csv("q5model_results_table2.0.csv", index=False)

class GridSearch_Q3(GridSearch_Q1):
    def _init_(self,env,dual_model,param_grid = None,evaluate_function = None,train_function = None):
        super()._init_(env,param_grid,evaluate_function,train_function)
        self.agent = cartpole_agent
        self.results_table = pd.DataFrame(columns=['episodes','learn_rate','epsilon', 'epsilon_decay', 
        'min_epsilon', 'gamma', 'memory_capacity','batch_size','target_update_freq','fc1_size','fc2_size','f3c_size','fc_adv_size','fc_val_size',"mean score"])
        self.dual_model = dual_model

    def check_params(self, params):
        print("==================================================\n")
        print(f"Testing parameters: {params}")

        model = self.dual_model(
            self.env.observation_space.shape[0], 
            self.env.action_space.n,
            params['fc1_size'],
            params['fc2_size'],
            params['fc3_size'],
            params['fc_adv_size'],
            params['fc_val_size'],
        )
        agent = self.agent(self.env, model=model, params=params) 
        agent.train()
        score = self.evaluate_function(agent = agent)
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
        self.results_table.to_csv("dual_model_results_table.csv", index=False)


def Q1_grid_search(param_grid, train_function, evaluate_function):
    grid_search = GridSearch_Q1(param_grid, train_function, evaluate_function)
    grid_search.train()
    return grid_search.get_best_params()

def Q2_grid_search(env,param_grid, qmodel, train_function, evaluate_function):
    grid_search = GridSearch_Q2(env,qmodel,param_grid,evaluate_function,train_function)
    grid_search.train()
    return grid_search.get_best_params()

def Q3_grid_search(env,param_grid, dual_model, train_function, evaluate_function):
    grid_search = GridSearch_Q3(env,dual_model,param_grid,evaluate_function,train_function)
    grid_search.train()
    return grid_search.get_best_params()


if __name__ == "__main__":

    def Q1():
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
    def Q2():
        print("Starting Q2 Grid Search...\n")
        params_cartpole_q3 = {
            'learn_rate': [0.001, 0.005],
            'epsilon': [0.95,1],
            'epsilon_decay': [0.99, 0.95],
            'min_epsilon': [0.1],
            'gamma': [0.9, 0.99],
            'episodes': [1000],
            'memory_capacity': [10000],
            'batch_size': [64,128],
            'target_update_freq': [10,15],
            'fc1_size': [16,32],
            'fc2_size': [16,32],
        }
        params_cartpole_q5 = {
            'learn_rate': [0.001],
            'epsilon': [1],
            'epsilon_decay': [0.99],
            'min_epsilon': [0.1],
            'gamma': [0.99],
            'episodes': [1000],
            'memory_capacity': [10000],
            'batch_size': [64,128],
            'target_update_freq': [10],
            'fc1_size': [8,16,32],
            'fc2_size': [8,16,32],
            'fc3_size': [8,16,32],
            'fc4_size': [8,16,32],
        }

        train_env = gym.make("CartPole-v1")
        eval_function = partial(evaluate_cartpole_agent, env=train_env, eval_episodes=200)
        best_params_cartpole, best_score_cartpole = Q2_grid_search(train_env, params_cartpole_q5, qmodel=q_model5, train_function=None, evaluate_function=eval_function)

        return 

    def Q3():
        print("Starting Q3 Grid Search...\n")
        params_cartpole_dueling = {
            'learn_rate': [0.001],
            'epsilon': [1],
            'epsilon_decay': [0.99],
            'min_epsilon': [0.1],
            'gamma': [0.99],
            'episodes': [1000],
            'memory_capacity': [10000],
            'batch_size': [64],
            'target_update_freq': [10],
            'fc1_size': [8,16,32],
            'fc2_size': [8,16,32],
            'fc3_size': [8,16,32],
            'fc_adv_size': [8,16,32],
            'fc_val_size': [8,16,32],
        }

        train_env = gym.make("CartPole-v1")
        eval_function = partial(evaluate_cartpole_agent_Q3, env=train_env, eval_episodes=200)
        best_params_cartpole_dueling, best_score_cartpole_dueling = Q3_grid_search(train_env, params_cartpole_dueling, dual_model=DuelingQNetwork, train_function=None, evaluate_function=eval_function)

        return

 #Call Grid Search functions
#Q1()
#Q2()
Q3()
