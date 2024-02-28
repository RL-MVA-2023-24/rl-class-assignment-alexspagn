from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import numpy as np
import torch.nn as nn
import os
import xgboost as xgb
from tqdm import tqdm
import pickle

from statistics import mean

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.


env2 = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

def evaluate_agent(Qfunct, env, nb_episode):
    """
    Evaluate an agent in a given environment.

    Args:
        agent (Agent): The agent to evaluate.
        env (gym.Env): The environment to evaluate the agent in.
        nb_episode (int): The number of episode to evaluate the agent.

    Returns:
        float: The mean reward of the agent over the episodes.
    """
    rewards: list[float] = []
    for _ in range(nb_episode):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        while not done and not truncated:
            Qsa = []
            for a in range(env.action_space.n):
                sa = np.append(obs,a).reshape(1, -1)
                Qsa.append(Qfunct.predict(sa))
            action = np.argmax(Qsa)

            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return mean(rewards)

def collect_samples(env, horizon, disable_tqdm=False, print_done_states=False):
    s, _ = env.reset()
    #dataset = []
    S = []
    A = []
    R = []
    S2 = []
    D = []
    for _ in tqdm(range(horizon), disable=disable_tqdm):
        a = env.action_space.sample()
        s2, r, done, trunc, _ = env.step(a)
        #dataset.append((s,a,r,s2,done,trunc))
        S.append(s)
        A.append(a)
        R.append(r)
        S2.append(s2)
        D.append(done)
        if done or trunc:
            s, _ = env.reset()
            if done and print_done_states:
                print("done!")
        else:
            s = s2
    S = np.array(S)
    A = np.array(A).reshape((-1,1))
    R = np.array(R)
    S2= np.array(S2)
    D = np.array(D)
    return S, A, R, S2, D


def collect_samples_90(env, horizon, Qfunct, disable_tqdm=False, print_done_states=False):
    s, _ = env.reset()
    #dataset = []
    S = []
    A = []
    R = []
    S2 = []
    D = []
    for _ in tqdm(range(horizon), disable=disable_tqdm):
        if np.random.rand() < 0.10:
            a = env.action_space.sample()
        else:
            # Assuming 's' is your current state as a numpy array and 'env.action_space.n' is the number of actions
            actions = np.arange(env.action_space.n)
            # Replicate 's' for each action
            s_replicated = np.tile(s, (env.action_space.n, 1))
            # Create a column vector of actions
            actions_column = actions.reshape(-1, 1)
            # Concatenate 's' with actions to form state-action pairs
            sa_batch = np.hstack((s_replicated, actions_column))
            # Predict Q-values for all state-action pairs
            Qsa = Qfunct.predict(sa_batch)
            # Select the action with the maximum Q-value
            a = np.argmax(Qsa)
        s2, r, done, trunc, _ = env.step(a)
        #dataset.append((s,a,r,s2,done,trunc))
        S.append(s)
        A.append(a)
        R.append(r)
        S2.append(s2)
        D.append(done)
        if done or trunc:
            s, _ = env.reset()
            if done and print_done_states:
                print("done!")
        else:
            s = s2
    S = np.array(S)
    A = np.array(A).reshape((-1,1))
    R = np.array(R)
    S2= np.array(S2)
    D = np.array(D)
    return S, A, R, S2, D


def rf_fqi(S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
    nb_samples = S.shape[0]
    Qfunctions = []
    SA = np.append(S, A.reshape(-1, 1), axis=1)

    for iter in tqdm(range(iterations), disable=disable_tqdm):
        if iter == 0:
            value = R.copy()
        else:
            Q2 = np.zeros((nb_samples, nb_actions))
            for a2 in range(nb_actions):
                A2 = a2 * np.ones((S.shape[0], 1))
                S2A2 = np.append(S2, A2, axis=1)
                Q2[:, a2] = Qfunctions[-1].predict(S2A2)
            max_Q2 = np.max(Q2, axis=1)
            value = R + gamma * (1 - D) * max_Q2
        
        Q = xgb.XGBRegressor(
            booster='gbtree',         # Use tree-based models
            learning_rate = 0.5,        # Set the learning rate
            max_depth = 11,             # Maximum tree depth
            n_estimators = 200,         # Number of trees
            objective = 'reg:squarederror', # Default objective for regression tasks in XGBoost
            tree_method='hist'       # Use 'hist' for faster computation
        )

        Q.fit(SA, value)
        Qfunctions.append(Q)

    return Qfunctions

class EnsembleQFunction:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        # Average the predictions from all models
        predictions = np.array([model.predict(X) for model in self.models])
        return np.min(predictions, axis=0)

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        self.Qfunctions = []
        self.num_samples = 6e3
        self.gamma = .99
        self.nb_iter1 = 400
        self.nb_iter2 = 500
        self.nb_actions = env.action_space.n

        self.final_model = xgb.XGBRegressor()
        

    def train(self):
        # Initial collection of samples
        S, A, R, S2, D = collect_samples(env, int(self.num_samples))
        # Store initial samples separately to ensure they are not removed
        initial_S, initial_A, initial_R, initial_S2, initial_D = S, A, R, S2, D
        
        # List to hold additional samples from each iteration
        additional_samples = []
        values = []

        max_iter = 39
        
        for i in range(max_iter):
            if i<=10:
                self.Qfunctions = rf_fqi(S, A, R, S2, D, self.nb_iter1, self.nb_actions, self.gamma)
            else:
                self.Qfunctions = rf_fqi(S, A, R, S2, D, self.nb_iter2, self.nb_actions, self.gamma)
            
            current_path = os.getcwd()
            fullpath = os.path.join(current_path, f"Q_function{i}.pkl")
            with open(fullpath, 'wb') as file:
                pickle.dump(self.Qfunctions[-1], file)  # Saving all Q-functions

            value = evaluate_agent(self.Qfunctions[-1], env, 1)
            values.append(value)
            print("Specific patient: ", value)
            print("Random patient: ", evaluate_agent(self.Qfunctions[-1], env2, 1))
            
            # Every 3 steps use the random patient
            if i % 3 == 0:
                env_to_use = env2
            else:
                env_to_use = env
            
            # Collect new samples
            S1, A1, R1, S21, D1 = collect_samples_90(env_to_use, int(self.num_samples), self.Qfunctions[-1])
            
            # Add the new samples to the list
            additional_samples.append((S1, A1, R1, S21, D1))
            
            # Concatenate new samples with the existing ones for training
            S = np.concatenate((S, S1), axis=0)
            A = np.concatenate((A, A1), axis=0)
            R = np.concatenate((R, R1), axis=0)
            S2 = np.concatenate((S2, S21), axis=0)
            D = np.concatenate((D, D1), axis=0)
            
            # After 15 steps, drop the oldest samples (but keep the very first ones)
            if i >= 15:
                # Remove the oldest additional samples
                oldest_samples = additional_samples.pop(0)
                # Update S, A, R, S2, D to remove the oldest samples
                # This involves reconstructing each array without the oldest samples
                S, A, R, S2, D = initial_S, initial_A, initial_R, initial_S2, initial_D
                for samples in additional_samples:
                    S = np.concatenate((S, samples[0]), axis=0)
                    A = np.concatenate((A, samples[1]), axis=0)
                    R = np.concatenate((R, samples[2]), axis=0)
                    S2 = np.concatenate((S2, samples[3]), axis=0)
                    D = np.concatenate((D, samples[4]), axis=0)

        # Final training with all collected samples
        self.Qfunctions = rf_fqi(S, A, R, S2, D, self.nb_iter2, self.nb_actions, self.gamma)

        current_path = os.getcwd()
        fullpath = os.path.join(current_path, f"Q_function{max_iter}.pkl")
        with open(fullpath, 'wb') as file:
            pickle.dump(self.Qfunctions[-1], file)

        value = evaluate_agent(self.Qfunctions[-1], env, 1)
        values.append(value)

        # Convert the list to a numpy array
        values_array = np.array(values)

        # Get the indexes that would sort the array, and select the top value
        indexes_of_best = np.argsort(values_array)[-1:][::-1]

        best_index = indexes_of_best.tolist()[0]

        # set final_model as the best one
        current_path = os.getcwd()
        fullpath = os.path.join(current_path, f'Q_function{best_index}.pkl')
        with open(fullpath, 'rb') as file:
            self.final_model = pickle.load(file)


    def act(self, observation, use_random=False):
        if use_random:
            action = np.random.randint(0, self.nb_actions)
        else:
            Qsa = []
            for a in range(env.action_space.n):
                sa = np.append(observation,a).reshape(1, -1)
                Qsa.append(self.final_model.predict(sa))
            action = np.argmax(Qsa)
        return action

    def save(self, path):
        # Save the model to a file
        fullpath = os.path.join(path, 'Q_function_final.pkl')
        with open(fullpath, 'wb') as file:
            pickle.dump(self.final_model, file)

    def load(self):
        # Load the model from a file
        current_path = os.getcwd()
        fullpath = os.path.join(current_path, 'Q_function_final.pkl')
        with open(fullpath, 'rb') as file:
            self.final_model = pickle.load(file)


            
#Train agent
'''
agent = ProjectAgent()
agent.train()

current_path = os.getcwd()
agent.save(current_path)
'''