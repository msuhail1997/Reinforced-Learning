import numpy as np
import scipy.signal
import gym
import pdb
import matplotlib.pyplot as plt
from pendulum import *

class ContinuousPolicy(object):
    def __init__(self, num_states, num_actions):
        self.num_states = num_states 
        self.num_actions = num_actions
        # here are the weights for the policy - you may change this initialization       
        self.weights = np.zeros((self.num_states, self.num_actions))


    # TODO: fill this function in    
    # it should take in an environment state
    def act(self, state):
        actions = np.zeros(self.num_actions)
        for j in range(self.num_actions) :
            mu = np.dot(self.weights[:, j], get_feature_vector(state))
            sigma = 1
            actions[j] = np.random.normal(mu, sigma)

        for i in range(len(actions)) :
            max_action = 2.0
            min_action = -2.0
            if actions[i] < min_action :
                actions[i] = min_action
            elif actions[i] > max_action :
                actions[i] = max_action

        return actions

    # TODO: fill this function in    
    # computes the gradient of the discounted return    
    # at a specific state and action    
    # use the computed advantage function appropriately.
    # return the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, state, action, advantage):
        grad = np.zeros((self.num_states, self.num_actions))
        for i in range(self.num_actions) :
            sigma = 1
            coeff = (1 / (sigma ** 2))
            print(action[i])
            print(get_feature_vector(state))
            first_term = action[i] * get_feature_vector(state)
            second_term = self.weights[:, i] * np.square(get_feature_vector(state))
            dlog = coeff * (first_term - second_term) 
            grad[:, i] = dlog

        return advantage * grad

    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient (from compute_gradient())    
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size):
        self.weights += step_size * grad

#design linear baseline
class LinearValueEstimator(object):
    def __init__(self, num_states):
        self.num_states = num_states
        self.weights = np.zeros(num_states)


    # TODO: fill this function in
    #takes in a state and predicts a value for the state
    def predict(self, state):
        return np.dot(get_feature_vector(state), self.weights)

    # TODO: fill this function in
    # construct a suitable loss function and use it to update the 
    # values of the value estimator. choose suitable step size for updating the value estimator
    def update(self, state, value_estimate, target, value_step_size):
        delta = (target - value_estimate)
        grad = get_feature_vector(state)
        self.weights += (value_step_size * delta * grad) 


# TODO: fill this function in
# takes in a list of rewards from an episode
# and returns a list of discounted rewards
# Ex. get_discounted_returns([1, 1, 1], 0.5)
# should return [1.75, 1.5, 1]
def get_discounted_returns(rewards, gamma) :
    discounted_returns = []
    for i in range(len(rewards)) :
        discounted_return = rewards[i]
        for j in range(i + 1, len(rewards)) :
            discounted_return += (gamma ** (j-i)) * rewards[j]
        discounted_returns.append(discounted_return)
    return discounted_returns


def Gaussian2D(mu, sigma, state) :
    exponent = (((state - mu) ** 2) / (sigma ** 2))
    return np.exp((-0.5) * exponent) / (2 * sigma)

# Get the RBF feature vector for the inputted state
def get_feature_vector(state) :
    features1, features2 = [], []
    thetas = np.linspace(0, 2*np.pi, num = 5, endpoint = True)
    theta_dots = np.linspace(-8, 8, num = 5, endpoint = True)

    for theta in thetas :
        features1.append(Gaussian2D(theta, 0.25, np.arcsin(state[1])))

    for theta_dot in theta_dots :
        features2.append(Gaussian2D(theta_dot, 0.25, state[2]))

    return np.array([np.sum(features1), np.sum(features2)])

# TODO: fill this function in 
# this will take in an environment, GridWorld
# a continuous policy 
# a value estimator,
# a discount rate, gamma
# and the number of episodes you want to run the algorithm for
# make sure to add in the baseline computation here. 
# Using the computed baseline, compute the advantage. 
# Use this advantage in the policy gradient calculation
def reinforce(env, policy, value_estimator, baseline = False):
    ep_rewards = []
    num_episodes = 20000
    learning_rate = 1e-4
    if baseline :
        learning_rate = 1e-3
    value_step_size = 1e-1
    gamma = 0.9
    grouped_reward = 0

    for i in range(num_episodes) :

        value_estimates, value_targets, states, actions = [], [], [], []
        state = env.reset()
        done = False

        #Generate a run
        while not done :
            action = policy.act(state)
            actions.append(action)
            states.append(state)
            value_estimates.append(value_estimator.predict(state))
            state, reward, done, _ = env.step(action)
            value_targets.append(reward)

        grouped_reward += np.sum(value_targets)
        value_targets = get_discounted_returns(value_targets, gamma)

        if (i + 1) % 500 == 0 :
            ep_rewards.append(grouped_reward / 500)
            grouped_reward = 0

        #Update weights
        for action, state, value_estimate, value_target in zip(actions, states, value_estimates, value_targets) :
            if baseline :
                advantage = (value_target - value_estimate)
                grad = policy.compute_gradient(state, action, advantage)
                policy.gradient_step(grad, learning_rate)
                value_estimator.update(state, value_estimate, value_target, value_step_size)
            else :
                advantage = value_target
                grad = policy.compute_gradient(state, action, advantage)
                policy.gradient_step(grad, learning_rate)

    return ep_rewards

if __name__ == "__main__":

    env = Continuous_Pendulum()

    # # TODO: define num_states and num_actions

    num_states, num_actions = 2, 1 # Number of RBF features

    policy = ContinuousPolicy(num_states, num_actions)
    value_estimator = LinearValueEstimator(num_states)
    reinforce(env, policy, value_estimator)

    # Test time
    state = env.reset()
    done = False
    state_hist = []
    while not done:
        input("press enter:")
        action = policy.act(state)
        state, reward, done, _ = env.step(action)
        state_hist.append(state)

    # Plotting test time results
    state_hist = np.array(state_hist)
    plt.plot(state_hist[0, :])
    plt.xlabel("time (s)")
    plt.ylabel("angle (rad)")
    plt.show()
