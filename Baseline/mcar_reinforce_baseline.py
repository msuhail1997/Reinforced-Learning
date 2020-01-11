from grid_world import *
import numpy as np
import scipy.signal
import gym
import pdb
import matplotlib.pyplot as plt
import random
from mountain_car import *
N_POS = 2
N_VEL = 2

class DiscreteSoftmaxPolicy(object):
    def __init__(self, num_states, num_actions):
        self.num_states = num_states 
        self.num_actions = num_actions
        self.temperature = 1
        # here are the weights for the policy - you may change this initialization       
        self.weights = np.zeros((self.num_states, self.num_actions))


    # TODO: fill this function in    
    # it should take in an environment state
    def act(self, state):
        probabilities = self._softmax(state)
        num = random.random()
        acc, action_taken = 0, self.num_actions - 1
        for i in range(len(probabilities)) :
            acc += probabilities[i]
            if (num <= acc) :
                action_taken = i
                break
        return [action_taken]

    # TODO: fill this function in    
    # computes the gradient of the discounted return    
    # at a specific state and action    
    # use the computed advantage function appropriately.
    # return the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, state, action, advantage):
        probabilities = self._softmax(state)
        first_term = np.zeros((self.num_states, self.num_actions))
        first_term[state, action] = 1 / self.temperature
        second_term = np.zeros((self.num_states, self.num_actions))
        second_term[state, :] = probabilities / self.temperature
        dlog = first_term - second_term

        return advantage * dlog

    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient (from compute_gradient())    
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size):
        self.weights += (step_size * grad)

    # Takes the state and computes the list of probabilities to take 
    # each action accoridng to the softmax function.
    def _softmax(self, state) :
        weights = self.weights[state, :]
        x = weights / self.temperature
        e_s = np.exp(x)
        probabilities = e_s / np.sum(e_s)
        return probabilities


class ValueEstimator(object):
    def __init__(self, num_states):
        self.num_states = num_states
        #initial value estimates or weights of the value estimator are set to zero. 
        self.values = np.zeros((self.num_states))

    # TODO: fill this function in
    #takes in a state and predicts a value for the state
    def predict(self,state):
        return self.values[state]

    # TODO: fill this function in
    # construct a suitable loss function and use it to update the 
    # values of the value estimator. choose suitable step size for updating the value estimator
    def update(self, state, value_estimate, target, value_step_size):
        delta = (target - value_estimate)
        self.values[state] += (value_step_size * delta)


# TODO: fill this function in
# takes in a list of rewards from an episode
# and returns a list of discounted rewards
# Ex. get_discounted_returns([1, 1, 1], 0.5)
# should return [1.75, 1.5, 1]
def get_discounted_returns(rewards, gamma):
    discounted_returns = []
    for i in range(len(rewards)) :
        discounted_return = rewards[i]
        for j in range(i + 1, len(rewards)) :
            discounted_return += (gamma ** (j-i)) * rewards[j]
        discounted_returns.append(discounted_return)
    return discounted_returns

# TODO: fill this function in 
# this will take in an environment, GridWorld
# a policy (DiscreteSoftmaxPolicy)
# a value estimator,
# a discount rate, gamma
# and the number of episodes you want to run the algorithm for
# make sure to add in the baseline computation here. 
# Using the computed baseline, compute the advantage. 
# Use this advantage in the policy gradient calculation
def reinforce(env, policy, value_estimator, baseline = True):
    
    ep_rewards = []
    grouped_reward = 0

    num_episodes = 5000
    learning_rate = 1e-3
    value_step_size = 1e-3
    gamma = 0.9
    batch_size = 3
    discrete_actions = np.linspace(-1.0, 1.0, policy.num_actions)
    saved_trajectories = []

    for i in range(num_episodes) :

        value_estimates, value_targets, states, actions, rewards = [], [], [], [], []
        state = discretize_state(env.reset())
        done = False

        #Generate a run
        while not done :
            action = policy.act(state)
            actions.append(action)
            states.append(state)
            value_estimates.append(value_estimator.predict(state))
            state, reward, done, _ = env.step(discrete_actions[action])
            state = discretize_state(state)
            rewards.append(reward)

        grouped_reward += np.sum(rewards)
        value_targets = get_discounted_returns(rewards, gamma)

        if (i + 1) % 500 == 0 :
            print("::::::" + str(i) + "::::::")
            ep_rewards.append(grouped_reward / 500)
            grouped_reward = 0

        #Update weights
        goal_reached = (rewards[len(rewards) - 1] == 100)

        if goal_reached :
            print(i)
            trajectory = zip(actions, states, value_estimates, value_targets)
            saved_trajectories.append(trajectory)

        if len(saved_trajectories) >= batch_size :
            for traj in saved_trajectories :
                for action, state, value_estimate, value_target in traj :
                    if baseline :
                        advantage = (value_target - value_estimate)
                        grad = policy.compute_gradient(state, action, advantage)
                        policy.gradient_step(grad, learning_rate)
                        value_estimator.update(state, value_estimate, value_target, value_step_size)
                    else :
                        advantage = value_target
                        grad = policy.compute_gradient(state, action, advantage)
                        policy.gradient_step(grad, learning_rate)
            saved_trajectories = []

    return ep_rewards

# Takes a continuous state (from env.step() or env.reset())
# and discretizes it into a number in the set {0, 1, ..., N-1}
# where N is the number of discrete states you have
def discretize_state(state):
    position = state[0]
    velocity = state[1]

    position_bins = np.linspace(-1.2, 0.6, num = N_POS, endpoint = False)
    velocity_bins = np.linspace(-0.07, 0.07, num = N_VEL, endpoint = False)

    discrete_position = np.digitize([position], position_bins)[0] - 1
    discrete_velocity = np.digitize([velocity], velocity_bins)[0] - 1

    if (N_POS * discrete_velocity) + discrete_position > 179 or (N_POS * discrete_velocity) + discrete_position < 0:
        print('ISSUE ALERT')
        print(state)
        print(discrete_position, discrete_velocity)

    return (N_POS * discrete_velocity) + discrete_position
    

if __name__ == "__main__":
    env = Continuous_MountainCarEnv()
    num_actions = 3
    discrete_actions = np.linspace(-1.0, 1.0, num_actions)

    num_states = N_POS * N_VEL 

    policy = DiscreteSoftmaxPolicy(num_states, num_actions)
    value_estimator = ValueEstimator(num_states)

    no_baseline_rewards = []
    for i in range(5) :
        env = Continuous_MountainCarEnv()
        policy = DiscreteSoftmaxPolicy(num_states, num_actions)
        value_estimator = ValueEstimator(num_states)
        ep_rewards = reinforce(env, policy, value_estimator, baseline = False)
        no_baseline_rewards.append(ep_rewards)

    baseline_rewards = []
    for i in range(5) :
        env = Continuous_MountainCarEnv()
        policy = DiscreteSoftmaxPolicy(num_states, num_actions)
        value_estimator = ValueEstimator(num_states)
        ep_rewards = reinforce(env, policy, value_estimator, baseline = True)
        baseline_rewards.append(ep_rewards)


    t = np.arange(0, 5000, 500)

    plt.plot(t, no_baseline_rewards[0], 'C1', label = 'No Baseline')
    plt.plot(t, no_baseline_rewards[1], 'C1')
    plt.plot(t, no_baseline_rewards[2], 'C1')
    plt.plot(t, no_baseline_rewards[3], 'C1')
    plt.plot(t, no_baseline_rewards[4], 'C1')
    plt.plot(t, baseline_rewards[0], 'C2', label = 'Baseline')
    plt.plot(t, baseline_rewards[1], 'C2')
    plt.plot(t, baseline_rewards[2], 'C2')
    plt.plot(t, baseline_rewards[3], 'C2')
    plt.plot(t, baseline_rewards[4], 'C2')
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward Earned Per Episode')
    plt.legend()
    plt.show()

    # # Test time
    # states = []
    # state = discretize_state(env.reset())
    # env.print()
    # done = False
    # while not done:
    #     input("press enter:")
    #     action = policy.act(state)
    #     state, reward, done, _ = env.step(discrete_actions[action])
    #     states.append(state)
    #     state = discretize_state(state)

    

    # You can plot the position of the car over time from your test run above
    # This can be useful for debugging or just seeing how it is doing


