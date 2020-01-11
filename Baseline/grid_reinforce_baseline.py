from grid_world import *
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import random


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
        return action_taken


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
    def predict(self, state):
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
def reinforce(env, policy, value_estimator, baseline=True):
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
            state, reward, done = env.step(action)
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

    # No Baseline Policy
    # env = GridWorld(MAP2)
    # policy = DiscreteSoftmaxPolicy(env.get_num_states(), env.get_num_actions())
    # value_estimator = ValueEstimator(env.get_num_states())
    # ep_rewards = reinforce(env, policy, value_estimator, baseline=False)

    # # Baseline Policy
    # env = GridWorld(MAP2)
    # policy_baseline = DiscreteSoftmaxPolicy(env.get_num_states(), env.get_num_actions())
    # value_estimator = ValueEstimator(env.get_num_states())
    # ep_rewards_baseline = reinforce(env, policy_baseline, value_estimator, baseline=True)

    # Graph learning curve
    # print(ep_rewards[:5])
    # print(ep_rewards_baseline[:5])

    # t = np.arange(0, 20000, 500)

    # er_b, = plt.plot(t, ep_rewards_baseline, label='Baseline')
    # er, = plt.plot(t, ep_rewards, label='No Baseline')
    # plt.legend([er_b, er], ['Baseline', 'No Baseline'])
    # plt.xlabel('Episode Number')
    # plt.ylabel('Reward Earned')
    # plt.show()

    # # Inspect Policy
    # print('No Baseline Policy Inspection')
    # for i in range(policy.num_states) :
    #     print(i, policy._softmax(i))

    # # Inspect Policy
    # print('Baseline Policy Inspection')
    # for i in range(policy_baseline.num_states) :
    #     print(i, policy_baseline._softmax(i))


    print('20 No Baseline Policies')
    for i in range(20) :
        env = GridWorld(MAP2)
        policy = DiscreteSoftmaxPolicy(env.get_num_states(), env.get_num_actions())
        value_estimator = ValueEstimator(env.get_num_states())
        ep_rewards = reinforce(env, policy, value_estimator, baseline=False)

        state = env.reset()
        reward = 0
        rewards = []
        actions = []
        done = False
        while not done:
            action = policy.act(state)
            actions.append(action)
            state, reward, done = env.step(action)
            rewards.append(reward)
        print(actions, rewards)

    print('20 Baseline Policies')
    for i in range(20) :
        env = GridWorld(MAP2)
        policy = DiscreteSoftmaxPolicy(env.get_num_states(), env.get_num_actions())
        value_estimator = ValueEstimator(env.get_num_states())
        ep_rewards = reinforce(env, policy, value_estimator, baseline=True)

        state = env.reset()
        reward = 0
        rewards = []
        actions = []
        done = False
        while not done:
            action = policy.act(state)
            actions.append(action)
            state, reward, done = env.step(action)
            rewards.append(reward)
        print(actions, rewards)


    # #Test time
    # state = env.reset()
    # env.print()
    # done = False
    # while not done:
    #     input("press enter:")
    #     action = policy.act(state)
    #     state, reward, done = env.step(action)
    #     env.print()


