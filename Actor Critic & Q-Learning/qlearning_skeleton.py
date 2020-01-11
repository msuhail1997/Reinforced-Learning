import numpy as np
from grid_world import *
import matplotlib as mpl
import matplotlib.pyplot as plt

import random


# TODO: Fill this function in
# Function that takes an a 2d numpy array Q (num_states by num_actions)
# an epsilon in the range [0, 1] and the state
# to output actions according to an Epsilon-Greedy policy
# (random actions are chosen with epsilon probability)
def tabular_epsilon_greedy_policy(Q, eps, state):
    action = 0 # placeholder
    rand = random.uniform(0, 1)
    if rand <= eps :
        action = random.randrange(0, Q.shape[1])
    else :
        action = np.argmax(Q[state, :])
    return action


class QLearning(object):
    # Initialize a Qlearning object
    # alpha is the "learning_rate"
    def __init__(self, num_states, num_actions, alpha=0.5, gamma=0.9):
         # initialize Q values to something
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma


    # TODO: fill in this function
    # updates the Q value table
    # with a (state, action, reward, next_state) from one step in the environment
    # done is a bool indicating if the episode terminated at this step
    # you can return anything from this function to help with plotting/debugging
    def update(self, state, action, reward, next_state, done):
        self.Q[state, action] += self.alpha * (reward + (self.gamma * np.max(self.Q[next_state, :])) - self.Q[state, action])



# TODO: fill this in
# run the greedy policy (no randomness) on the environment for niter number of times
# and return the fraction of times that it reached the goal
def evaluate_greedy_policy(qlearning, env, niter=100):
    reached_goal = 0
    for i in range(niter) :
        state = env.reset()
        done = False
        while not done :
            action = np.argmax(qlearning.Q[state, :])
            next_state, reward, done = env.step(action)
            if reward == 100 :
                reached_goal += 1
            state = next_state
    return (reached_goal / niter) * 100


def visualize(Q, n_cols) :

    top_values = []
    for i in range(Q.shape[0]) :
        top_values.append(np.max(Q[i, :]))

    matrix = np.zeros((len(top_values) // n_cols, n_cols))
    for i in range(matrix.shape[0]) :
        for j in range(matrix.shape[1]) :
            matrix[i, j] = top_values[j + n_cols*i]


    cmap = mpl.colors.ListedColormap(['red', 'orange', 'gold', 'yellow', 'black', 'blue', 'seagreen', 'green'])


    bounds=[-100,-75, -50, -25, 0, 25, 50, 75,100]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    img = plt.imshow(matrix,interpolation='nearest', cmap = cmap,norm=norm)
    plt.colorbar(img,cmap=cmap, norm=norm,boundaries=bounds,ticks=[-75, -50, -25, 0, 25, 50, 75])

    plt.show()



if __name__ == '__main__' :
    env = GridWorld(MAP4)

    greedy_policy = QLearning(env.get_num_states(), env.get_num_actions())
    ## TODO: write training code here
    num_episodes = 1000
    greedy_rewards = []
    for i in range(num_episodes):
        state = env.reset()
        done = False
        ep_rewards = 0
        while not done :
            action = np.argmax(greedy_policy.Q[state, :])
            next_state, reward, done = env.step(action)
            greedy_policy.update(state, action, reward, next_state, done)
            state = next_state
            ep_rewards += reward
        greedy_rewards.append(ep_rewards)

    epsilon_greedy_policy = QLearning(env.get_num_states(), env.get_num_actions())

    num_episodes = 1000
    eps = 0.2
    epsilon_greedy_rewards = []
    epsilon_greedy_start_qs = []
    saved_Q_table = []
    Qs = []
    for i in range(num_episodes) :
        state = env.reset()
        done = False
        ep_rewards = 0
        while not done :
            action = tabular_epsilon_greedy_policy(epsilon_greedy_policy.Q, eps, state)
            next_state, reward, done = env.step(action)
            epsilon_greedy_policy.update(state, action, reward, next_state, done)
            state = next_state
            ep_rewards += reward
       # eps -= (0.1 * eps) #reduce epsilon over time
        if i == 0 or i == 10 or i == 99 or i == 499 or i == 999 or i == 5000:
            Qs.append(epsilon_greedy_policy.Q.copy())
        start_q = np.max(epsilon_greedy_policy.Q[0, :])
        epsilon_greedy_start_qs.append(start_q)
        epsilon_greedy_rewards.append(ep_rewards)



    t = np.arange(0, num_episodes)
   # #plt.plot(t, greedy_rewards)
    plt.plot(t, epsilon_greedy_rewards)
    plt.xlabel('Episode Number')
    plt.ylabel('Starting State Q Value (Best Action)')
    plt.show()

    print(epsilon_greedy_policy.Q)


    for Q in Qs :
        visualize(Q, env.n_cols)

    # evaluate the greedy policy to see how well it performs
    # frac = evaluate_greedy_policy(greedy_policy, env)
    # print("Finding goal " + str(frac) + "% of the time.")


    done = False
    state = env.reset()
    while not done:
        action = np.argmax(epsilon_greedy_policy.Q[state, :])
        state, reward, done = env.step(action)
        env.print()



