from tictactoe_env import TicTacToe
import pdb 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import random

def tabular_epsilon_greedy_policy(Q, eps, state):
        action = 0
        rand = random.uniform(0, 1)
        if rand <= eps :
                action = random.randrange(1, 9)
        else :
                action = np.argmax(Q[state])
        return action

class QLearning(object):
        def __init__(self, num_states, num_actions, alpha=0.5, gamma=0.9):
                self.Q = {}
                self.alpha = alpha
                self.gamma = gamma

        def update(self, state, action, reward, next_state, done):
                self.Q[state][action]+= self.alpha * (reward + (self.gamma * np.max(self.Q[next_state])) - self.Q[state][action])
                        
def main():
        env = TicTacToe()
        done = False
        state = env.reset()
        epsilon_greedy_policy = QLearning(19683, 9)  #3^9=19683
        num_episodes = 100
        eps = 0.005
        epsilon_greedy_rewards = []
        epsilon_greedy_start_qs = []
        saved_Q_table = []
        Qs = []
        Qdict=epsilon_greedy_policy.Q
        count=0
        for i in range(num_episodes):
                state = env.reset()
                done = False
                ep_rewards = 0
                list_states=[''.join(list(state.flatten().astype(str)))]
                list_actions=[]
                list_rewards=[]
                while done == False:
                        currentstateval=''.join(list(state.flatten().astype(str)))
                        if currentstateval not in Qdict.keys():
                                 Qdict[currentstateval]=np.zeros((9))
                        action = tabular_epsilon_greedy_policy(epsilon_greedy_policy.Q, eps, currentstateval)
                        #action = int(input("Choose where to place (1 to 9): "))
                        if state[int((action)/3)][(action)%3]==0:  #to eliminate output of an action to a cell that is already occupied.
                                next_state,reward,done = env.step(action+1)
                                list_actions.append(action)
                                list_rewards.append(reward)
                                stateval=''.join(list(next_state.flatten().astype(str)))
                                list_states.append(stateval)
                                ep_rewards += reward
                                if stateval not in Qdict.keys():
                                        Qdict[stateval]=np.zeros((9))
                                        state = next_state
                                state = next_state
                epsilon_greedy_rewards.append(ep_rewards)
                for i in range(len(list_actions)) :
                        action=list_actions[i]
                        state=list_states[i]
                        reward=list_rewards[i]
                        next_state=list_states[i+1]
                        epsilon_greedy_policy.update(state, action, reward,next_state,done)
        t = np.arange(0, num_episodes)
        plt.plot(t, epsilon_greedy_rewards)
        plt.xlabel('Episode Number')
        plt.ylabel('Starting State Q Value (Best Action)')
        plt.show()


if __name__ == '__main__':
        main()
