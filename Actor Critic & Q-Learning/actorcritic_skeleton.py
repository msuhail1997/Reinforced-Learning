import numpy as np
from pendulum import PendulumEnv
import random
import sys
from matplotlib import pyplot as plt


class ActorCritic(object):

    # TODO fill in this function to set up the Actor Critic model.
    # You may add extra parameters to this function to help with discretization
    # Also, you will need to tune the sigma value and learning rates
    def __init__(self, env, gamma=0.99, sigma=2, alpha_value=0.1, alpha_policy=0.05):
        # Upper and lower limits of the state 
        self.min_state = env.min_state
        self.max_state = env.max_state

        self.num_centers = 650
        # TODO initialize the table for the value function
        self.centers = self._create_RBF_centers()
        
        # TODO initialize the table for the mean value of the policy
        self.mean_policy = np.zeros((self.num_centers, )).astype(np.float64)
        self.value = np.zeros((self.num_centers, )).astype(np.float64)

        # Discount factor (don't tune)
        self.gamma = gamma

        # Standard deviation of the policy (need to tune)
        self.sigma = sigma

        # Step sizes for the value function and policy
        self.alpha_value = alpha_value
        self.alpha_policy = alpha_policy
        # These need to be tuned separately

        

    # TODO: fill in this function. 
    # This function should return an action given the
    # state by evaluating the Gaussian polic
    def act(self, state):
        s = self._normalize_state(state)
        x = self._calculate_x(s)
        mu = self._calculate_mu(s, x)
        action = np.random.normal(mu, self.sigma)

        if action < -2 :
            action = -2
        elif action > 2 :
            action = 2

        return action

    # TODO: fill in this function that:
    #   1) Computes the value function gradient
    #   2) Computes the policy gradient
    #   3) Performs the gradient step for the value and policy functions
    # Given the (state, action, reward, next_state) from one step in the environment
    # You may return anything from this function to help with plotting/debugging
    def update(self, state, action, reward, next_state, done, I) :

        s = self._normalize_state(state)
        s_next = self._normalize_state(next_state)
        grad = self._calculate_grad(s, action)
        v_grad = self._calculate_k(s)

        v_s = np.dot(self.value, self._calculate_k(s))
        v_s_next = np.dot(self.value, self._calculate_k(s_next))
        if done :
            v_s_next = 0

        delta = reward + (self.gamma * v_s_next) - v_s

        self.value += self.alpha_value * delta * v_grad
        self.mean_policy += self.alpha_policy * I * delta * grad


    def _calculate_k(self, s) :
        return self._calculate_x(s)

    def _calculate_grad(self, s, action) :
        x = self._calculate_x(s)
        mu = self._calculate_mu(s, x)
        coeff = 1 / (self.sigma * self.sigma)
        grad = coeff * (action - mu) * x
        return grad


    def _calculate_mu(self, s, x) :
        return np.dot(self.mean_policy, x)

    def _create_RBF_centers(self) :
        centers = []
        for i in range(self.num_centers) :
            centers.append([random.random(), random.random(), random.random()])
        return np.array(centers).astype(np.float64)

    def _calculate_x(self, s) :
        xs = []
        x_sigma = 0.01 #can be tuned
        coeff = -1 / (2 * x_sigma * x_sigma)
        xs = coeff * np.linalg.norm(self.centers - s, axis=1)
        final_coeff = 1 / (self.sigma * np.sqrt(2 * np.pi))
        return np.array(final_coeff * np.exp(xs)).astype(np.float64)

    def _normalize_state(self, s) :
        s_0 = (s[0] + 1) / 2
        s_1 = (s[1] + 1) / 2
        s_2 = (s[2] + 8) / 16
        return [s_0, s_1, s_2]



def train(env, model):
    num_episodes = 30000

    all_rewards = []
    grouped_rewards = 0

    # TODO: write training and plotting code here
    for i in range(num_episodes + 1):
        state = env.reset()
        I = 1
        done = False

        while not done :
            action = model.act(state)
            new_state, reward, done, _ = env.step([action])
            grouped_rewards += reward
            model.update(state, action, reward, new_state, done, I)
            I *= model.gamma
            state = new_state

        if i % 100 == 0 and i != 0:
            print(grouped_rewards / 100)
            print(action)
            print(np.max(model.mean_policy))
            all_rewards.append(grouped_rewards / 100)
            grouped_rewards = 0
            print(i)
            if model.sigma > 0.2 :
                model.sigma -= 0.05


    return all_rewards


if __name__ == "__main__":
    env = PendulumEnv()
    policy = ActorCritic(env)
    all_rewards = train(env, policy)



    t = np.arange(0, 30000, 100)

    print(t)
    print(all_rewards)


    plt.plot(t, all_rewards)
    plt.xlabel('Episode Number')
    plt.ylabel('Reward Earned')
    plt.show()






