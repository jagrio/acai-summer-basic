#! /usr/bin/env python
from agents import *
import matplotlib.pyplot as plt

def main():
    average_cumulative_reward = 0.0

    # Create the Gym environment (CartPole)
    env = gym.make('CartPole-v1')

    print('Action space is:', env.action_space)
    print('Observation space is:', env.observation_space)

    # Q-table for the discretized states, and two actions
    num_states = DISCRETE_STEPS ** 4
    qtable = [[0., 0.] for state in range(num_states)]

    # Choose method
    # C, A = qlearning(env, qtable, num_states)
    # plt.plot(C)
    # plt.show()

    # C, thetas = fapprox(env)
    C = fapprox_mlp(env)

    plt.plot(C)
    plt.show()
    # plt.plot(thetas)
    # plt.show()


if __name__ == '__main__':
    main()
