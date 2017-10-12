import random
import gym
from main import*
import numpy as np
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam
import sys

EPISODES = 25000
EPSILON = 0.2
GAMMA = 0.8
LEARNING_RATE = 0.0001
DISCRETE_STEPS = 20     # 10 discretization

def argmax(l):
    """ Return the index of the maximum element of a list
    """
    return max(enumerate(l), key=lambda x:x[1])[0]

def make_state(observation):
    """ Map a 4-dimensional state to a state index
    """
    low = [-4.8, -10., -0.41, -10.]
    high = [4.8, 10., 0.41, 10.]
    state = 0

    for i in range(4):
        # State variable, projected to the [0, 1] range
        state_variable = (observation[i] - low[i]) / (high[i] - low[i])

        # Discretize. A variable having a value of 0.53 will lead to the integer 5,
        # for instance.
        state_discrete = int(state_variable * DISCRETE_STEPS)
        state_discrete = max(0, state_discrete)
        state_discrete = min(DISCRETE_STEPS-1, state_discrete)

        state *= DISCRETE_STEPS
        state += state_discrete

    return state

def qlearning(env, qtable, numstates):
    eps = EPSILON
    cumlist = [0]
    for i in range(EPISODES):
        state = env.reset()
        state = make_state(state) #discritizing state

        terminate = False
        cumulative_reward = 0.0

        # Loop over time-steps
        while not terminate:
            if i > (EPISODES - 20):
        	    env.render()
        	    eps = 0
            # Compute what the greedy action for the current state is
            qvalues = qtable[state]
            greedy_action = argmax(qvalues)

            # Sometimes, the agent takes a random action, to explore the environment
            if random.random() < eps:
                action = random.randrange(2)
            else:
                action = greedy_action

            # Perform the action
            next_state, reward, terminate, info = env.step(action)  # info is ignored
            next_state = make_state(next_state)

            # Update the Q-Table
            td_error = reward + GAMMA * max(qtable[next_state]) - qtable[state][action]
            qtable[state][action] += LEARNING_RATE * td_error

            # Update statistics
            cumulative_reward += reward
            state = next_state
        cumlist.append(0.999*cumlist[-1]+0.001*cumulative_reward)
        print("%03.2f %%" % (100.0*i/EPISODES), end='\r')
		        # Per-episode statistics
        # print(i, cumulative_reward, sep=',')
    env.close()


    return cumlist, qtable

def fapprox(env):
    cumlist = [0]
    lr = 0.0008
    gamma = 0.95
    eps = 0.01
    #initializations
    fisa = np.zeros((8, 2))
    theta = np.random.randn(8, 1)
    thetas = []
    #start
    i = 0
    while i < EPISODES:
        state = env.reset()
        # make fi table from state vector
        fisa[0:4, 0] = np.array(state)
        fisa[4:8, 1] = np.array(state)
        # fisa[8:12, 0] = np.array(state)
        # fisa[12:16, 1] = np.array(state)
        # fisa[16:20, 0] = np.array(state)
        # fisa[20:24, 1] = np.array(state)
        terminate = False
        cumulative_reward = 0.0

        while not terminate:
            if i > (EPISODES - 20):
        	    env.render()
        	    eps = 0.0

            #find action to take
            logits = np.matmul(theta.T, fisa)
            action = logits.argmax()
            if random.random() < eps:
                action = random.randrange(2)

            #take the step
            next_state, reward, terminate, info = env.step(action)
            cumulative_reward+=reward

            #compute Q+(s,a)
            # fisa[0:16, :] = fisa[8:24, :]
            # next_state /= np.array([4.8, 10., 0.41, 10.])
            fisa[0:4, 0] = np.array(next_state)
            fisa[4:8, 1] = np.array(next_state)
            Qplus = np.matmul(theta.T, fisa).max()

            #compute error
            target = reward + gamma*Qplus
            delta = target - logits[0, action]

            #update thetas
            theta -= lr*delta*fisa[:, action][:, np.newaxis]

            state = next_state
        cumlist.append(0.999*cumlist[-1] + 0.001*cumulative_reward)
        print("%03.2f, %03.2f %%" % (cumlist[-1], 100.0*i/EPISODES), end='\r')
        thetas.append(theta.tolist()[0])
        # if len(thetas) > 20 and thetas[-1][0]-thetas[-20][0] < -100:
            # env.render()
            # eps = 0.0
            # i = EPISODES-20
            # break  # early break
        i += 1
    return cumlist, thetas

def fapprox_mlp(env):
    cumlist = [0]
    gamma = 0.8
    eps = 0.1
    #initializations
    fisa = np.zeros((8, 2))
    theta = np.random.randn(8, 1)
    thetas = []
    # MLP construction
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(8,)))
    model.add(Dense(8, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.0008))
    #start
    for i in range(EPISODES):
        state = env.reset()
        # make fi table from state vector
        fisa[0:4, 0] = np.array(state)
        fisa[4:8, 1] = np.array(state)
        # fisa[8:12, 0] = np.array(state)
        # fisa[12:16, 1] = np.array(state)
        # fisa[16:20, 0] = np.array(state)
        # fisa[20:24, 1] = np.array(state)
        terminate = False
        cumulative_reward = 0.0

        while not terminate:
            if i > (EPISODES - 20):
        	    env.render()
        	    eps = 0.0

            #find action to take
            logits = model.predict(fisa.T)
            action = logits.argmax()
            inp = fisa[:, action][:, np.newaxis]
            if random.random() < eps:
                action = random.randrange(2)

            #take the step
            next_state, reward, terminate, info = env.step(action)
            cumulative_reward+=reward

            #compute Q+(s,a)
            # fisa[0:16, :] = fisa[8:24, :]
            next_state /= np.array([4.8, 10., 0.41, 10.])
            fisa[0:4, 0] = np.array(next_state)
            fisa[4:8, 1] = np.array(next_state)
            Qplus = model.predict(fisa.T).max()

            #compute error
            target = np.array([[reward + gamma*Qplus]])

            # update model
            model.fit(inp.T, target, epochs=1, verbose=0)

            state = next_state
        cumlist.append(0.999*cumlist[-1] + 0.001*cumulative_reward)
        print("%03.2f, %03.2f %%" % (cumlist[-1], 100.0*i/EPISODES), end='\r')
    return cumlist
