import random
import gym

EPISODES = 10000
EPSILON = 0.1
GAMMA = 0.9
LEARNING_RATE = 0.1
DISCRETE_STEPS = 10     # 10 discretization

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
