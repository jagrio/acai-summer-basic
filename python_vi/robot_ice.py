#! /usr/bin/env python
import argparse
import itertools
import os
import sys
import time
import numpy as np
import random
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib import cm
from copy import deepcopy
import VI

# 0: Normal ice
# 1: Cracked ice
# 2: Wrecked ship (treasure)
# 3: Goal

SQUARE_SIZE = 4

world = [[0,0,0,3],
         [0,1,0,1],
         [0,0,2,1],
         [0,1,1,1]]

world_column = lambda i: enumerate(reversed([row[i] for row in world]))
argmax = lambda iterable, func: max(iterable, key=func)

pSlip = 0.05

# Model

def getTransitionProbability(coord1, action, coord2):
    """
    Given two coordinates and an action, return the transition probability.

    Parameters
    ----------
    coord1: tuple of int
        Two element tuple containing the current coordinates of the robot.
    action: str
        String representing the robot action, options are i
        'up', 'down', 'left', 'right'.
    coord2: tuple of int
        Two element tuple containing the next coordinates of the robot.
    """
    # We check where the robot is
    x1, y1 = coord1
    x2, y2 = coord2

    # We compute the distances traveled by the robot
    robotMovementX = x2 - x1
    robotMovementY = y2 - y1

    # The robot can only move in a single direction
    if robotMovementX * robotMovementY != 0:
        return 0.0

    # Remove cases where the robot cannot possibly move
    if ((action == 'up' and y1 == 3) or
        (action == 'down' and y1 == 0) or
        (action == 'left' and x1 == 0) or
        (action == 'right' and x1 == 3) or
        (world[3-y1][x1] == 3) or
        (world[3-y1][x1] == 1)):
        if robotMovementX + robotMovementY == 0:
            return 1.0
        return 0.0

    if action == 'up':
        if robotMovementX != 0:
            return 0.0
        next_crack = next((i for i,v in world_column(x1) if v == 1 and i >= y1), None)
        # If we arrived at the wall, or at the next crack
        if y2 == next_crack or (y2 == 3 and next_crack == None):
            # If we moved by one, it was inevitable
            if robotMovementY == 1:
                return 1.0
            # Otherwise we slipped
            return pSlip
        # Else it should have been a normal movement
        if robotMovementY == 1:
            return 1 - pSlip
        return 0.0

    if action == 'down':
        if robotMovementX != 0:
            return 0.0
        next_crack = next((i for i,v in reversed(list(world_column(x1))) if v == 1 and i <= y1), None)
        if y2 == next_crack or (y2 == 0 and next_crack == None):
            # If we moved by one, it was inevitable
            if robotMovementY == -1:
                return 1.0
            # Otherwise we slipped
            return pSlip
        # Else it should have been a normal movement
        if robotMovementY == -1:
            return 1 - pSlip
        return 0.0

    if action == 'left':
        if robotMovementY != 0:
            return 0.0
        next_crack = next((i for i,v in reversed(list(enumerate(world[3-y1]))) if v == 1 and i <= x1), None)
        if x2 == next_crack or (x2 == 0 and next_crack == None):
            # If we moved by one, it was inevitable
            if robotMovementX == -1:
                return 1.0
            # Otherwise we slipped
            return pSlip
        # Else it should have been a normal movement
        if robotMovementX == -1:
            return 1 - pSlip
        return 0.0

    if action == 'right':
        if robotMovementY != 0:
            return 0.0
        next_crack = next((i for i,v in enumerate(world[3-y1]) if v == 1 and i >= x1), None)
        if x2 == next_crack or (x2 == 3 and next_crack == None):
            # If we moved by one, it was inevitable
            if robotMovementX == 1:
                return 1.0
            # Otherwise we slipped
            return pSlip
        # Else it should have been a normal movement
        if robotMovementX == 1:
            return 1 - pSlip
        return 0.0


def getReward(coord1, action, coord2):
    """
    Given a state, return the reward.

    Parameters
    ----------
    coord1: tuple of int
        Two element tuple containing the current coordinates of the robot.
    action: str
        String representing the robot action, options are i
        'up', 'down', 'left', 'right'.
    coord2: tuple of int
        Two element tuple containing the next coordinates of the robot.

    Returns
    -------
    reward: float
        If the robot reaches the treasure, reward is +50. Reaching the goal is
        +100. Stepping onto cracked ice is -200.
    """
    if coord1 == coord2:
        return 0.0

    x, y = coord2

    if world[3-y][x] == 3:
        return 100.0
    if world[3-y][x] == 2:
        return 20.0
    if world[3-y][x] == 1:
        return -10.0

    return 0.0


def encodeState(coord):
    """
    Convert from coordinate to state_index.

    Parameters
    ----------
    coord: tuple of int
        Two element tuple containing the position of the robot

    Returns
    -------
    state: int
        Index of the state.
    """
    state = 0
    multiplier = 1
    for c in coord:
        state += multiplier * c
        multiplier *= SQUARE_SIZE

    return state


def decodeState(state):
    """
    Convert from state_index to coordinate.

    Parameters
    ----------
    state: int
        Index of the state.

    Returns
    -------
    coord: tuple of int
        Two element tuple containing the position of the robot
    """
    coord = []
    for _ in range(2):
        c = state % SQUARE_SIZE
        state /= SQUARE_SIZE
        coord.append(c)
    return tuple(coord)


#RENDERING

# Special character to go back up when drawing.
up = list("\033[XA")
# Special character to go back to the beginning of the line.
back = list("\33[2K\r")

def goup(x):
    """ Moves the cursor up by x lines """
    while x > 8:
        up[2] = '9'
        print "".join(up)
        x -= 8

    up[2] = str(x + 1)
    print "".join(up)

def godown(x):
    """ Moves the cursor down by x lines """
    while x:
        print ""
        x -= 1

def printState(coord):
    """
    Draw the grid world.

    - @ represents the robot.
    - T represents the treasure.
    - X represents the cracks in the ice.
    - G represents the goal.

    Parameters
    ----------
    coord: tuple of int
        Two element tuple containing the position of the robot
    """
    r_x, r_y = coord
    for y in range(SQUARE_SIZE-1, -1, -1):
        for x in range(SQUARE_SIZE):
            if (r_x, r_y) == (x, y):
                print "@",
            elif world[3-y][x] == 3:
                print "G",
            elif world[3-y][x] == 2:
                print "T",
            elif world[3-y][x] == 1:
                print "X",
            else:
                print ".",
        print ""

def sampleProbability(vec):
    """
    Returns the id of a random element of vec, assuming vec is a list of
    elements which sum up to 1.0.

    The random element is returned with the same probability of its value in
    the input vector.
    """
    p = random.uniform(0, 1)
    for i, v in enumerate(vec):
        if v > p:
            return i
        p -= v

    return i

def sampleSR(s, a, T):
    s1 = sampleProbability(T[s][a])

    return s1, getReward(decodeState(s), A[a], decodeState(s1))

def isTerminal(coord):
    x, y = coord
    if ((world[3-y][x] == 3) or
        (world[3-y][x] == 1)):
        return True
    return False

# Statespace contains the robot (x, y). Note that
# S = [(r_x, r_y), .. ]
S = list(itertools.product(range(SQUARE_SIZE), repeat=2))

# A = robot actions
A = ['up', 'down', 'left', 'right']

def valueIteration(ns, na, discount, horizon, epsilon, T, R):
    """
    Perform the Value Iteration solver. Expects as input the number of states
    and actions, the discount (gamma), the horizon, the minimum tolerated error,
    the transition probabilities among states and the corresponding rewards.

    Returns
    -------
    solution: triple
        First element is a boolean that indicates whether the method has
        converged. The second element is the best policy. The third
        element are the values.
    """
    Q, V, A1 = [[0]*na]*ns, [0]*ns, [0]*ns
    delta, iters = 10, 0
    while delta > epsilon or iters < horizon :
        delta = 0
        for s in range(ns):
            v = V[s]
            Q[s] = [[T[s][a][sn]*(R[s][a][sn]+discount*V[sn])
                    for sn in range(ns)] for a in range(na)]
            Q[s] = [sum(qi) for qi in Q[s]]
            V[s] = max(Q[s])
            A1[s] = np.argmax(Q[s])
            delta = max(delta, abs(v - V[s]))
        iters += 1
    if delta < epsilon:  #  iteration ended due to convergence
        return True, A1, V
    else:
        return False, A1, V

def policyIteration(ns, na, discount, horizon, epsilon, T, R):
    """
    Perform the Policy Iteration solver. Expects as input the number of states
    and actions, the discount (gamma), the horizon, the minimum tolerated error,
    the transition probabilities among states and the corresponding rewards.

    Returns
    -------
    solution: triple
        First element is the best policy. The second
        element are the values.
    """
    # V, A1 = np.random.randint(0,100,(ns)).tolist(), np.random.randint(0,na,(ns),'int').tolist()
    Q, V, A1 = [[0]*na]*ns , [0]*ns, np.random.randint(0,na-1,(ns)).tolist()
    delta, policy_stable = 10, False
    while not policy_stable:
        # policy evaluation
        delta = 10
        while delta > epsilon:
            delta = 0
            for s in range(ns):
                v = V[s]
                a1 = A1[s]
                V[s] = sum([T[s][a1][sn]*(R[s][a1][sn]+discount*V[sn])
                            for sn in range(ns)])
                delta = max(delta, abs(v - V[s]))
        # policy improvement
        policy_stable = True
        for s in range(ns):
            tmpA1 = A1[s]
            Q[s] = [[T[s][a][sn]*(R[s][a][sn]+discount*V[sn])
                     for sn in range(ns)] for a in range(na)]
            Q[s] = [sum(qi) for qi in Q[s]]
            A1[s] = np.argmax(Q[s])
            if not tmpA1 == A1[s]:
                policy_stable = False
    return A1, V

def qLearning(ns, na, discount, horizon, epsilon, T, R):
    """
    Perform the Q-Learning solver. Expects as input the number of states
    and actions, the discount (gamma), the horizon, the epsilon,
    the transition probabilities among states and the corresponding rewards.

    Returns
    -------
    solution: tuple
        First element is the best policy.
        Second element are the values.
    """
    horizon = 10000
    Q, V, A1 = np.random.rand(ns,na), [0]*ns, [0]*ns
    iters, lr, epsilon, beta, discount = 0, .2, 1-1e-4, 0, .97
    while iters < horizon :
        s = np.random.randint(0,ns)
        tmpdropprob = 1
        while not isTerminal(decodeState(s)) and tmpdropprob==1:
            # print s
            tmpa = np.argmax(Q[s])
            a = tmpa
            # probs = np.exp(beta*Q[s])/np.sum(np.exp(beta*Q[s]))
            # a = sampleProbability(probs)
            if sampleProbability([1-epsilon, epsilon])==1:  # if 1
                a = np.random.randint(0,na)
            # print T[s][a]
            sn = sampleProbability(T[s][a])  # choose a random new state
            r = R[s][a][sn]
            Q[s][a] += lr*(r+discount*(max(Q[sn]))-Q[s][a])
            s = sn
            tmpdropprob = sampleProbability([0.001, 1-0.001])
            if tmpdropprob==0:
                iters -=1
                epsilon /= 0.9
                # epsilon = iters**.5
            # raw_input()
        iters += 1
        # beta += 0.01
        epsilon *= 0.9
        # epsilon = 1/iters**.5
    for s in range(ns):
        V[s] = max(Q[s])
        A1[s] = np.argmax(Q[s])
    return A1, V


def solve_mdp(horizon, epsilon, discount=0.9, method='VIours',
              solution_v=[], solution_a=[], printit=False, plotit=False):
    """
    Construct the gridworld MDP, and solve it using value iteration. Print the
    best found policy for sample states.

    Returns
    -------
    solution: tuple
        First element is a boolean that indicates whether the method has
        converged. The second element is the value function. The third
        element is the Q-value function, from which a policy can be derived.
    """
    print time.strftime("%H:%M:%S"), "- Constructing MDP..."

    # T gives the transition probability for every s, a, s' triple.
    # R gives the reward associated with every s, a, s' triple.
    T = []
    R = []
    for state in range(len(S)):
        coord = decodeState(state)
        T.append([[getTransitionProbability(coord, action,
                                            decodeState(next_state))
                   for next_state in range(len(S))] for action in A])
        R.append([[getReward(coord, action, decodeState(next_state))
                   for next_state in range(len(S))] for action in A])
    if solution_a==[] or solution_v==[]:
        if method=='VIours':
            conv, solution_a, solution_v = valueIteration(len(S), len(A), discount, horizon, epsilon, T, R)
        elif method=='VItheirs':
            solution_a, solution_v = VI.valueIteration(len(S), len(A), discount, horizon, epsilon, T, R)
            conv = True
        elif method=='PIours':
            solution_a, solution_v = policyIteration(len(S), len(A), discount, horizon, epsilon, T, R)
            conv = True
    else:
        conv = True

    s = 0

    totalReward = []
    for t in xrange(horizon):
        if printit:
            printState(decodeState(s))

        if isTerminal(decodeState(s)):
            break

        s1, r = sampleSR(s, solution_a[s], T)
        if len(totalReward) > 0:
            totalReward += [totalReward[-1] + r]
        else:
            totalReward += [r]
        s = s1

        if printit:
            goup(SQUARE_SIZE)

        state = encodeState(coord)

        # Sleep 1 second so the user can see what is happening.
        if printit:
            time.sleep(1)
    if plotit:
        plt.plot(totalReward)
        plt.xlabel("Iteration")
        plt.ylabel("Cummulative Reward")
        plt.show()
    return (conv, solution_v, solution_a, totalReward)

def solve_ql(horizon, epsilon, discount=0.9, solution_v=[], solution_a=[],
             printit=False, plotit=False):
    """
    Q-Learning and solve it. Print the
    best found policy for sample states.

    Returns
    -------
    solution: tuple
        First element is the value function. The second
        element is the Q-value function, from which a policy can be derived.
    """
    print time.strftime("%H:%M:%S"), "- Constructing QL..."

    # T gives the transition probability for every s, a, s' triple.
    # R gives the reward associated with every s, a, s' triple.
    T = []
    R = []
    for state in range(len(S)):
        coord = decodeState(state)
        T.append([[getTransitionProbability(coord, action,
                                            decodeState(next_state))
                   for next_state in range(len(S))] for action in A])
        R.append([[getReward(coord, action, decodeState(next_state))
                   for next_state in range(len(S))] for action in A])

    if solution_a==[] or solution_v==[]:
        solution_a, solution_v = qLearning(len(S), len(A), discount, horizon, epsilon, T, R)

    s = 0

    totalReward = []
    for t in xrange(horizon):
        if printit:
            printState(decodeState(s))

        if isTerminal(decodeState(s)):
            break

        s1, r = sampleSR(s, solution_a[s], T)
        if len(totalReward) > 0:
            totalReward += [totalReward[-1] + r]
        else:
            totalReward += [r]
        s = s1

        if printit:
            goup(SQUARE_SIZE)

        state = encodeState(coord)

        # Sleep 1 second so the user can see what is happening.
        if printit:
            time.sleep(0.1)
    if plotit:
        plt.plot(totalReward)
        plt.xlabel("Iteration")
        plt.ylabel("Cummulative Reward")
        plt.show()
    return (solution_v, solution_a, totalReward)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ho', '--horizon', default=100000, type=int,
                        help="Horizon parameter for value iteration")
    parser.add_argument('-e', '--epsilon', default=0.0001, type=float,
                        help="Epsilon parameter for value iteration")
    parser.add_argument('-d', '--discount', default=0.9, type=float,
                        help="Discount parameter for value iteration")

    args = parser.parse_args()
    numexps = 20
    r0, r1, r2, r3 = [0]*numexps, [0]*numexps, [0]*numexps, [0]*numexps
    _, solv1, sola1, _ = solve_mdp(args.horizon, args.epsilon)
    _, solv2, sola2, _ = solve_mdp(args.horizon, args.epsilon, 0.9, 'VItheirs')
    _, solv3, sola3, _ = solve_mdp(args.horizon, args.epsilon, 0.9, 'PIours')
    tmpargs = [args.horizon, args.epsilon, 0.9, 'VIours', solv1, sola1]
    out0 = Parallel(n_jobs=-1)(delayed(solve_mdp) (*tmpargs) for k in range(numexps))
    out1 = Parallel(n_jobs=-1)(delayed(solve_ql) (args.horizon, args.epsilon) for k in range(numexps))
    tmpargs = [args.horizon, args.epsilon, 0.9, 'VItheirs', solv2, sola2]
    out2 = Parallel(n_jobs=-1)(delayed(solve_mdp) (*tmpargs) for k in range(numexps))
    tmpargs = [args.horizon, args.epsilon, 0.9, 'PIours', solv3, sola3]
    out3 = Parallel(n_jobs=-1)(delayed(solve_mdp) (*tmpargs) for k in range(numexps))
    for i in range(numexps):
        _, v0, _, tr0 = out0[i]
        v1, _, tr1 = out1[i]
        _, v2, _, tr2 = out2[i]
        _, v3, _, tr3 = out3[i]
        r0[i] = tr0[-1]
        r1[i] = tr1[-1]
        r2[i] = tr2[-1]
        r3[i] = tr3[-1]
    print r0, np.mean(r0)
    print r1, np.mean(r1)
    print r2, np.mean(r2)
    print r3, np.mean(r3)
    plt.plot(r0)
    plt.hold
    plt.plot(r1)
    plt.hold
    plt.plot(r2)
    plt.hold
    plt.plot(r3)
    plt.show(block=False)
    plt.matshow(np.reshape(v0, (SQUARE_SIZE, SQUARE_SIZE)),cmap=cm.gray)
    plt.gca().invert_yaxis()
    plt.show(block=False)
    plt.matshow(np.reshape(v1, (SQUARE_SIZE, SQUARE_SIZE)),cmap=cm.gray)
    plt.gca().invert_yaxis()
    plt.show(block=False)
    plt.matshow(np.reshape(v2, (SQUARE_SIZE, SQUARE_SIZE)),cmap=cm.gray)
    plt.gca().invert_yaxis()
    plt.show(block=False)
    plt.matshow(np.reshape(v3, (SQUARE_SIZE, SQUARE_SIZE)),cmap=cm.gray)
    plt.gca().invert_yaxis()
    plt.show()
