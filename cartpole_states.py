import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from sklearn.metrics import pairwise_distances

ENV_NAME = "CartPole-v1"
TAU = 0.5
GAMMA = 0.99
DISCOUNT = 0.85
TOL = 1e-2
LEARNING_RATE = 0.001

MEMORY_SIZE = 100000
BATCH_SIZE = 32

EXPLORATION_MAX = 0.5
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.999

######################################################################################
# Double DQN Implementation
#
# reference:
# https://github.com/lsimmons2/double-dqn-cartpole-solution/blob/master/double_dqn.py
######################################################################################

class DDQNSolver:
    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_space
        self.memory = deque(maxlen = MEMORY_SIZE)

        # online network
        self.model = Sequential()
        self.model.add(Dense(24, input_shape = (observation_space, ), activation = "relu"))
        self.model.add(Dense(24, activation = "relu"))
        self.model.add(Dense(self.action_space, activation = "linear"))
        self.model.compile(loss = "mse", optimizer = Adam(lr = LEARNING_RATE))

        # target network
        self.target = Sequential()
        self.target.add(Dense(24, input_shape = (observation_space, ), activation = "relu"))
        self.target.add(Dense(24, activation = "relu"))
        self.target.add(Dense(self.action_space, activation = "linear"))
        self.target.compile(loss = "mse", optimizer = Adam(lr = LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.target.predict(state)
        return np.argmax(q_values[0])

    def update_target_network(self):
        q_network_theta = self.model.get_weights()
        target_network_theta = self.target.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_network_theta,target_network_theta):
            target_weight = target_weight * (1 - TAU) + q_weight * TAU
            target_network_theta[counter] = target_weight
            counter += 1
        self.target.set_weights(target_network_theta)

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                a = np.argmax(self.model.predict(state_next)[0])
                q_update = (reward + GAMMA * self.target.predict(state_next)[0][a])
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose = 0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


##############################################################################
# Upper Confidence Reinforcement Learning with Nearest Neighbor Approximator 
##############################################################################

class UCRL:
    def __init__(self, H, observation_space, action_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.H = H
        # the maximum distance between two (S, A) pairs is 5 in this game
        self.L = 5#action_space + observation_space - 1
        self.L1 = 5
        self.memory = deque(maxlen = MEMORY_SIZE)
        self.states = [[] for action in range(self.action_space)]
        self.next_states = []
        self.states_all = []
        self.rewards = []
        # Q values
        self.q = [[dict() for x in range(self.H)] for action in range(self.action_space)]
        # store sup Q(s', a') for updating Q
        self.q_sup_in_series = None
        self.q_sup_0 = []
        self.last_updated = self.H - 1
        self.distances = None


    def remember(self, state, action, reward, next_state, step):
        self.memory.append((state, action, reward, next_state, step))
        self.states_all.append(np.concatenate((state[0], [action])))
        self.states[action].append(state[0])
        self.next_states.append(next_state[0])
        self.rewards.append(reward)
        self.q[action][0][tuple(state[0])] = 0
        self.q_sup_0.append(0)

    def act(self, run, step, state):
        if run == 0:
            return np.random.choice(self.action_space), None
        nearest = []
        qval = []
        # for (s, a') with a' belonging to A, find the nearest neighbor and the corresponding Q value
        for a in range(self.action_space):
            states = np.array([k for k in self.q[a][self.last_updated].keys()])
            tree = cKDTree(states)
            dd, ii = tree.query(state[0], k = 1, n_jobs = -1)
            nearest.append(states[ii])
            qval.append(self.q[a][self.last_updated][tuple(states[ii])]+ self.L1 * dd)
        print("nearest: %s" % nearest)
        print("qval: %s" % qval)

        action = np.argmax(qval)
        return action, qval[action]

    def update_q(self, run):
        self.distances = pairwise_distances(self.states_all, self.states_all, n_jobs = -1)
        mean_diff = np.inf
        self.last_updated = self.H - 1
        self.q_sup_in_series = self.q_sup_0
        for i in range(1, self.H):
            if mean_diff < TOL: 
                self.last_updated = i - 1
                break
            diff = []
            idx = 0
            for state, action, reward, state_next, _ in self.memory:
                self.q[action][i][tuple(state[0])] = min(np.min(np.array(self.rewards) + self.L * self.distances[idx] + \
                    DISCOUNT * np.array(self.q_sup_in_series)), self.H)
                diff.append(abs(self.q[action][i][tuple(state[0])] - self.q[action][i - 1][tuple(state[0])]))
                idx += 1

            val = []
            for a in range(self.action_space):
                tree = cKDTree(np.array(self.states[a]))
                dd, ii = tree.query(np.array(self.next_states), k = 1, n_jobs = -1)
                states = np.array(self.states[a])[ii]
                val.append(np.array([self.q[a][i][tuple(s)] for s in states]) + self.L1 * dd)
            self.q_sup_in_series = np.max(val, axis = 0)
            mean_diff = np.mean(diff)

def cartpole_UCRL(max_run):
    env = gym.make(ENV_NAME)
    env.seed(1234)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    horizon = 200
    solver = UCRL(horizon, observation_space, action_space)
    run = 0
    scores = []
    consec = 0
    stop = False
    while run <= max_run:
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        terminal = False
        while not terminal and step < horizon:
            # env.render()
            action, _ = solver.act(run, step, state)
            print("state: %s" % state[0])
            print("action: %d" % action)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            #print("reward: %d\n" % reward)
            state_next = np.reshape(state_next, [1, observation_space])
            solver.remember(state, action, reward, state_next, step)
            state = state_next
            step += 1
        print("Run: " + str(run) + ", step: " + str(step))
        scores.append(step)
        consec = consec + 1 if step == horizon else 0
        stop = True if consec == 3 else stop
        if not stop:
            print(run)
            solver.update_q(run)
        run += 1  
    return scores             
        
def cartpole_DDQN(max_run):
    env = gym.make(ENV_NAME)
    env.seed(1234)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    horizon = 200
    dqn_solver = DDQNSolver(observation_space, action_space)
    run = 0
    scores = []
    while run <= max_run:
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        terminal = False
        while not terminal and step < horizon:
            # env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            dqn_solver.experience_replay()
            step += 1
        print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
        scores.append(step)
        if run % 5 == 0:
            dqn_solver.update_target_network()
        run += 1
    return scores

def plot_learning_curve(scores, solver):
    plt.plot(scores)
    plt.ylabel('Score in Episode')
    plt.xlabel('Episode')
    plt.title(solver + ' CartPole Scores in A Learning Trial')
    plt.show()

if __name__ == "__main__":
    np.random.seed(1234)
    scores = []
    for i in range(1):
        scores.append(cartpole_UCRL(40))
    plot_learning_curve(np.mean(scores, axis = 0), "UCRL")
    # scores = cartpole_DDQN(50)
    # plot_learning_curve(scores, "DDQN")
