import random
import gym
import numpy as np
import cv2
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from sklearn.metrics import pairwise_distances
from skimage.color import rgb2gray
from scipy import ndimage

ENV_NAME = "CartPole-v1"
DISCOUNT = 0.85
TOL = 1e-2
MEMORY_SIZE = 100000

def crop_image(img, shape):
    cv2.resize(img, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
    img = np.array(crop_center(img, shape[0], shape[1]).flatten(), dtype = int)
    return img

def crop_center(img, cropx, cropy):
    y,x = img.shape
    startx = x//2 - (cropx//2)
    starty = y//2 - (cropy//2)    
    return img[starty:starty + cropy, startx:startx + cropx]

##############################################################################
# Upper Confidence Reinforcement Learning with Nearest Neighbor Approximator 
##############################################################################

class UCRL:
    def __init__(self, H, observation_space, action_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.H = H
        # the maximum distance between two (S, A) pairs is 5 in this game
        self.L = 4e-2#action_space + observation_space - 1
        self.L2 = 4e-2
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
        if len(state[0]) < self.observation_space:
            state = [np.pad(state[0], (0, self.observation_space - len(state[0])), 'constant')]
        if len(next_state[0]) < self.observation_space:
            next_state = [np.pad(next_state[0], (0, self.observation_space - len(next_state[0])), 'constant')]
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
        if len(state[0]) < self.observation_space:
            state = [np.pad(state[0], (0, self.observation_space - len(state[0])), 'constant')]
        nearest = []
        qval = []
        # for (s, a') with a' belonging to A, find the nearest neighbor and the corresponding Q value
        for a in range(self.action_space):
            states = np.array([k for k in self.q[a][self.last_updated].keys()],dtype=int)
            tree = cKDTree(states)
            dd, ii = tree.query(state[0], p = 1, k = 1, n_jobs = -1)
            nearest.append(ii)
            qval.append(self.q[a][self.last_updated][tuple(states[ii])] + self.L2 * dd)

        print("nearest: %s" % nearest)
        print("qval: %s" % qval)
        action = np.argmax(qval)
        return action, qval[action]

    def update_q(self, run):
        self.distances = pairwise_distances(self.states_all, self.states_all, metric = 'l1', n_jobs = -1)
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
                self.q[action][i][tuple(state[0])] = np.min(np.array(self.rewards) + self.L * self.distances[idx] + \
                    DISCOUNT * np.array(self.q_sup_in_series))
                diff.append(abs(self.q[action][i][tuple(state[0])] - self.q[action][i - 1][tuple(state[0])]))
                idx += 1

            val = []
            for a in range(self.action_space):
                tree = cKDTree(np.array(self.states[a]))
                dd, ii = tree.query(np.array(self.next_states), p = 1, k = 1, n_jobs = -1)
                states = np.array(self.states[a])[ii]
                val.append(np.array([self.q[a][i][tuple(s)] for s in states]) + self.L2 * dd)
            self.q_sup_in_series = np.max(val, axis = 0)
            mean_diff = np.mean(diff)


def cartpole_UCRL(max_run):
    env = gym.make(ENV_NAME)
    env.seed(1234)

    crop_size = (50, 200)
    observation_space = crop_size[0] * crop_size[1]
    action_space = env.action_space.n
    horizon = 200
    solver = UCRL(horizon, observation_space, action_space)
    run = 0
    scores = []
    consec = 0
    stop = False

    while run <= max_run:
        state = env.reset()

        step = 0
        terminal = False
        img = rgb2gray(env.render(mode = 'rgb_array'))
        img_prev = img = crop_image(img, crop_size)
        
        while not terminal and step < horizon:
            action, _ = solver.act(run, step, [(img - img_prev).flatten()])
            print("action: %d" % action)

            state_next, reward, terminal, info = env.step(action)
            img_next = rgb2gray(env.render(mode = 'rgb_array'))
            img_next = crop_image(img_next, crop_size)
            reward = reward if not terminal else -reward

            solver.remember([(img - img_prev).flatten()], action, reward, [(img_next - img).flatten()], step)
            state = state_next
            img_prev = img
            img = img_next
            step += 1

        print("Run: " + str(run) + ", step: " + str(step) + "\n")
        scores.append(step)
        consec = consec + 1 if step == horizon else 0
        stop = True if consec == 3 else stop
        if not stop:
            solver.update_q(run)
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
        scores.append(cartpole_UCRL(50))
    plot_learning_curve(np.mean(scores, axis = 0), "UCRL")
