import random
import numpy
import copy


class Agent:

    def __init__(self, nr_actions):
        self.nr_actions = nr_actions

    def policy(self, state):
        pass

    def update(self, state, action, reward, next_state):
        pass


class RandomAgent:

    def __init__(self, nr_actions):
        self.nr_actions = nr_actions

    def policy(self, state):
        return random.choice(range(self.nr_actions))

    def update(self, state, action, reward, next_state):
        pass


class QLearningAgent:

    def __init__(self, nr_actions, discount_factor, learning_rate, epsilon, epsilon_decay, min_epsilon=.01):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.nr_actions = nr_actions
        self.Q_table = {}
        self.experience = []

    def Q_values(self, state):
        if repr(state) not in self.Q_table:
            return numpy.zeros(self.nr_actions)
        else:
            return self.Q_table[repr(state)]

    def policy(self, state):
        # Select action according to epsilon-greedy strategy
        if numpy.random.rand() <= self.epsilon:
            return random.choice(range(self.nr_actions))
        else:
            Q_values = self.Q_values(state)
            return numpy.random.choice(numpy.flatnonzero(Q_values == Q_values.max()))

    def update(self, state, action, reward, next_state):
        old_Q_value = self.Q_values(state)[action]
        new_Q_value = reward + self.discount_factor * \
            numpy.max(self.Q_values(next_state))
        # Lazy initialization of Q-values
        self.Q_table[repr(state)] = self.Q_values(state)
        new_value =  (1. - self.learning_rate) * old_Q_value + self.learning_rate * new_Q_value
        self.Q_table[repr(state)][action] = new_value
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)
        return abs(new_Q_value - old_Q_value)
