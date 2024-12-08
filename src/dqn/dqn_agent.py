import numpy as np
from collections import deque
import random

# Constants 
NUM_ACTIONS = 4

# Hyperparameters
NUM_EPISODES = 10 # will increase
GAMMA = 0.99
EPSILON = 1.0 # initial epsilon value
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995 # EPSILON_DECAY = 0.995 # 0.995, I'm gonna drop this so that it decays faster. It needs to be a little greedier
BATCH_SIZE = 32 # 64. No 32 is fine
TARGET_UPDATE = 2 # change every 10
MAX_BUFFER_SIZE = 10000
 
class dqn_agent:
  def __init__(self, env, main_model, target_model, replay_buffer):
    self.env = env
    self.main_model = main_model
    self.target_model = target_model
    self.replay_buffer = replay_buffer

  def select_action(self, state, epsilon):
    if np.random.rand() < epsilon:
      return np.random.choice(NUM_ACTIONS)
    new_state = np.expand_dims(state, axis=0) 
    q_values = self.main_model.predict(new_state, verbose=0)
    return np.argmax(q_values[0])

  def train_step(self):
    if self.replay_buffer.size() < BATCH_SIZE: # ensure we have enough experiences to form a batch that we can train on
      return

    # sample a batch of experiences
    batch = self.replay_buffer.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = np.array(states)
    next_states = np.array(next_states)
    rewards = np.array(rewards)
    dones = np.array(dones)

    # states = np.array([self.env.normalize_log_values(s) for s in states])
    # next_states = np.array([self.env.normalize_log_values(s) for s in next_states])
    q_values = self.main_model.predict(states, verbose=0)
    next_q_values = self.target_model.predict(next_states, verbose=0)

    for i in range(BATCH_SIZE):
      if dones[i]:
        q_values[i][actions[i]] = rewards[i] # No future reward in terminal state
      else:
        q_values[i][actions[i]] = rewards[i] + GAMMA * np.max(next_q_values[i])

    # adjust number of epochs
    self.main_model.fit(states, q_values, verbose=0, epochs=1, batch_size=BATCH_SIZE)

  def update_target_model(self):
    self.target_model.set_weights(self.main_model.get_weights())
  
