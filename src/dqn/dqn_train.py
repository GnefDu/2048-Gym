import numpy as np
from collections import deque
import random
from dqn_agent import dqn_agent
from dqn_model import dqn_model
import os
import sys
sys.path.append('../src')
from envs.env import Game2048Env



# CONSTANTS
NUM_ACTIONS = 4
LEARNING_RATE = 0.0001 # Could it be too small?? No, the learning rate has to be small for the model to learn properly for this kind of task
# LEARNING_RATE = 0.00001

class ReplayBuffer:
  def __init__(self, max_size):
    self.buffer = deque(maxlen=max_size)
  def add(self, experience):
    self.buffer.append(experience)
  def sample(self, batch_size):
    batch = random.sample(self.buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    return batch
  def size(self):
    return len(self.buffer)

  
# Hyperparameters
NUM_EPISODES = 10 # will increase
GAMMA = 0.99
EPSILON = 1.0 # initial epsilon value
EPSILON_MIN = 0.01
# EPSILON_DECAY = 0.995 # 0.995, I'm gonna drop this so that it decays faster. It needs to be a little greedier
EPSILON_DECAY = 0.5
BATCH_SIZE = 32 # 64
TARGET_UPDATE = 2 # change every 10
MAX_BUFFER_SIZE = 10000



# use a target and a main model
def training_loop():
  max_steps = 500 # 500
  env = Game2048Env(board_size=4)
  main_model = dqn_model().build_model3()
  target_model = dqn_model().build_model3()

  replay_buffer = ReplayBuffer(MAX_BUFFER_SIZE)
  agent = dqn_agent(env, main_model, target_model, replay_buffer)
  epsilon = EPSILON
  maxTile = 0
  max_scores = []
  for episode in range(NUM_EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    ep_max = 0
    for step in range(max_steps):
      action = agent.select_action(state, epsilon) # pick action using e-greedy policy

      next_state, reward, done, _ = env.step(action)
      replay_buffer.add((state, action, reward, next_state, done)) # store experience in replay buffer as a tuple of state, action, reward, next_state, and done flag
      
      # train DQN
      agent.train_step() # only predicts and updates q values if replay buffer has enough experiences

      state = next_state
      total_reward += reward
      ep_max = max(ep_max, np.max(state))
      maxTile = max(ep_max, maxTile)
      print_board(state)


      if done:
        break

    if episode % TARGET_UPDATE == 0: # update target network periodically
      agent.update_target_model()

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY) # decay epsilon
    max_scores.append(ep_max*2048)
    print_board(state)
    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")
    print(f"Max Tile reached: {ep_max}")
  print(max_scores)

def print_board(state):
    state = state * 2048
    state = state.astype(int)
    print(state)
  

def main():
  training_loop()

if __name__ == "__main__":
  main()