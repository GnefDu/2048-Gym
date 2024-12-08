import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from dqn_agent import dqn_agent
from dqn_model import dqn_model
import sys
sys.path.append('../src')
from envs.env import Game2048Env


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
NUM_EPISODES = 100 # will increase
GAMMA = 0.99
EPSILON = 1.0 # initial epsilon value
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
TARGET_UPDATE = 10 # change every 10
MAX_BUFFER_SIZE = 5000
TRAIN_FREQ = 2

# use a target and a main model
def training_loop():
  max_steps = 500 # 500
  env = Game2048Env(board_size=4)

  main_model = dqn_model().build_model3()
  target_model = dqn_model().build_model3()

  replay_buffer = ReplayBuffer(MAX_BUFFER_SIZE)
  agent = dqn_agent(env, main_model, target_model, replay_buffer)
  epsilon = EPSILON

  # use an object
  max_tiles = []
  episode_rewards = []
  avg_tiles = []
  epsilons = []

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
      if step % TRAIN_FREQ == 0:
        agent.train_step() # only predicts and updates q values if replay buffer has enough experiences

      state = next_state
      total_reward += reward
      board = env.get_board()
      ep_max = max(ep_max, np.max(board))

      if done:
        break

    max_tiles.append(ep_max)
    avg_tiles.append(np.mean(state))
    epsilons.append(epsilon)
    episode_rewards.append(total_reward)

    if episode % TARGET_UPDATE == 0: # update target network periodically
      agent.update_target_model()

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY) # decay epsilon
    max_tiles.append(ep_max)
    print(env.get_board())
    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")
    print(f"Max Tile reached: {ep_max}")
  print(max_tiles)
  print(avg_tiles)
  print(epsilons)
  plot_scores(max_tiles)

def print_board(state):
    state = state * 13
    state = np.power(state, 2)
    state = state.astype(int)
    print(state)
  
def plot_scores(scores):
  plt.figure(figsize=(12, 6))
  plt.plot(scores, label="Max Tile Achieved")
  plt.xlabel("Episodes")
  plt.ylabel("Max Tile")
  plt.title("Max Tile Achieved During Training")
  plt.legend()
  plt.grid()
  plt.show()


def main():
  training_loop()

if __name__ == "__main__":
  main()