import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import gym
import gym_game2048
from gym_game2048.envs.env import Game2048Env
from qr_dqn import QRDQN
import os

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)
        
    def push(self, state, action, reward, next_state, done, priority=None):

        max_priority = np.max(self.priorities) if self.buffer else 1.0
        priority = priority if priority is not None else max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
        
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = list(zip(*samples))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])
        dones = np.array(batch[4])
        
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

def calculate_auxiliary_reward(state, reward, max_tile):
    """Enhanced reward function"""
    auxiliary_reward = 0
    
    # Base reward from environment
    if reward > 0:  # Valid merge
        auxiliary_reward += np.log2(reward) * 0.3
    
    # Progressive tile bonuses
    if max_tile >= 256:
        auxiliary_reward += 200
    elif max_tile >= 128:
        auxiliary_reward += 100
    elif max_tile >= 64:
        auxiliary_reward += 30
    elif max_tile >= 32:
        auxiliary_reward += 10
    
    # Strategic rewards
    empty_cells = np.sum(state == 0)
    auxiliary_reward += empty_cells * 0.3  # Reward for keeping space
    
    # Corner bonus
    corners = [state[0,0], state[0,-1], state[-1,0], state[-1,-1]]
    max_corner = max(corners)
    if max_corner == max_tile:
        auxiliary_reward += max_tile * 0.1  # Bonus for keeping max tile in corner
    
    # Monotonicity reward (keeping large numbers together)
    for i in range(4):
        row = state[i,:]
        col = state[:,i]
        if all(row[j] >= row[j+1] for j in range(3)):
            auxiliary_reward += 5
        if all(col[j] >= col[j+1] for j in range(3)):
            auxiliary_reward += 5
            
    return auxiliary_reward


def plot_training_results(episode_rewards, max_tiles, episode_num, save_dir='training_plots'):
    if len(episode_rewards) < 1:
        print("No data to plot yet")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(15, 5))
    
    # Plot episode rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.6, label='Episode Rewards')
    if len(episode_rewards) >= 100:
        moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        plt.plot(range(99, len(episode_rewards)), moving_avg, 
                label='100-episode moving average', linewidth=2)
    plt.title(f'Episode Rewards Over Time (Episode {episode_num})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    if len(episode_rewards) >= 100:  
        plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot max tiles
    plt.subplot(1, 2, 2)
    plt.plot(max_tiles, alpha=0.6, label='Max Tiles')
    if len(max_tiles) >= 100:
        moving_avg = np.convolve(max_tiles, np.ones(100)/100, mode='valid')
        plt.plot(range(99, len(max_tiles)), moving_avg,
                label='100-episode moving average', linewidth=2)
    plt.title(f'Max Tile Values Over Time (Episode {episode_num})')
    plt.xlabel('Episode')
    plt.ylabel('Max Tile Value')
    if len(max_tiles) >= 100:  
        plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_results_episode_{episode_num}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def train_quick():
    env = Game2048Env(board_size=4)
    agent = QRDQN(
        state_size=37, 
        action_size=4,
        memory_size=10000,
        batch_size=256
    )
    
    max_steps_per_episode = 500
    best_game_score = float('-inf')
    best_max_tile = 0

    # Add lists to store metrics
    episode_rewards = []
    episode_max_tiles = []
    episode_num = 100
    tile_frequencies = {}
    
    for episode in range(episode_num):
        state = env.reset()
        game_score = 0
        total_reward = 0
        max_tile = 0
        
        for step in range(max_steps_per_episode):
            current_max_tile = np.max(state)
            max_tile = max(max_tile, current_max_tile)
            
            epsilon = agent.get_dynamic_epsilon(max_tile, 0.3)
            action = agent.select_action(state, epsilon)
            next_state, reward, done, info = env.step(action)

            # Update max_tile after action too
            current_max_tile = np.max(next_state)
            max_tile = max(max_tile, current_max_tile)

            # Update tile frequencies
            unique, counts = np.unique(next_state, return_counts=True)
            
            for tile, count in zip(unique, counts):
                tile_frequencies[tile] = tile_frequencies.get(tile, 0) + count
            
            # Store transition with preprocessed states
            agent.memory.push(
                state,  
                action,
                reward,
                next_state, 
                done
            )
            
            loss = agent.train_step()
            
            if step % 10 == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
            state = next_state
            total_reward += reward
            game_score = info['total_score']  # Get actual game score
            
            if done:
                break
        
        # Store episode metrics
        episode_rewards.append(game_score)
        episode_max_tiles.append(max_tile)
        
        best_game_score = max(best_game_score, game_score)
        best_max_tile = max(best_max_tile, max_tile)
        
        print(f"Episode {episode + 1}/{episode_num}")
        print(f"Game Score: {game_score}")
        print(f"Max Tile: {max_tile}")
        print(f"Best Game Score So Far: {best_game_score}")
        print(f"Best Max Tile So Far: {best_max_tile}")
        print("-" * 50)

    # Calculate averages
    avg_game_score = np.mean(episode_rewards)
    avg_max_tile = np.mean(episode_max_tiles)

    # Print final metrics
    print("Training Summary")
    print(f"Best Game Score: {best_game_score}")
    print(f"Best Max Tile: {best_max_tile}")
    print(f"Average Game Score: {avg_game_score:.2f}")
    print(f"Average Max Tile: {avg_max_tile:.2f}")

    # Generate final plot
    plot_training_results(episode_rewards, episode_max_tiles, episode_num)

if __name__ == "__main__":
    train_quick()