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
        auxiliary_reward += np.log2(reward) * 0.2
    
    # Bonus for reaching new max tiles
    if max_tile >= 256:
        auxiliary_reward += 100
    elif max_tile >= 128:
        auxiliary_reward += 50
    elif max_tile >= 64:
        auxiliary_reward += 20
    
    # Strategic rewards
    empty_cells = np.sum(state == 0)
    auxiliary_reward += empty_cells * 0.2  # Reward for keeping space
    
    # Corner bonus
    corners = [state[0,0], state[0,-1], state[-1,0], state[-1,-1]]
    max_corner = max(corners)
    if max_corner == max_tile:
        auxiliary_reward += max_tile * 0.1  # Bonus for keeping max tile in corner
    
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


def train():
    # Initialize environment and agent
    env = Game2048Env(board_size=4, penalty=-32)
    state_size = 16
    action_size = 4

    target_update_freq = 1000  # Update target network every 1000 steps
    total_steps = 0
    
    # Training parameters
    batch_size = 256
    learning_rate = 0.001
    memory_size = 50000
    gamma = 0.99  # Discount factor
    tau = 0.005  # Soft update parameter
    
    # Initialize agent with parameters
    agent = QRDQN(
        state_size=37,  # Increased for additional features
        action_size=4,
        memory_size=memory_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau
    )

    # Add prioritized experience replay
    agent.memory = PrioritizedReplayBuffer(memory_size)

    max_reward_ever = float('-inf')
    max_reward_episode = 0
    
    # Training loop parameters
    episodes = 500  # Increased episodes
    max_steps = 1000
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.991  # Slower decay
    
    # Tracking metrics
    episode_rewards = []
    max_tiles = []
    epsilon = epsilon_start
    try:
        for episode in tqdm(range(episodes)):
            state = env.reset()
            episode_reward = 0.0
            max_tile = 0
            
            for step in range(max_steps):

                state_2d = state.reshape(4, 4)
                processed_state = agent.preprocess_state(state_2d)

                # Use dynamic epsilon
                current_epsilon = agent.get_dynamic_epsilon(max_tile, epsilon)
                action = agent.select_action(processed_state, current_epsilon)
                
                next_state, reward, done, info = env.step(action)
                
                # Enhanced reward shaping
                auxiliary_reward = calculate_auxiliary_reward(state_2d, reward, max_tile)
                
                total_reward = reward + auxiliary_reward

                # Preprocess next state
                next_state_2d = next_state.reshape(4, 4)
                processed_next_state = agent.preprocess_state(next_state_2d)

                priority = abs(total_reward) + 1e-6  # Priority based on reward magnitude
                agent.memory.push(processed_state, action, total_reward, processed_next_state, done, priority)
                
                loss = agent.train_step()
                total_steps += 1

                # Update target network periodically
                if total_steps % target_update_freq == 0:
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())

                state = next_state
                episode_reward += total_reward
                max_tile = max(max_tile, np.max(state))
                if done:
                    break
            
            if episode_reward > max_reward_ever:
                max_reward_ever = episode_reward
                max_reward_episode = episode + 1
                # Save best model
                torch.save({
                    'model_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'episode': episode,
                    'reward': max_reward_ever,
                }, 'best_model.pth')
            
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            episode_rewards.append(float(episode_reward))
            max_tiles.append(float(max_tile))

            # Plot every 250 episodes instead of 500
            if (episode + 1) % 250 == 0:  # Changed from 500 to 250
                plot_training_results(episode_rewards, max_tiles, episode + 1)
                np.save(f'episode_rewards_{episode+1}.npy', episode_rewards)
                np.save(f'max_tiles_{episode+1}.npy', max_tiles)
                        
            # Plot every 500 episodes
            # Save more frequently
            if (episode + 1) % 100 == 0:  # Every 100 episodes
                torch.save({
                    'model_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'episode': episode,
                }, f'qrdqn_checkpoint_{episode+1}.pth')
            
            # Print stats every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_max_tile = np.mean(max_tiles[-100:])
                print(f"\nEpisode {episode + 1}")
                print(f"Average Reward: {avg_reward:.2f}")
                print(f"Average Max Tile: {avg_max_tile:.2f}")
                print(f"Epsilon: {epsilon:.3f}")
                print(f"Best Reward Ever: {max_reward_ever:.2f} (Episode {max_reward_episode})")
                print("------------------------")
                env.render()
        
                
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving final plot and data...")
    finally:
        # Always create final plot
        plot_training_results(episode_rewards, max_tiles, episode + 1)
        
        # Save final metrics
        np.save('final_episode_rewards.npy', episode_rewards)
        np.save('final_max_tiles.npy', max_tiles)
        
        # Save final model
        torch.save({
            'model_state_dict': agent.policy_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'episode': episode,
            'episode_rewards': episode_rewards,
            'max_tiles': max_tiles,
        }, 'final_model.pth')
        
        print("\nFinal plots and data saved!")

if __name__ == "__main__":
    train()