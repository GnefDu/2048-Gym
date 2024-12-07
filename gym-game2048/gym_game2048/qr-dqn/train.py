import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import gym
import gym_game2048
from gym_game2048.envs.env import Game2048Env
from qr_dqn import QRDQN

def train():
    # Initialize environment and agent
    env = Game2048Env(board_size=4, penalty=-512)  # Original penalty value
    state_size = 16
    action_size = 4
    
    # Training parameters
    batch_size = 128  # Original smaller batch size
    learning_rate = 0.00025
    memory_size = 100000  # Original smaller memory size
    
    # Initialize agent with parameters
    agent = QRDQN(
        state_size=state_size,
        action_size=action_size,
        memory_size=memory_size,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Training loop parameters
    episodes = 10000
    max_steps = 2000  # Original shorter max steps
    epsilon_start = 1.0
    epsilon_end = 0.01  # Original lower epsilon end
    epsilon_decay = 0.995  # Original faster decay
    
    # Tracking metrics
    episode_rewards = []
    max_tiles = []
    epsilon = epsilon_start
    
    for episode in tqdm(range(episodes)):
        state = env.reset()
        episode_reward = 0.0
        max_tile = 0
        
        for step in range(max_steps):
            flat_state = state.flatten()
            action = agent.select_action(flat_state, epsilon)
            next_state, reward, done, info = env.step(action)
            
            agent.memory.push(flat_state, action, reward, next_state.flatten(), done)
            loss = agent.train_step()
            
            state = next_state
            episode_reward += reward
            max_tile = max(max_tile, np.max(state))
            
            if done:
                break
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(float(episode_reward))
        max_tiles.append(float(max_tile))
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_max_tile = np.mean(max_tiles[-100:])
            print(f"\nEpisode {episode + 1}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Max Tile: {avg_max_tile:.2f}")
            print(f"Epsilon: {epsilon:.3f}")
            print("------------------------")
            env.render()
        
        # Save model periodically
        if (episode + 1) % 1000 == 0:
            torch.save({
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }, f'qrdqn_checkpoint_{episode+1}.pth')
    
    # Plot training results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(max_tiles)
    plt.title('Max Tiles Achieved')
    plt.xlabel('Episode')
    plt.ylabel('Max Tile Value')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

if __name__ == "__main__":
    train()