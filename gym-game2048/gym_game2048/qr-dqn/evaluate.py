import numpy as np
import torch
from gym_game2048.envs.env import Game2048Env
from qr_dqn import QRDQN

def quick_evaluate(model_path, num_episodes=3):
    """Simplified evaluation function"""
    # Initialize environment and agent
    env = Game2048Env(board_size=4)
    state_size = 16
    action_size = 4
    agent = QRDQN(state_size, action_size)
    
    # Load model
    checkpoint = torch.load(model_path, weights_only=True)
    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
    agent.policy_net.eval()
    
    # Track results
    all_max_tiles = []
    all_scores = []
    
    print("Starting evaluation...")
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_score = 0
        max_tile = 0
        steps = 0
        
        while not done and steps < 1000:  # Add step limit
            flat_state = state.flatten()
            action = agent.select_action(flat_state, epsilon=0)
            state, reward, done, info = env.step(action)
            episode_score += reward
            max_tile = max(max_tile, np.max(state))
            steps += 1
            
            # Print current board every 100 steps
            if steps % 100 == 0:
                print(f"\nEpisode {episode + 1}, Step {steps}")
                env.render()
        
        print(f"\nEpisode {episode + 1} finished:")
        print(f"Max Tile: {max_tile}")
        print(f"Score: {episode_score}")
        print("-" * 30)
        
        all_max_tiles.append(max_tile)
        all_scores.append(episode_score)
    
    # Print final results
    print("\nEvaluation Results:")
    print(f"Average Max Tile: {np.mean(all_max_tiles):.2f}")
    print(f"Best Max Tile: {np.max(all_max_tiles)}")
    print(f"Average Score: {np.mean(all_scores):.2f}")

if __name__ == "__main__":
    quick_evaluate("qrdqn_checkpoint_1000.pth", num_episodes=3)