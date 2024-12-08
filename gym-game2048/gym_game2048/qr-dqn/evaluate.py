import os
import sys
import numpy as np
import torch
from tqdm import tqdm  # For progress bar
from gym_game2048.envs.env import Game2048Env
from qr_dqn import QRDQN

def quick_evaluate(model_path, num_episodes):
    """Enhanced evaluation function with robust error handling"""
    env = Game2048Env(board_size=4, penalty=-32)
    agent = QRDQN(
        state_size=37,
        action_size=4,
        memory_size=50000,
        batch_size=256,
        learning_rate=0.001
    )

    max_score_ever = float('-inf')
    max_score_episode = 0
    
    # Load model safely
    try:
        checkpoint = torch.load(model_path, weights_only=True)
        agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        agent.policy_net.eval()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None
    
    # Track results
    from collections import OrderedDict
    tile_achievements = OrderedDict([
        (32, 0), (64, 0), (128, 0), (256, 0),
        (512, 0), (1024, 0), (2048, 0)
    ])
    
    all_max_tiles = []
    all_scores = []
    valid_moves = []
    completed_episodes = 0
    
    try:
        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            state = env.reset()
            done = False
            max_tile = 0
            episode_score = 0  # Total score including rewards AND penalties
            valid_move_count = 0
            total_moves = 0
            step_count = 0
            max_steps = 1000
            
            try:
                while not done and step_count < max_steps:
                    state_2d = state.reshape(4, 4)
                    processed_state = agent.preprocess_state(state_2d)
                    action = agent.select_action(processed_state, epsilon=0)
                    next_state, reward, done, info = env.step(action)
                    
                    # Detailed reward tracking
                    merge_score = info.get('merge_score', 0)  # Score from merging tiles
                    move_valid = reward != -32
                    
                    if move_valid:
                        valid_move_count += 1
                        episode_score += merge_score  # Only count merge scores
                    
                    # Debug printing
                    if step_count % 50 == 0:
                        print(f"\nStep {step_count}:")
                        print(f"Action: {action}")
                        print(f"Raw Reward: {reward}")
                        print(f"Merge Score: {merge_score}")
                        print(f"Valid: {move_valid}")
                        print(f"Current Board:")
                        print(state_2d)
                    
                    total_moves += 1
                    state = next_state
                    max_tile = max(max_tile, np.max(state))
                    step_count += 1
                
                if episode_score > max_score_ever:
                    max_score_ever = episode_score
                    max_score_episode = episode + 1
                
                # Record episode results
                all_max_tiles.append(max_tile)
                all_scores.append(episode_score)
                valid_moves.append(valid_move_count / total_moves if total_moves > 0 else 0)
                
                for tile in tile_achievements.keys():
                    if max_tile >= tile:
                        tile_achievements[tile] += 1
                
                completed_episodes += 1
                
                # Print progress every 50 episodes
                if (episode + 1) % 50 == 0:
                    print(f"\nIntermediate Results (Episode {episode + 1}):")
                    print(f"Current Max Tile: {max_tile}")
                    print(f"Current Score: {episode_score:.2f}")
                    print(f"Best Score So Far: {max_score_ever:.2f} (Episode {max_score_episode})")
                    print(f"Valid Move %: {(valid_move_count/total_moves)*100:.2f}%")
                    print("-" * 40)
            
            except Exception as e:
                print(f"\nError in episode {episode}: {str(e)}")
                continue  # Skip to next episode on error
        
        # Print final results
        print("\n" + "="*50)
        print(f"Evaluation completed for {completed_episodes} episodes")
        print("="*50)
        
        if completed_episodes > 0:
            print("\nTile Statistics:")
            print("-"*50)
            print(f"Average Max Tile: {np.mean(all_max_tiles):.2f}")
            print(f"Best Max Tile: {np.max(all_max_tiles)}")
            
            print("\nScore Statistics:")
            print("-"*50)
            print(f"Average Score: {np.mean(all_scores):.2f}")
            print(f"Max Score: {max_score_ever:.2f} (Episode {max_score_episode})")
            print(f"Average Valid Moves: {np.mean(valid_moves)*100:.2f}%")
            
            print("\nTile Achievement Statistics:")
            print("-"*50)
            for tile, count in tile_achievements.items():
                percentage = (count / completed_episodes) * 100
                print(f"Reached {tile:4d}: {count:4d} times ({percentage:6.2f}%)")
        
        return all_max_tiles, all_scores, tile_achievements, valid_moves
    
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user!")
        if completed_episodes > 0:
            print(f"Completed {completed_episodes} episodes before interruption")
            # Print partial results
            return all_max_tiles, all_scores, tile_achievements, valid_moves
        return None
    
    except Exception as e:
        print(f"\nUnexpected error during evaluation: {str(e)}")
        return None

def find_checkpoint():
    """Find available checkpoint files"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))   # Can be modified
    
    checkpoint_files = [
        'qrdqn_checkpoint_100.pth',
        'qrdqn_checkpoint_200.pth',
        'qrdqn_checkpoint_300.pth',
        'qrdqn_checkpoint_400.pth',
        'qrdqn_checkpoint_500.pth',
        'final_model.pth'
    ]
    
    # First check project root
    for file in checkpoint_files:
        full_path = os.path.join(project_root, file)
        if os.path.exists(full_path):
            print(f"Found checkpoint in project root: {full_path}")
            return full_path
            
    print(f"\nNo checkpoint files found in: {project_root}")
    return None

if __name__ == "__main__":
    checkpoint_path = find_checkpoint()
    if checkpoint_path:
        results = quick_evaluate(checkpoint_path, num_episodes=500)
    else:
        print("\nPlease ensure one of these files exists:")
        print("- qrdqn_checkpoint_1000.pth")
        print("- qrdqn_checkpoint_1500.pth")
        print("- final_model.pth")