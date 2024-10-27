from gym_game2048.envs.env import Game2048Env
from gym_game2048.envs.game_2048 import Game2048
import numpy as np

def set_board(env, board):
    env.board = board

def generate_random_board():
    board = np.zeros((4, 4), dtype=np.int32)
    for _ in range(2):
        row = np.random.randint(0, 4)
        col = np.random.randint(0, 4)
        value = np.random.choice([2, 4])
        board[row, col] = value
    return board

def main():
    env = Game2048Env(board_size=4)
    init_state = env.reset()
    done = False
    print("Initial state:")
    print(init_state.reshape(4, 4))
    for i in range(100):
        if done:
            print("Game over!")
            break
        move = env.action_space.sample()
        next_state, reward, done, info = env.step(move)

        print(f"Step: {i+1}")
        print(f"Move: {info['move']}")
        print(f"Reward: {reward}")
        print(f"Total score: {info['total_score']}")
        print(next_state)
        print("-----------------")
        

if __name__ == "__main__":
    main()