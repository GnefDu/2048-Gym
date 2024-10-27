import gym
# import gym_game2048
# from envs.env import Game2048Env
from gym_game2048.envs.env import Game2048Env
# import game_2048
import numpy as np

def set_board(env, board):
    env.board = board

def play_to_64(env):
    done = False
    while not done:
        action = env.action_space.sample()  # Random action, replace with a better strategy
        # yeah it should not be a random action
        _, reward, done, info = env.step(action)
        if any(64 in row for row in env.board):
            print("Reached 64!")
            break


def generate_random_board():
    board = np.zeros((4, 4), dtype=np.int32)
    for _ in range(2):
        row = np.random.randint(0, 4)
        col = np.random.randint(0, 4)
        value = np.random.choice([2, 4])
        board[row, col] = value
    return board


def main():
    env = Game2048Env(board_size=4, binary=False, extractor="cnn")
    env.reset()
    done = False
    for _ in range(20):
        if done:
            break
        move = env.action_space.sample()
        # move = np.random.randint(0, 4)
        next_state, reward, done, info = env.step(move)

        # game.moveTranslate(move)
        # game.make_move(move)
        # game.confirm_move()
        print(f"Reward: {reward}")
        print(f"Total score: {info['total_score']}")
        print(next_state)
        

if __name__ == "__main__":
    main()