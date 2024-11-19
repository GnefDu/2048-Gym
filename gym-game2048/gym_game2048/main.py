import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_game2048.envs.env import Game2048Env



def main():
    for i in range(20):
        env = Game2048Env(board_size=4)
        init_state = env.reset()
        done = False
        print("Initial state:")
        print(init_state.reshape(4, 4))
        print("-----------------")
        total_reward = 0
        for i in range(200):
            if done:
                print("Game over!")
                print(f"Step: {i}")
                print(f"Move: {info['move']}")
                print(f"Reward: {reward}")
                print(f"Total Reward: {total_reward}")
                print(f"Total score: {info['total_score']}")
                print(next_state)
                print("-----------------")
                break
            move = env.action_space.sample()
            next_state, reward, done, info = env.step(move)
            total_reward += reward

        # print(f"Step: {i}")
        # print(f"Move: {info['move']}")
        # print(f"Reward: {reward}")
        # print(f"Total Reward: {total_reward}")
        # print(f"Total score: {info['total_score']}")
        # print(next_state)
        # print("-----------------")
        

if __name__ == "__main__":
    main()