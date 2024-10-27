import numpy as np
from gym_game2048.envs.game_2048 import Game2048


def show_obstacle_collision():
    
    '''
    Simulates a move in a 2048 game with a predefined board made up of several obstacles and shows obstacle collision.
    '''
    game = Game2048(board_size=4)
    board = np.array([[2, 2, 0, 8],
                      [1, 1, 0, 0],
                      [1, 0, 0, 2],
                      [0, 0, 4, 0]])
    game.set_board(board)
    print("Initial board:")
    print(game.get_board())
    print("-----------------")
    game.make_move(0)
    print(f"Move: Up")
    game.confirm_move()
    print(game.get_board())

    return board


if __name__ == "__main__":
    show_obstacle_collision()