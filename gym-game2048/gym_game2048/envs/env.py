import gym
import numpy as np
import sys
from gym.utils import seeding
from gym import spaces
from .game_2048 import Game2048


class Game2048Env(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(
        self, board_size, invalid_move_warmup=16, invalid_move_threshold=0.1, penalty=-512, seed=None
    ):
        """
        Iniatialize the environment.

        Parameters
        ----------
        board_size : int
            Size of the board. Default=4
        invalid_move_warmup : int
            Minimum of invalid movements to finish the game. Default=16
        invalid_move_threshold : float
                    How much(fraction) invalid movements is necessary according to the total of moviments already executed. to finish the episode after invalid_move_warmup. Default 0.1
        penalty : int
            Penalization score of invalid movements to sum up in reward function. Default=-512
        seed :  int
            Seed
        """

        # State space flattens the 4x4 matrix in a 1D matrix
        self.state = np.zeros(board_size * board_size)
        # Action space
        self.action_space = spaces.Discrete(4)  # Up, down, right, left

        # Attributes
        if penalty > 0:
            raise ValueError("The value of penalty needs to be between [0, -inf)")
        self.__game = Game2048(board_size, invalid_move_warmup, invalid_move_threshold, penalty)
        self.__n_iter = 0
        self.__done = False
        self.__total_score = 0
        self.__board_size = board_size


    def step(self, action):
        """
        Execute an action.

        Parameters
        ----------
        action : int
            Action selected by the model.

        ----------
        Returns the next state, reward, termination status, and game info.
        """
        # Reward and info dict to store information about the the game state before and after a step
        reward = 0
        info = dict()

        # Execute a move
        before_move = self.__game.get_board().copy()
        self.__game.make_move(action)
        self.__game.confirm_move()
        self.state = self.normalized_values(self.__game.get_board())
        
        # Verify the game state and update reward and penalties
        self.__done, penalty = self.__game.verify_game_state()

        # reward = self.__game.get_move_score() + penalty
        # if self.__done:
        #     reward += self.__game.get_total_score()

        self.__n_iter = self.__n_iter + 1 
        after_move = self.__game.get_board()

        move_score = self.__game.get_move_score()
        reward = self.calculate_reward(move_score, self.__done)

        # Update info
        info["total_score"] = self.__game.get_total_score()
        info["steps_duration"] = self.__n_iter
        info["move"] = self.__game.moveTranslate(action)
        info["before_move"] = before_move
        info["after_move"] = after_move

        return (self.state, reward, self.__done, info)
    
    def reset(self):
        "Reset the environment."
        self.__n_iter = 0
        self.__done = False
        self.__total_score = 0
        self.__game.reset()
        self.state = self.__game.get_board()
        return self.state


    def calculate_reward(self, move_score, done):
        reward = 0

        # # Reward for score increment (normalize to [-1, 1] based on max score delta)
        # max_score_delta = 2048  # Adjust as needed
        # reward += np.clip((new_board.score - board.score) / max_score_delta, -1, 1)
        
        # Reward for tile merges (normalize to [-1, 1] based on max merge value)
        max_merge_value = 4096  # Adjust as needed
        reward += np.clip(0.1 * move_score/ max_merge_value, -1, 1)
        
        # Bonus for empty tiles (normalize based on max possible tiles)
        max_empty_tiles = 16  # For a 4x4 grid
        reward += np.clip(2 * self.__game.getNumEmptyTiles() / max_empty_tiles, -1, 1)
        
        # Penalty for lack of smoothness (assume max penalty is known)
        max_smoothness_penalty = 100  # Empirical or estimated value
        reward -= np.clip(self.__game.compute_smoothness_penalty() / max_smoothness_penalty, -1, 1)
        
        # Game over penalty
        if done:
            reward -= 1  # Already scaled appropriately

        return reward

    def render(self, mode="human"):
        """Render the current board state in a human-readable format."""
        # Obtain the board from the game
        board = self.__game.get_board()
        board_display = "\n".join(["\t".join(map(str, row)) for row in board])
        
        # Display the current board state
        print("Current Board State:\n")
        print(board_display)
        print("\n" + "-" * 20)

    def oneHotEncoding(self, board):
        """
        Converts the 2048 grid state into a one-hot encoding representation of the board with 13 channels. Each tile corresponds to a one-hot vector where its value x will be 1 in the index log_2 x.

        one_hot[0]: the empty tile index
        one_hot[1]: the dynamic tile index
        one_hot[2:13]: the power of 2 tiles, from 1(2^1=2) to 11(2^11=2048)
        Returns a numpy array of shape (4,4,13)
        """
        one_hot = np.zeros((4,4,13), dtype=np.float32)
        for row in range(4):
            for col in range(4):
                tile = board[row][col]
                if tile == 0:
                    one_hot[row][col][0] = 1
                elif tile == 1: # account for dynamic obstacle values
                    one_hot[row][col][0] = 1
                else:
                    # shift tile value by 2 because the first 2 indices are reserved for empty tile and dynamic tile. No, shift by 1 because you start at log 2 =1 (+1) = 2.
                    hot_index = int(np.log2(tile)) + 1
                    one_hot[row][col][hot_index] = 1
        return one_hot
    
    def normalized_values(self, board, max_value=2048):
        """
        Normalize the board values by dividing by the maximum value.
        """
        return board / max_value

    def get_board(self):
        return self.__game.get_board()

