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
        self.state = self.__game.get_board()
        
        # Verify the game state and update reward and penalties
        self.__done, penalty = self.__game.verify_game_state()
        reward = self.__game.get_move_score() + penalty
        self.__n_iter = self.__n_iter + 1 
        after_move = self.__game.get_board()

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

    def render(self, mode="human"):
        """Render the current board state in a human-readable format."""
        # Obtain the board from the game
        board = self.__game.get_board()
        board_display = "\n".join(["\t".join(map(str, row)) for row in board])
        
        # Display the current board state
        print("Current Board State:\n")
        print(board_display)
        print("\n" + "-" * 20)

    def get_board(self):
        "Get the board."

        return self.__game.get_board()

