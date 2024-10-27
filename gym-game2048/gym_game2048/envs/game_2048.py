import numpy as np


class Game2048:
    def __init__(self, board_size: int, invalid_move_warmup=16, invalid_move_threshold=0.1, penalty=-512, dynamic_obstacle_interval=5, dynamic_obstacle_value=1):
        """
        Initialize the 2048 game with dynamic obstacles.

        Parameters
        ----------
        board_size : int
            Size of the board. Default=4
        invalid_move_warmup : int
            Minimum of invalid movements to finish the episode. Default=16
        invalid_move_threshold : float
            Fraction of invalid movements necessary to finish the episode after invalid_move_warmup. Default=0.1 
        penalty : int
            Penalization of invalid movements. Default=-512
        dynamic_obstacle_interval : int
            Number of steps between spawning dynamic obstacles. Default=5
        """
        self.__board_size = board_size
        self.__score = 0
        self.__total_score = 0
        self.__invalid_count = 0
        self.__total_count = 0
        self.__invalid_move_warmup = invalid_move_warmup
        self.__invalid_move_threshold = invalid_move_threshold
        self.__penalty = penalty
        self.__dynamic_obstacle_interval = dynamic_obstacle_interval
        self.__num_obstacles = 0
        self.__dynamic_obstacle_value = dynamic_obstacle_value
        self.__board = np.zeros((board_size, board_size), dtype=np.int32)  # Allow obstacles
        self.__temp_board = np.zeros((board_size, board_size), dtype=np.int32)
        self.__obstacle_reward = 4 # small reward for colliding obstacles 
        self.__add_two_or_four()
        self.__add_two_or_four()
        self.__add_dynamic_obstacle()

    def __add_two_or_four(self):
        """Add tile with number two."""

        indexes = np.where(self.__board == 0)

        if len(indexes[0]) == 0:
            return

        # Coordinates to add a tile with number two
        index = np.random.choice(np.arange(len(indexes[0])))

        if np.random.uniform(0, 1) >= 0.9:
            self.__board[indexes[0][index]][indexes[1][index]] = 4
        else:
            self.__board[indexes[0][index]][indexes[1][index]] = 2

    def __add_dynamic_obstacle(self):
        """Add a dynamic obstacle (-1) to a random empty position."""
        indexes = np.where(self.__board == 0)
        if len(indexes[0]) == 0:
            return
        index = np.random.choice(np.arange(len(indexes[0])))
        self.__board[indexes[0][index]][indexes[1][index]] = 1
        self.__num_obstacles+=1

    def __transpose(self, board):
        """Transpose a matrix."""

        temp = np.zeros((self.__board_size, self.__board_size), dtype=np.uint32)

        for line in range(self.__board_size):
            for column in range(self.__board_size):
                temp[column][line] = board[line][column]

        return temp

    def __reverse(self, board):
        """Reverse a matrix."""

        temp = np.zeros((self.__board_size, self.__board_size), dtype=np.uint32)

        for line in range(self.__board_size):
            for column in range(self.__board_size):
                temp[line][column] = board[self.__board_size - line - 1][column]

        return temp

    def __cover_up(self, board):
        """Cover the most antecedent zeros with non-zero number. """

        temp = np.zeros((self.__board_size, self.__board_size), dtype=np.uint32)
        self.__done_cover_up = False

        for column in range(self.__board_size):
            up = 0
            for line in range(self.__board_size):
                if board[line][column] != 0:
                    temp[up][column] = board[line][column]
                    up = up + 1
                    if up != line:
                        self.__done_cover_up = True

        return temp

    def __merge(self, board):
        """Verify if a merge is possible and execute."""

        self.__done_merge = False

        for line in range(1, self.__board_size):
            for column in range(self.__board_size):
                if board[line][column] == board[line - 1][column]:
                    if (board[line][column] == self.__dynamic_obstacle_value):
                        self.__score = self.__score + (board[line][column] * 2)
                        self.__score += self.__obstacle_reward * self.__num_obstacles
                        board[line - 1][column] = 0
                        board[line][column] = 0
                        self.__num_obstacles -= 2
                    else:
                        self.__score = self.__score + (board[line][column] * 2)
                        board[line - 1][column] = board[line - 1][column] * 2
                        board[line][column] = 0
                    self.__done_merge = True
                else:
                    continue

        return board

    def __up(self):

        temp = self.__cover_up(self.__board)
        temp = self.__merge(temp)
        temp = self.__cover_up(temp)
        self.__temp_board = temp

    def __down(self):

        temp = self.__reverse(self.__board)
        temp = self.__merge(temp)
        temp = self.__cover_up(temp)
        temp = self.__reverse(temp)
        self.__temp_board = temp

    def __right(self):

        temp = self.__reverse(self.__transpose(self.__board))
        temp = self.__merge(temp)
        temp = self.__cover_up(temp)
        temp = self.__transpose(self.__reverse(temp))
        self.__temp_board = temp

    def __left(self):

        temp = self.__transpose(self.__board)
        temp = self.__merge(temp)
        temp = self.__cover_up(temp)
        temp = self.__transpose(temp)
        self.__temp_board = temp

    def get_move_score(self):
        """Get the last score move."""

        return self.__score

    def get_total_score(self):
        """Get the total score gained until now."""

        return self.__total_score

    def set_board(self, board):
        """This function is only for test purpose."""

        self.__board = board

    def get_board(self):
        """Get the actual board."""

        return self.__board

    def confirm_move(self):
        """Finalize the move and add dynamic obstacles periodically."""
        self.__total_count += 1
        self.__total_score += self.__score

        if np.array_equal(self.__board, self.__temp_board):
            self.__invalid_count += 1
            self.__score = self.__penalty
        else:
            self.__board = self.__temp_board.copy()
            self.__add_two_or_four()

            # Add dynamic obstacles every few steps
            if self.__total_count % self.__dynamic_obstacle_interval == 0:
                self.__add_dynamic_obstacle()

    def make_move(self, move):
        """Make a move."""
        self.__score = 0

        if move == 0:
            self.__up()
        if move == 1:
            self.__down()
        if move == 2:
            self.__right()
        if move == 3:
            self.__left()

    def verify_game_state(self):
        "Check if the game has done or not."
        if (
            self.__invalid_count > self.__invalid_move_warmup
            and self.__invalid_count > self.__invalid_move_threshold * self.__total_count
        ):
            return True, self.__penalty


        has_zero_entries = np.any(self.__board == 0)
        if has_zero_entries:
            return False, 0
        
        # Verify possible merges
        for line in range(1, self.__board_size):
            for column in range(1, self.__board_size):
                if (
                    self.__board[line][column] == self.__board[line][column - 1]
                    or self.__board[line][column] == self.__board[line - 1][column]
                ):
                    return False, 0

        # Verify possible merges in first column and first line
        for line in range(1, self.__board_size):
            if self.__board[line][0] == self.__board[line - 1][0]:
                return False, 0

        for column in range(1, self.__board_size):
            if self.__board[0][column] == self.__board[0][column - 1]:
                return False, 0

        return True, self.__penalty

    def reset(self):
        "Reset the game."
        self.__board = np.zeros((self.__board_size, self.__board_size), dtype=np.uint32)
        self.__temp_board = np.zeros((self.__board_size, self.__board_size), dtype=np.uint32)
        self.__score = 0
        self.__total_score = 0
        self.__invalid_count = 0
        self.__total_count = 0
        self.__num_obstacles = 0
        self.__add_two_or_four()
        self.__add_two_or_four()

    def moveTranslate(self,move):
        '''
        Translate the move to a human readable format.
        '''
        if move == 0:
            return("Up ↑")
        if move == 1:
            return("Down ↓")
        if move == 2:
            return("Right →")
        if move == 3:
            return("Left ←")
