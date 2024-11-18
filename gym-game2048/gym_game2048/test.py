from envs.env import Game2048Env
import numpy as np

# Helper function to check if the game is over
def is_game_over(board):
    if np.any(board == 0):
        return False, "There are still empty tiles on the board."

    size = board.shape[0]
    for i in range(size):
        for j in range(size):
            if j < size - 1 and board[i][j] == board[i][j + 1]:
                return False, "There are mergeable tiles horizontally."
            if i < size - 1 and board[i][j] == board[i + 1][j]:
                return False, "There are mergeable tiles vertically."

    return True, "No empty tiles or mergeable tiles."


def test_final_board_state():
    env = Game2048Env(board_size=4)
    init_state = env.reset()
    done = False
    final_state = None
    last_step = 0

    for i in range(500):  # Limit to 500 moves
        if done:
            last_step = i
            break
        move = env.action_space.sample()
        next_state, reward, done, info = env.step(move)

        if done:
            final_state = next_state.reshape(4, 4)

    if final_state is None:
        print("Test failed: Game did not finish within 500 steps!\n")
        return False, last_step, None, "Game did not finish within the step limit."

    print("Final board state:")
    print("Last step:", last_step)
    print(final_state)

    game_over, reason = is_game_over(final_state)
    if game_over:
        print("Test passed: Final state is valid and game is over.")
        print("-----------------")
        return True, last_step, final_state, reason
    else:
        print(f"Test failed: Final state is invalid. Reason: {reason}")
        print("-----------------")
        return False, last_step, final_state, reason


if __name__ == "__main__":
    pass_count = 0
    fail_count = 0
    total_test = 50

    for test_num in range(1, total_test + 1):
        print(f"Running test #{test_num}...")
        result, last_step, final_state, reason = test_final_board_state()
        if result:
            print(f"Test #{test_num} passed.\n")
            pass_count += 1
        else:
            print(f"Test #{test_num} failed.")
            print(f"Failure reason: {reason}\n")
            fail_count += 1

    print("Test Summary:")
    print(f"Total tests run: {total_test}")
    print(f"Passed: {pass_count}")
    print(f"Failed: {fail_count}")
