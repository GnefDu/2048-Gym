import random
import numpy as np

class Game2048:
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.add_new_tile()
        self.add_new_tile()
        self.total_count = 0

    def add_dynamic_obstacle(self):
        """Add a dynamic obstacle (-1) to a random empty position."""
        empty_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if self.grid[r][c] == 0]
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.grid[r][c] = 1

    def add_new_tile(self):
        empty_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if self.grid[r][c] == 0]
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.grid[r][c] = 2 if random.random() < 0.9 else 4

    def slide_and_merge(self, row):
        non_zero = [num for num in row if num != 0]
        merged = []
        skip = False
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged.append(non_zero[i] * 2)
                skip = True
            else:
                merged.append(non_zero[i])
        return merged + [0] * (len(row) - len(merged))

    def move(self, direction):
        rotated = False
        moved = False
        self.total_count += 1
        if direction in ['up', 'down']:
            self.grid = self.grid.T
            rotated = True

        for i in range(self.grid_size):
            original = self.grid[i].copy()
            if direction in ['right', 'down']:
                self.grid[i] = self.slide_and_merge(self.grid[i][::-1])[::-1]
            else:
                self.grid[i] = self.slide_and_merge(self.grid[i])
            if not np.array_equal(original, self.grid[i]):
                moved = True

        if rotated:
            self.grid = self.grid.T

        if moved:
            if self.total_count % 10 == 0 and self.getNumObstacles() == 0:
                self.add_dynamic_obstacle()
            elif self.total_count % 2 == 0 and self.getNumObstacles() == 1:
                self.add_dynamic_obstacle()
            else:
                self.add_new_tile()

        return moved

    def is_game_over(self):
        if any(0 in row for row in self.grid):
            return False
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r + 1 < self.grid_size and self.grid[r][c] == self.grid[r + 1][c]) or \
                   (c + 1 < self.grid_size and self.grid[r][c] == self.grid[r][c + 1]):
                    return False
        return True

    def score(self):
        return self.grid.sum()
    
    def getNumObstacles(self):
        return np.sum(self.grid == 1)


def monte_carlo_2048(game, num_simulations=100):
    directions = ['up', 'down', 'left', 'right']
    best_direction = None
    best_score = -1

    for direction in directions:
        total_score = 0

        for _ in range(num_simulations):
            simulation_game = Game2048(grid_size=game.grid_size)
            simulation_game.grid = game.grid.copy()

            if not simulation_game.move(direction):
                continue

            while not simulation_game.is_game_over():
                simulation_game.move(random.choice(directions))

            total_score += simulation_game.score()

        average_score = total_score / num_simulations
        if average_score > best_score:
            best_score = average_score
            best_direction = direction

    return best_direction

# Example usage
game = Game2048()
while not game.is_game_over():
    print("Current Grid:")
    print(game.grid)
    best_move = monte_carlo_2048(game, num_simulations=50)
    if best_move:
        print(f"Best move: {best_move}")
        game.move(best_move)
    else:
        break

print("Game Over!")
print("Final Grid:")
print(game.grid)