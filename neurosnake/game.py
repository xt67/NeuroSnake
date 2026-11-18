"""Grid environment independent of the display and training framework."""
import random
import numpy as np

DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


class Snake:
    def __init__(self, size=12, seed=42):
        if size < 6:
            raise ValueError('Board size must be at least 6')
        self.size = size
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        c = self.size // 2
        self.body = [(c, c), (c-1, c), (c-2, c)]
        self.direction = 0
        self.score = self.idle = 0
        self.done = False
        self._food()
        return self.state()

    def _food(self):
        free = [(x, y) for x in range(self.size) for y in range(self.size)
                if (x, y) not in self.body]
        self.food = self.rng.choice(free) if free else None

    def next_cell(self, direction):
        x, y = self.body[0]
        dx, dy = DIRECTIONS[direction]
        return x+dx, y+dy

    def collision(self, cell):
        x, y = cell
        # Tail moves away unless this move eats food.
        occupied = self.body if cell == self.food else self.body[:-1]
        return not (0 <= x < self.size and 0 <= y < self.size) or cell in occupied

    def state(self):
        x, y = self.body[0]
        fx, fy = self.food if self.food is not None else (x, y)
        return np.array([
            *[self.collision(self.next_cell((self.direction+d) % 4)) for d in (0, 1, -1)],
            *[self.direction == d for d in (2, 0, 3, 1)],
            fx < x, fx > x, fy < y, fy > y,
        ], dtype=np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError('Reset after an episode ends')
        if action not in (0, 1, 2):
            raise ValueError('Actions: 0 straight, 1 right, 2 left')
        self.direction = (self.direction + (0, 1, -1)[action]) % 4
        cell = self.next_cell(self.direction)
        self.idle += 1
        if self.collision(cell):
            self.done = True
            return self.state(), -10., True
        self.body.insert(0, cell)
        reward = 0.
        if cell == self.food:
            self.score += 1
            self.idle = 0
            reward = 10.
            self._food()
            self.done = self.food is None
        else:
            self.body.pop()
        if self.idle >= 100 * len(self.body):
            self.done = True
            reward = -10.
        return self.state(), reward, self.done
