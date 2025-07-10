import torch
import pygame
import random
import numpy as np
import torch.nn as nn

# Configurações do jogo
BLOCK_SIZE = 20
ROWS, COLS = 10, 10  # 10x10 blocos
WIDTH, HEIGHT = COLS * BLOCK_SIZE, ROWS * BLOCK_SIZE

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(5, 5)]
        self.direction = RIGHT
        self.spawn_food()
        self.prev_distance = self.get_distance()
        return self.get_state()

    def spawn_food(self):
        while True:
            self.food = (random.randint(0, COLS - 1), random.randint(0, ROWS - 1))
            if self.food not in self.snake:
                break

    def get_distance(self):
        head = self.snake[0]
        return abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

    def step(self, action):
        self.direction = (self.direction + action - 1) % 4
        x, y = self.snake[0]
        if self.direction == UP:
            y -= 1
        elif self.direction == DOWN:
            y += 1
        elif self.direction == LEFT:
            x -= 1
        elif self.direction == RIGHT:
            x += 1

        new_head = (x, y)
        self.snake.insert(0, new_head)

        done = False
        reward = 0

        if (x < 0 or x >= COLS or y < 0 or y >= ROWS or new_head in self.snake[1:]):
            done = True
            reward = -10
        elif new_head == self.food:
            reward = 10
            self.spawn_food()
            self.prev_distance = self.get_distance()
        else:
            self.snake.pop()
            new_distance = self.get_distance()
            # Recompensa se aproximou, penaliza se afastou
            if new_distance < self.prev_distance:
                reward = 1
            else:
                reward = -1
            self.prev_distance = new_distance

        return self.get_state(), reward, done

    def get_state(self):
        # Vetor para o grid todo, codificando:
        # 0 = vazio, 1 = cobra, 2 = comida
        grid_state = np.zeros((ROWS, COLS), dtype=np.float32)

        for (x, y) in self.snake:
            if 0 <= x < COLS and 0 <= y < ROWS:
                grid_state[y][x] = 1.0  # cobra

        fx, fy = self.food
        grid_state[fy][fx] = 2.0  # comida

        # Flatten grid em vetor 1D
        grid_flat = grid_state.flatten()

        # Vetor direção one-hot
        dir_vec = np.array([
            int(self.direction == UP),
            int(self.direction == RIGHT),
            int(self.direction == DOWN),
            int(self.direction == LEFT)
        ], dtype=np.float32)

        # Combina vetor plano do grid + vetor direção
        return np.concatenate((grid_flat, dir_vec))

    def move_point(self, point, direction):
        x, y = point
        if direction == UP: y -= 1
        elif direction == DOWN: y += 1
        elif direction == LEFT: x -= 1
        elif direction == RIGHT: x += 1
        return (x, y)

    def is_collision(self, point):
        x, y = point
        if x < 0 or x >= COLS or y < 0 or y >= ROWS:
            return 1
        if point in self.snake[1:]:
            return 1
        return 0

class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)

def draw_game(screen, game):
    screen.fill((0, 0, 0))
    food_x, food_y = game.food
    pygame.draw.rect(screen, (255, 0, 0), (food_x * BLOCK_SIZE, food_y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    for block in game.snake:
        pygame.draw.rect(screen, (0, 255, 0), (block[0] * BLOCK_SIZE, block[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    pygame.display.flip()

def play():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake RL - Play Mode")
    clock = pygame.time.Clock()

    game = SnakeGame()
    model = LinearQNet(104, 128, 3)
    model.load_state_dict(torch.load("snake_agent_weights.pt"))
    model.eval()

    state = game.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                pygame.quit()
                return

        state_t = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(state_t)
        action = torch.argmax(prediction).item()

        next_state, reward, done = game.step(action)
        state = next_state

        draw_game(screen, game)
        clock.tick(10)

    print("Fim do jogo! Pontuação:", len(game.snake) - 1)
    pygame.quit()

if __name__ == "__main__":
    play()
