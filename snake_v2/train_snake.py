import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import pygame
import matplotlib.pyplot as plt

BLOCK_SIZE = 20
ROWS, COLS = 10, 10
WIDTH, HEIGHT = COLS * BLOCK_SIZE, ROWS * BLOCK_SIZE

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

pygame.init()
font = pygame.font.SysFont('Arial', 25)

class SnakeGame:
    def __init__(self):
        self.hamiltonian_cycle = self.generate_hamiltonian_cycle()
        self.reset()

    def generate_hamiltonian_cycle(self):
        cycle = []
        for y in range(ROWS):
            row = range(COLS) if y % 2 == 0 else reversed(range(COLS))
            for x in row:
                cycle.append((x, y))
        return cycle

    def reset(self):
        self.snake = [(0, 0)]
        self.direction = RIGHT
        self.spawn_food()
        self.frame = 0
        return self.get_state()

    def spawn_food(self):
        while True:
            self.food = (random.randint(0, COLS - 1), random.randint(0, ROWS - 1))
            if self.food not in self.snake:
                break

    def step(self, action):
        self.direction = (self.direction + action - 1) % 4
        head = self.move_point(self.snake[0], self.direction)
        self.snake.insert(0, head)

        x, y = head
        done, reward = False, -0.1

        if x < 0 or x >= COLS or y < 0 or y >= ROWS or head in self.snake[1:]:
            done = True
            reward = -100
            return self.get_state(), reward, done

        if head == self.food:
            reward = 50
            self.spawn_food()
        else:
            self.snake.pop()

        try:
            curr_idx = self.hamiltonian_cycle.index(self.snake[1])
            expected_next = self.hamiltonian_cycle[(curr_idx + 1) % len(self.hamiltonian_cycle)]
            if head == expected_next:
                reward += 5
            else:
                reward -= 5
        except:
            reward -= 5  # penaliza caso o índice falhe (início do jogo)

        return self.get_state(), reward, done

    def move_point(self, point, direction):
        x, y = point
        if direction == UP: y -= 1
        elif direction == DOWN: y += 1
        elif direction == LEFT: x -= 1
        elif direction == RIGHT: x += 1
        return (x, y)

    def get_state(self):
        grid = np.zeros((ROWS, COLS), dtype=np.float32)
        for x, y in self.snake:
            if 0 <= x < COLS and 0 <= y < ROWS:
                grid[y][x] = 1.0
        fx, fy = self.food
        grid[fy][fx] = 2.0
        grid_flat = grid.flatten()
        dir_vec = np.array([
            int(self.direction == UP),
            int(self.direction == RIGHT),
            int(self.direction == DOWN),
            int(self.direction == LEFT)
        ], dtype=np.float32)
        return np.concatenate((grid_flat, dir_vec))


class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class DQNAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.9997
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=100_000)
        self.batch_size = 1000
        self.model = LinearQNet(104, 128, 3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state_t)
            return torch.argmax(q_values).item()

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states)
        next_states = torch.tensor(next_states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        q_values = self.model(states)
        next_q_values = self.model(next_states)

        target_q = q_values.clone()
        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * torch.max(next_q_values[i]).item()
            target_q[i][actions[i]] = target

        loss = self.loss_fn(q_values, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def draw_game(screen, game):
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (255, 0, 0), (game.food[0] * BLOCK_SIZE, game.food[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
    for block in game.snake:
        pygame.draw.rect(screen, (0, 255, 0), (block[0] * BLOCK_SIZE, block[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
    pygame.display.flip()


def play():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake RL - Play Mode")
    clock = pygame.time.Clock()
    game = SnakeGame()
    agent = DQNAgent()
    agent.model.load_state_dict(torch.load("snake_agent_weights.pt"))
    agent.epsilon = 0
    state = game.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        action = agent.act(state)
        state, _, done = game.step(action)
        draw_game(screen, game)
        clock.tick(10)
    print("Pontuação:", len(game.snake) - 1)
    pygame.quit()


def train():
    game = SnakeGame()
    agent = DQNAgent()
    scores, avg_scores, episodes_avg = [], [], []
    max_apples = 0

    for episode in range(1, 20001):
        state = game.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
        agent.train_long_memory()

        apples = len(game.snake) - 1
        scores.append(apples)

        if apples > max_apples:
            max_apples = apples
            torch.save(agent.model.state_dict(), "snake_agent_weights.pt")
            print(f"Novo recorde: {apples} maçãs! Modelo salvo.")

        if episode % 100 == 0:
            avg = sum(scores[-100:]) / 100
            avg_scores.append(avg)
            episodes_avg.append(episode)
            print(f"Eps {episode-99}-{episode}: Média {avg:.2f} | Épsilon {agent.epsilon:.4f}")

    plt.plot(episodes_avg, avg_scores, marker='o')
    plt.xlabel("Episódio")
    plt.ylabel("Média de maçãs (100 episódios)")
    plt.title("DQN Snake - Desempenho")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    train()
    play()
