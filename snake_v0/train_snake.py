import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import pygame
import matplotlib.pyplot as plt

# Configurações do jogo
BLOCK_SIZE = 20
ROWS, COLS = 10, 10  # 10x10 blocos
WIDTH, HEIGHT = COLS * BLOCK_SIZE, ROWS * BLOCK_SIZE

# Direções
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Inicializar pygame
pygame.init()
font = pygame.font.SysFont('Arial', 25)

# Ambiente Snake
class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(5, 5)]
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
        x, y = self.snake[0]
        if self.direction == UP: y -= 1
        elif self.direction == DOWN: y += 1
        elif self.direction == LEFT: x -= 1
        elif self.direction == RIGHT: x += 1

        new_head = (x, y)
        self.snake.insert(0, new_head)

        done = False
        reward = 0

        # Checar colisão
        if (x < 0 or x >= COLS or y < 0 or y >= ROWS or new_head in self.snake[1:]):
            done = True
            reward = -10
            return self.get_state(), reward, done

        # Distância antes e depois do movimento
        prev_head = self.snake[1]
        prev_dist = abs(self.food[0] - prev_head[0]) + abs(self.food[1] - prev_head[1])
        curr_dist = abs(self.food[0] - new_head[0]) + abs(self.food[1] - new_head[1])

        if new_head == self.food:
            reward = 10
            self.spawn_food()
        else:
            self.snake.pop()
            if curr_dist < prev_dist:
                reward = 1
            else:
                reward = -1

        return self.get_state(), reward, done

    def get_state(self):
        head = self.snake[0]
        food_x, food_y = self.food

        return np.array([
            int(self.direction == UP),
            int(self.direction == RIGHT),
            int(self.direction == DOWN),
            int(self.direction == LEFT),
            int(food_x > head[0]),   # comida está à direita
            int(food_x < head[0]),   # comida está à esquerda
            int(food_y > head[1]),   # comida está abaixo
            int(food_y < head[1]),   # comida está acima
        ], dtype=np.float32)

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

# DQN Model
class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)

# Agente RL
class DQNAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=100_000)
        self.batch_size = 1000
        self.model = LinearQNet(8, 64, 3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # 0: turn left, 1: straight, 2: turn right
        state_t = torch.tensor(state, dtype=torch.float32)
        prediction = self.model(state_t)
        return torch.argmax(prediction).item()

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

        self.optimizer.zero_grad()
        loss = self.loss_fn(q_values, target_q.detach())
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Loop de treino
def draw_game(screen, game):
    screen.fill((0, 0, 0))  # Fundo preto

    # Desenhar a comida
    food_x, food_y = game.food
    pygame.draw.rect(screen, (255, 0, 0), (food_x * BLOCK_SIZE, food_y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    # Desenhar a cobra
    for block in game.snake:
        pygame.draw.rect(screen, (0, 255, 0), (block[0] * BLOCK_SIZE, block[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    pygame.display.flip()

# Função para jogar com o modelo treinado e mostrar visualmente
def play():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake RL - Play Mode")
    clock = pygame.time.Clock()

    game = SnakeGame()
    agent = DQNAgent()
    agent.model.load_state_dict(torch.load("snake_agent_weights.pt"))
    agent.model.eval()
    agent.epsilon = 0  # Sem exploração, só exploração (modelo já treinado)

    state = game.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                pygame.quit()
                return

        action = agent.act(state)
        next_state, reward, done = game.step(action)
        state = next_state

        draw_game(screen, game)
        clock.tick(10)  # FPS do jogo (10 frames por segundo)

    print("Fim do jogo! Pontuação:", len(game.snake) - 1)
    pygame.quit()

# Loop de treino
def train():
    game = SnakeGame()
    agent = DQNAgent()
    scores = []         # para guardar o score (maçãs comidas)
    avg_scores = []     # médias a cada 100 episódios
    episodes_avg = []

    max_score = 0

    for episode in range(1, 20001):
        state = game.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

        # Pontuação = maçãs comidas = tamanho da cobra - 1
        score = len(game.snake) - 1
        scores.append(score)

        agent.train_long_memory()

        if score > max_score:
            print(f"New Max score! {score}")
            max_score = score
        
        if episode % 100 == 0:
            avg_score = sum(scores[-100:]) / 100
            avg_scores.append(avg_score)
            episodes_avg.append(episode)
            print(f"Episodes {episode-99} to {episode} — Average Apples: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")

    # Plotar média de maçãs a cada 100 episódios
    plt.plot(episodes_avg, avg_scores, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Average Apples Eaten (per 100 episodes)")
    plt.title("DQN Snake Training Performance")
    plt.grid()
    plt.show()

    torch.save(agent.model.state_dict(), "snake_agent_weights.pt")
    print("Modelo salvo em snake_agent_weights.pt")


if __name__ == "__main__":
    train()
    play()  # Roda o jogo visualmente depois do treino

