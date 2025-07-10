import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import argparse
import matplotlib.pyplot as plt

# Configurações do jogo
WIDTH, HEIGHT = 400, 600
PLAYER_SIZE = 40
OBSTACLE_WIDTH = 40
OBSTACLE_HEIGHT = 40
OBSTACLE_SPEED = 10
FPS = 60

# Ações
LEFT = 0
STAY = 1
RIGHT = 2

# Ambiente
class TravelGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.player_x = WIDTH // 2
        self.obstacle = self.create_obstacle()
        self.score = 0
        return self.get_state()

    def create_obstacle(self):
        x = random.randint(0, WIDTH - OBSTACLE_WIDTH)
        y = 0
        return [x, y]

    def step(self, action):
        if action == LEFT:
            self.player_x -= 5
        elif action == RIGHT:
            self.player_x += 5
        self.player_x = max(0, min(WIDTH - PLAYER_SIZE, self.player_x))

        self.obstacle[1] += OBSTACLE_SPEED

        player_rect = pygame.Rect(self.player_x, HEIGHT - PLAYER_SIZE - 10, PLAYER_SIZE, PLAYER_SIZE)
        obstacle_rect = pygame.Rect(self.obstacle[0], self.obstacle[1], OBSTACLE_WIDTH, OBSTACLE_HEIGHT)

        done = False
        reward = 0.1

        if player_rect.colliderect(obstacle_rect):
            done = True
            reward = -10

        if self.obstacle[1] > HEIGHT:
            self.obstacle = self.create_obstacle()
            reward = 1
            self.score += 1

        return self.get_state(), reward, done

    def get_state(self):
        player_x_norm = self.player_x / (WIDTH - PLAYER_SIZE)
        obstacle_x_norm = self.obstacle[0] / (WIDTH - OBSTACLE_WIDTH)
        obstacle_y_norm = self.obstacle[1] / HEIGHT
        return np.array([player_x_norm, obstacle_x_norm, obstacle_y_norm], dtype=np.float32)

# Rede DQN
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Agente
class Agent:
    def __init__(self, use_gpu):
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print("CUDA disponível:", torch.cuda.is_available())
        print(f"Usando dispositivo: {self.device}")
        if self.device.type == "cuda":
            print("Dispositivo atual:", torch.cuda.current_device())
            print("Nome da GPU:", torch.cuda.get_device_name(0))

        self.model = DQN(3, 64, 3).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.loss_fn = nn.MSELoss()
        self.batch_size = 64
        self.train_every = 4
        self.step_count = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        state_t = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def train(self):
        self.step_count += 1
        if self.step_count % self.train_every != 0:
            return

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_, d = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32).to(self.device)
        s_ = torch.tensor(np.array(s_), dtype=torch.float32).to(self.device)
        a = torch.tensor(a).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.bool).to(self.device)

        q_vals = self.model(s)
        next_q_vals = self.model(s_)

        target_q = q_vals.clone()

        for i in range(self.batch_size):
            target = r[i]
            if not d[i]:
                target += self.gamma * torch.max(next_q_vals[i]).item()
            target_q[i][a[i]] = target

        self.optimizer.zero_grad()
        loss = self.loss_fn(q_vals, target_q.detach())
        loss.backward()
        self.optimizer.step()

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

def train(visualize, use_gpu):
    if visualize:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        clock = pygame.time.Clock()
    else:
        screen = None
        clock = None

    game = TravelGame()
    agent = Agent(use_gpu)
    scores = []

    max_steps = 1000

    for episode in range(1, 501):
        state = game.reset()
        done = False
        ep_score = 0
        steps = 0

        print(f"Episode {episode}")

        while not done and steps < max_steps:
            if visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

            action = agent.act(state)
            next_state, reward, done = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            ep_score += reward
            steps += 1

            if visualize:
                screen.fill((0, 0, 0))
                pygame.draw.rect(screen, (0, 255, 0),
                                 (game.player_x, HEIGHT - PLAYER_SIZE - 10, PLAYER_SIZE, PLAYER_SIZE))
                pygame.draw.rect(screen, (255, 0, 0),
                                 (game.obstacle[0], game.obstacle[1], OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
                pygame.display.flip()
                clock.tick(FPS)

        scores.append(game.score)
        # print(f"Ep {episode} — Score: {game.score:.2f}")
        if episode % 100 == 0:
            avg = sum(scores[-100:]) / 100
            print(f"Ep {episode} — Avg Score: {avg:.2f} — Epsilon: {agent.epsilon:.3f}")
            
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

    if visualize:
        pygame.quit()

    torch.save(agent.model.state_dict(), "travel_dqn.pt")
    print("Modelo salvo em travel_dqn.pt")

    # Plotar gráfico da média dos últimos 100 episódios
    if len(scores) >= 100:
        avg_scores = [np.mean(scores[max(0, i - 99):i + 1]) for i in range(len(scores))]
        plt.figure(figsize=(10, 5))
        plt.plot(avg_scores)
        plt.xlabel("Episódio")
        plt.ylabel("Média de Score (últimos 100)")
        plt.title("Desempenho do agente ao longo do tempo")
        plt.grid(True)
        plt.show()

def play(use_gpu):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    game = TravelGame()
    agent = Agent(use_gpu)
    agent.model.load_state_dict(torch.load("travel_dqn.pt"))
    agent.epsilon = 0

    state = game.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        action = agent.act(state)
        next_state, reward, done = game.step(action)
        state = next_state

        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (0, 255, 0),
                         (game.player_x, HEIGHT - PLAYER_SIZE - 10, PLAYER_SIZE, PLAYER_SIZE))
        pygame.draw.rect(screen, (255, 0, 0),
                         (game.obstacle[0], game.obstacle[1], OBSTACLE_WIDTH, OBSTACLE_HEIGHT))

        pygame.display.flip()
        clock.tick(FPS)

    print(f"Fim do jogo! Score: {game.score}")
    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Usar GPU se disponível")
    parser.add_argument("--play", action="store_true", help="Modo play em vez de treino")
    args = parser.parse_args()

    if args.play:
        play(args.gpu)
    else:
        train(visualize=False, use_gpu=args.gpu)
