import pygame
import random

# Configurações
BLOCK_SIZE = 20
ROWS, COLS = 10, 10  # Par x par para permitir ciclo Hamiltoniano
WIDTH, HEIGHT = COLS * BLOCK_SIZE, ROWS * BLOCK_SIZE

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Hamiltonian Ciclo")

def create_hamiltonian_cycle():
    path = []
    
    # Descer a coluna 0: (0,0) até (9,0)
    for row in range(ROWS):
        path.append((0, row))
    
    # Colunas 1 até 8 zigzag vertical:
    for col in range(1, 9):
        if col % 2 == 1:
            # col ímpar: sobe da linha 9 até 1
            for row in range(9, 0, -1):
                path.append((col, row))
        else:
            # col par: desce da linha 1 até 9
            for row in range(1, 10):
                path.append((col, row))
    
    # Coluna 9 sobe da linha 9 até 0
    for row in range(9, -1, -1):
        path.append((9, row))
    
    # Agora, na linha 0, vai para esquerda de (0,9) até (0,1)
    for col in range(8, 0, -1):
        path.append((col, 0))
    
    # Por fim, volta para (0,0) para fechar o ciclo
    # (0,0) já está no início do caminho, fechando o ciclo
    # Mas se quiser que o ciclo volte em um passo, podemos deixar (0,0) no início
    # e o caminho é circular via índice.
    
    return path


class SnakeGameHamiltonian:
    def __init__(self):
        self.snake = [(0, 0)]
        self.path = create_hamiltonian_cycle()
        self.path_index = 0
        self.spawn_food()

    def spawn_food(self):
        free_spaces = [(x, y) for y in range(ROWS) for x in range(COLS) if (x, y) not in self.snake]
        if free_spaces:
            self.food = random.choice(free_spaces)
        else:
            self.food = None

    def step(self):
        self.path_index = (self.path_index + 1) % len(self.path)
        next_head = self.path[self.path_index]

        if next_head in self.snake:
            print("Colisão detectada!")
            pygame.quit()
            exit()

        self.snake.insert(0, next_head)

        if next_head == self.food:
            self.spawn_food()
        else:
            self.snake.pop()

    def draw(self):
        screen.fill((0, 0, 0))

        # Desenhar ciclo (opcional para visualizar o caminho)
        for point in self.path:
            pygame.draw.rect(screen, (50, 50, 50),
                             (point[0]*BLOCK_SIZE, point[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

        # Desenhar comida
        if self.food:
            pygame.draw.rect(screen, (255, 0, 0),
                             (self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # Desenhar cobra
        for block in self.snake:
            pygame.draw.rect(screen, (0, 255, 0),
                             (block[0]*BLOCK_SIZE, block[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        pygame.display.flip()

def play_hamiltonian():
    clock = pygame.time.Clock()
    game = SnakeGameHamiltonian()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        game.step()
        game.draw()
        clock.tick(50)

    pygame.quit()

if __name__ == "__main__":
    play_hamiltonian()
