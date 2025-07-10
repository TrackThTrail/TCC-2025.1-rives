import pygame
import random

# Configurações do jogo
WIDTH, HEIGHT = 400, 600
PLAYER_SIZE = 40
OBSTACLE_WIDTH = 40
OBSTACLE_HEIGHT = 40
OBSTACLE_SPEED = 5
FPS = 60

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Desvie dos obstáculos")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 30)

# Função para criar um novo obstáculo
def create_obstacle():
    x_pos = random.randint(0, WIDTH - OBSTACLE_WIDTH)
    return pygame.Rect(x_pos, 0, OBSTACLE_WIDTH, OBSTACLE_HEIGHT)

def main():
    player = pygame.Rect(WIDTH // 2, HEIGHT - PLAYER_SIZE - 10, PLAYER_SIZE, PLAYER_SIZE)
    obstacles = [create_obstacle()]
    score = 0
    running = True

    while running:
        clock.tick(FPS)
        screen.fill((0, 0, 0))  # fundo preto

        # Eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Controle manual (para testar o jogo antes do DQN)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            player.x -= 5
        if keys[pygame.K_RIGHT]:
            player.x += 5

        # Limitar movimento do player na tela
        player.x = max(0, min(WIDTH - PLAYER_SIZE, player.x))

        # Atualizar obstáculos
        for obstacle in obstacles:
            obstacle.y += OBSTACLE_SPEED

        # Remover obstáculos que saíram da tela e adicionar novos
        obstacles = [obs for obs in obstacles if obs.y < HEIGHT]
        if len(obstacles) == 0 or obstacles[-1].y > 150:
            obstacles.append(create_obstacle())
            score += 1

        # Checar colisão
        for obstacle in obstacles:
            if player.colliderect(obstacle):
                running = False

        # Desenhar player e obstáculos
        pygame.draw.rect(screen, (0, 255, 0), player)
        for obstacle in obstacles:
            pygame.draw.rect(screen, (255, 0, 0), obstacle)

        # Mostrar pontuação
        score_surface = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_surface, (10, 10))

        pygame.display.flip()

    # Fim de jogo
    print(f"Game Over! Pontuação final: {score}")
    pygame.quit()

if __name__ == "__main__":
    main()
