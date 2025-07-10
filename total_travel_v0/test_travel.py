import pygame
import torch
from train_travel import TravelGame, Agent, WIDTH, HEIGHT, PLAYER_SIZE, OBSTACLE_WIDTH, OBSTACLE_HEIGHT, FPS

def play():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Travel DQN Teste")
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 36)  # Fonte padrão, tamanho 36

    game = TravelGame()
    agent = Agent(use_gpu=False)
    agent.model.load_state_dict(torch.load("travel_dqn.pt"))
    agent.model.eval()
    agent.epsilon = 0  # sem exploração

    state = game.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        action = agent.act(state)
        next_state, reward, done = game.step(action)
        state = next_state

        # Desenhar
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (0, 255, 0),
                         (game.player_x, HEIGHT - PLAYER_SIZE - 10, PLAYER_SIZE, PLAYER_SIZE))
        pygame.draw.rect(screen, (255, 0, 0),
                         (game.obstacle[0], game.obstacle[1], OBSTACLE_WIDTH, OBSTACLE_HEIGHT))

        # Renderizar a pontuação
        score_text = font.render(f"Score: {game.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))  # Posição: 10 pixels da esquerda e 10 do topo

        pygame.display.flip()
        clock.tick(FPS)

    print(f"Fim do teste! Score: {game.score}")
    pygame.quit()

if __name__ == "__main__":
    play()
