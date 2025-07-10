import torch
from train_travel import TravelGame, Agent

def test_no_visualization(n_episodes=50):
    game = TravelGame()
    agent = Agent(use_gpu=False)
    agent.model.load_state_dict(torch.load("travel_dqn.pt"))
    agent.model.eval()
    agent.epsilon = 0  # sem exploração

    scores = []

    for ep in range(1, n_episodes + 1):
        state = game.reset()
        done = False
        ep_score = 0
        max_steps = 1000  # evitar loops infinitos, caso necessário

        steps = 0
        while not done and steps < max_steps:
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            state = next_state
            ep_score = game.score  # game.score já conta os obstáculos evitados
            steps += 1

        scores.append(ep_score)
        print(f"Episódio {ep}: Score = {ep_score}")

    avg_score = sum(scores) / len(scores)
    print(f"\nScore médio em {n_episodes} episódios: {avg_score:.2f}")

if __name__ == "__main__":
    test_no_visualization()
