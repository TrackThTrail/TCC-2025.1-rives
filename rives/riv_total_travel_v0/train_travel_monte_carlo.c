#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#define MAP_WIDTH 10
#define MAP_HEIGHT 10
#define ACTIONS 3

float Q[MAP_WIDTH][MAP_WIDTH][MAP_HEIGHT][ACTIONS];
int visit_count[MAP_WIDTH][MAP_WIDTH][MAP_HEIGHT][ACTIONS];

typedef struct {
    int player_x;
    int obs_x;
    int obs_y;
    int action;
} EpisodeStep;

typedef struct {
    int player_x;
    int obs_x;
    int obs_y;
    bool active;
} GameState;

#define MAX_EPISODE_STEPS 100
EpisodeStep episode_steps[MAX_EPISODE_STEPS];

int choose_action(GameState *gs, float epsilon) {
    if ((float)rand() / RAND_MAX < epsilon) {
        return rand() % ACTIONS;
    } else {
        float best_q = Q[gs->player_x][gs->obs_x][gs->obs_y][0];
        int best_a = 0;
        for (int a = 1; a < ACTIONS; a++) {
            if (Q[gs->player_x][gs->obs_x][gs->obs_y][a] > best_q) {
                best_q = Q[gs->player_x][gs->obs_x][gs->obs_y][a];
                best_a = a;
            }
        }
        return best_a;
    }
}

void reset_game(GameState *gs) {
    gs->player_x = MAP_WIDTH / 2;
    gs->obs_x = rand() % MAP_WIDTH;
    gs->obs_y = 0;
    gs->active = true;
}

bool step(GameState *gs, int action, int *reward) {
    if (action == 0 && gs->player_x > 0) gs->player_x--;
    if (action == 2 && gs->player_x < MAP_WIDTH - 1) gs->player_x++;

    if (gs->active) {
        gs->obs_y++;
        if (gs->obs_y >= MAP_HEIGHT) {
            gs->active = false;
            *reward = 1;
            return false;
        }
        if (gs->obs_y == MAP_HEIGHT - 1 && gs->obs_x == gs->player_x) {
            *reward = -5;
            return true;
        }
    }
    *reward = 0;
    return false;
}

void train_monte_carlo(int episodes) {
    float epsilon = 1.0f;
    int sum_reward_100 = 0;

    for (int ep = 1; ep <= episodes; ep++) {
        printf("Episódio");
        GameState gs;
        reset_game(&gs);
        int step_idx = 0;
        bool done = false;
        int reward;
        int total_reward = 0;

        while (!done && step_idx < MAX_EPISODE_STEPS) {
            int action = choose_action(&gs, epsilon);
            episode_steps[step_idx].player_x = gs.player_x;
            episode_steps[step_idx].obs_x = gs.obs_x;
            episode_steps[step_idx].obs_y = gs.obs_y;
            episode_steps[step_idx].action = action;

            done = step(&gs, action, &reward);
            total_reward += reward;
            step_idx++;
        }

        // Update Q for all visited states
        for (int i = 0; i < step_idx; i++) {
            int px = episode_steps[i].player_x;
            int ox = episode_steps[i].obs_x;
            int oy = episode_steps[i].obs_y;
            int a = episode_steps[i].action;

            visit_count[px][ox][oy][a]++;
            float old_q = Q[px][ox][oy][a];
            Q[px][ox][oy][a] += (total_reward - old_q) / visit_count[px][ox][oy][a];
        }

        sum_reward_100 += total_reward;

        if (ep % 100 == 0) {
            float avg = sum_reward_100 / 100.0f;
            printf("Episódio %d - Média dos últimos 100: %.2f - Epsilon: %.4f\n", ep, avg, epsilon);
            sum_reward_100 = 0;
        }

        if (epsilon > 0.01f) epsilon *= 0.999f;
    }
}

int main() {
    srand(1234); // semente fixa para reprodutibilidade no RIV

    train_monte_carlo(10000);

    printf("Treinamento concluído!\n");

    return 0;
}
