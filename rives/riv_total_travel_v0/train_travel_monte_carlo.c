#include <riv.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#define MAP_WIDTH 10
#define MAP_HEIGHT 10
#define TILE_SIZE 12
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

int choose_action_greedy(GameState *gs) {
    int best_a = 0;
    float best_q = Q[gs->player_x][gs->obs_x][gs->obs_y][0];
    for (int a = 1; a < ACTIONS; a++) {
        if (Q[gs->player_x][gs->obs_x][gs->obs_y][a] > best_q) {
            best_q = Q[gs->player_x][gs->obs_x][gs->obs_y][a];
            best_a = a;
        }
    }
    return best_a;
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
            printf("Ep %d - Média dos últimos 100: %.2f - Eps: %.4f\n", ep, avg, epsilon);
            sum_reward_100 = 0;
        }

        if (epsilon > 0.01f) epsilon *= 0.999f;
    }
}

void run_trained_agent() {
    GameState gs;
    reset_game(&gs);

    int score = 0;
    int frame = 0;
    bool game_over = false;

    while (riv_present()) {
        riv_clear(RIV_COLOR_DARKSLATE);

        if (!game_over) {
            if (frame % 10 == 0 && !gs.active) {
                reset_game(&gs);
            }

            int reward;
            int action = choose_action_greedy(&gs);
            game_over = step(&gs, action, &reward);
            score += reward;
        } else {
            riv_draw_text("GAME OVER", RIV_SPRITESHEET_FONT_5X7, RIV_CENTER,
                          riv->width / 2, riv->height / 2 - 10, 1, RIV_COLOR_RED);
            riv_draw_text("PRESS ANY KEY", RIV_SPRITESHEET_FONT_5X7, RIV_CENTER,
                          riv->width / 2, riv->height / 2 + 10, 1, RIV_COLOR_WHITE);

            if (riv->key_toggle_count > 0) {
                reset_game(&gs);
                score = 0;
                frame = 0;
                game_over = false;
            }
        }

        // desenha jogador
        riv_draw_rect_fill(gs.player_x * TILE_SIZE, (MAP_HEIGHT - 1) * TILE_SIZE,
                           TILE_SIZE, TILE_SIZE, RIV_COLOR_LIGHTGREEN);

        // desenha obstáculo
        if (gs.active) {
            riv_draw_rect_fill(gs.obs_x * TILE_SIZE, gs.obs_y * TILE_SIZE,
                               TILE_SIZE, TILE_SIZE, RIV_COLOR_LIGHTRED);
        }

        char buf[64];
        sprintf(buf, "SCORE: %d", score);
        riv_draw_text(buf, RIV_SPRITESHEET_FONT_5X7, RIV_TOPLEFT, 2, 2, 1, RIV_COLOR_WHITE);

        riv_present();
        frame++;
    }
}

int main() {
    srand(1234);

    train_monte_carlo(10000);
    printf("Treinamento concluído. Visualizando...\n");

    riv->width = MAP_WIDTH * TILE_SIZE;
    riv->height = MAP_HEIGHT * TILE_SIZE;
    riv->target_fps = 15;

    run_trained_agent();
    return 0;
}
