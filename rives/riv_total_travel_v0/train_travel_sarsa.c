#include <riv.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#define MAP_WIDTH 20
#define MAP_HEIGHT 20
#define TILE_SIZE 8
#define MAX_OBSTACLES 5
#define ACTIONS 3  // 0 = esquerda, 1 = ficar, 2 = direita

#define TRAINING_EPISODES 10000

float Q[MAP_WIDTH / 2][MAP_WIDTH / 2][MAP_HEIGHT / 2][ACTIONS];

float epsilon = 1.0f;
float alpha = 0.1f;
float gamma = 0.9f;

typedef struct {
    int x;
    int y;
    bool active;
} Obstacle;

typedef struct {
    int player_x;
    Obstacle obstacles[MAX_OBSTACLES];
} GameState;

typedef struct {
    int player_x;
    int obs_x;
    int obs_y;
} StateIndices;

void initialize_q() {
    for(int px=0; px<MAP_WIDTH/2; px++)
        for(int ox=0; ox<MAP_WIDTH/2; ox++)
            for(int oy=0; oy<MAP_HEIGHT/2; oy++)
                for(int a=0; a<ACTIONS; a++)
                    Q[px][ox][oy][a] = 0.0f;
}

StateIndices get_state_indices(GameState* gs) {
    int closest_idx = -1;
    int min_y = MAP_HEIGHT + 1;
    for(int i=0; i<MAX_OBSTACLES; i++) {
        if(gs->obstacles[i].active && gs->obstacles[i].y < min_y) {
            min_y = gs->obstacles[i].y;
            closest_idx = i;
        }
    }
    StateIndices si;
    si.player_x = gs->player_x / 2;
    if(closest_idx == -1) {
        si.obs_x = 0;
        si.obs_y = MAP_HEIGHT/2 - 1;
    } else {
        si.obs_x = gs->obstacles[closest_idx].x / 2;
        si.obs_y = gs->obstacles[closest_idx].y / 2;
    }
    return si;
}

int choose_action(StateIndices s, float curr_epsilon) {
    float r = (float)rand() / RAND_MAX;
    if(r < curr_epsilon) {
        return rand() % ACTIONS;
    } else {
        // greedy
        int best_a = 0;
        float best_q = Q[s.player_x][s.obs_x][s.obs_y][0];
        for(int a=1; a<ACTIONS; a++) {
            float val = Q[s.player_x][s.obs_x][s.obs_y][a];
            if(val > best_q) {
                best_q = val;
                best_a = a;
            }
        }
        return best_a;
    }
}

void spawn_obstacle(GameState* gs) {
    for(int i=0; i<MAX_OBSTACLES; i++) {
        if(!gs->obstacles[i].active) {
            gs->obstacles[i].active = true;
            gs->obstacles[i].x = rand() % MAP_WIDTH;
            gs->obstacles[i].y = 0;
            break;
        }
    }
}

bool step_game(GameState* gs, int action, int* reward, int* score) {
    *reward = 1;

    if(action == 0 && gs->player_x > 0) gs->player_x--;
    else if(action == 2 && gs->player_x < MAP_WIDTH - 1) gs->player_x++;

    for(int i=0; i<MAX_OBSTACLES; i++) {
        if(gs->obstacles[i].active) {
            gs->obstacles[i].y++;
            if(gs->obstacles[i].y >= MAP_HEIGHT) {
                gs->obstacles[i].active = false;
                (*score)++;
            } else if(gs->obstacles[i].x == gs->player_x && gs->obstacles[i].y == MAP_HEIGHT - 1) {
                *reward = -5;
                return true; // game over
            }
        }
    }
    return false;
}

void train_sarsa() {
    FILE *fp = fopen("training_log_sarsa.csv", "w");
    if(!fp) {
        printf("Erro ao abrir CSV\n");
        return;
    }
    fprintf(fp, "Episode,AverageScore,Epsilon\n");

    int total_score_100 = 0;

    for(int ep=1; ep<=TRAINING_EPISODES; ep++) {
        GameState gs = { MAP_WIDTH/2 };
        for(int i=0; i<MAX_OBSTACLES; i++) gs.obstacles[i].active = false;

        int score = 0;
        int frame = 0;
        bool game_over = false;

        StateIndices state = get_state_indices(&gs);
        int action = choose_action(state, epsilon);

        while(!game_over && frame < 1000) {
            if(frame % (10 + ep/50000) == 0) spawn_obstacle(&gs);

            int reward;
            game_over = step_game(&gs, action, &reward, &score);

            StateIndices next_state = get_state_indices(&gs);
            int next_action = choose_action(next_state, epsilon);

            // SARSA update
            float q = Q[state.player_x][state.obs_x][state.obs_y][action];
            float q_next = Q[next_state.player_x][next_state.obs_x][next_state.obs_y][next_action];

            Q[state.player_x][state.obs_x][state.obs_y][action] +=
                alpha * (reward + gamma * q_next - q);

            state = next_state;
            action = next_action;

            frame++;
        }

        total_score_100 += score;
        if(epsilon > 0.01f) epsilon *= 0.999f;

        if(ep % 100 == 0) {
            fprintf(fp, "%d,%.2f,%.4f\n", ep, total_score_100 / 100.0f, epsilon);
            printf("Ep %d Avg %.2f Eps %.4f\n", ep, total_score_100 / 100.0f, epsilon);
            total_score_100 = 0;
        }
    }
    fclose(fp);
}

int main() {
    srand(time(NULL));
    initialize_q();

    printf("Treinando SARSA...\n");
    train_sarsa();
    printf("Treinamento concluído! Visualizando...\n");

    riv->width = MAP_WIDTH * TILE_SIZE;
    riv->height = MAP_HEIGHT * TILE_SIZE;
    riv->target_fps = 15;

    GameState gs = { MAP_WIDTH/2 };
    for(int i=0; i<MAX_OBSTACLES; i++) gs.obstacles[i].active = false;

    int frame = 0;
    int score = 0;
    bool game_over = false;

    while(riv_present()) {
        riv_clear(RIV_COLOR_DARKSLATE);

        if(!game_over) {
            if(frame % 10 == 0) spawn_obstacle(&gs);

            StateIndices state = get_state_indices(&gs);
            int action = choose_action(state, 0.0f);  // epsilon=0 no play (sempre greedy)

            int reward;
            game_over = step_game(&gs, action, &reward, &score);
        } else {
            riv_draw_text("GAME OVER", RIV_SPRITESHEET_FONT_5X7, RIV_CENTER,
                riv->width/2, riv->height/2 - 10, 1, RIV_COLOR_RED);
            riv_draw_text("PRESS ANY KEY", RIV_SPRITESHEET_FONT_5X7, RIV_CENTER,
                riv->width/2, riv->height/2 + 10, 1, RIV_COLOR_WHITE);

            if(riv->key_toggle_count > 0) {
                gs.player_x = MAP_WIDTH / 2;
                for(int i=0; i<MAX_OBSTACLES; i++) gs.obstacles[i].active = false;
                game_over = false;
                frame = 0;
                score = 0;
            }
        }

        riv_draw_rect_fill(gs.player_x * TILE_SIZE, (MAP_HEIGHT - 1) * TILE_SIZE,
            TILE_SIZE, TILE_SIZE, RIV_COLOR_LIGHTGREEN);

        for(int i=0; i<MAX_OBSTACLES; i++) {
            if(gs.obstacles[i].active) {
                riv_draw_rect_fill(gs.obstacles[i].x * TILE_SIZE, gs.obstacles[i].y * TILE_SIZE,
                    TILE_SIZE, TILE_SIZE, RIV_COLOR_LIGHTRED);
            }
        }

        char buf[64];
        sprintf(buf, "SCORE:%d", score);
        riv_draw_text(buf, RIV_SPRITESHEET_FONT_5X7, RIV_TOPLEFT, 2, 2, 1, RIV_COLOR_WHITE);

        riv_present();
        frame++;
    }

    return 0;
}
