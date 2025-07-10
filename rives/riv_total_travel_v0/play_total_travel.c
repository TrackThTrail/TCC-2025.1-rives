#include <riv.h>

#define MAP_WIDTH 16
#define MAP_HEIGHT 16
#define TILE_SIZE 8
#define PLAYER_Y (MAP_HEIGHT - 2)  // Player na parte de baixo

typedef struct {
    int x;
    int y;
    bool active;
} Obstacle;

#define MAX_OBSTACLES 5
Obstacle obstacles[MAX_OBSTACLES];
int player_x;
bool game_over = false;
int score = 0;
int frame_count = 0;

void reset_game() {
    player_x = MAP_WIDTH / 2;
    for (int i = 0; i < MAX_OBSTACLES; i++) {
        obstacles[i].active = false;
    }
    game_over = false;
    score = 0;
    frame_count = 0;
}

void spawn_obstacle() {
    for (int i = 0; i < MAX_OBSTACLES; i++) {
        if (!obstacles[i].active) {
            obstacles[i].x = riv_rand_uint(MAP_WIDTH - 1);
            obstacles[i].y = 0;
            obstacles[i].active = true;
            break;
        }
    }
}

void update_game() {
    frame_count++;
    if (frame_count % 10 == 0) {
        spawn_obstacle();
    }

    if (riv->keys[RIV_GAMEPAD_LEFT].press && player_x > 0) player_x--;
    if (riv->keys[RIV_GAMEPAD_RIGHT].press && player_x < MAP_WIDTH - 1) player_x++;

    for (int i = 0; i < MAX_OBSTACLES; i++) {
        if (obstacles[i].active) {
            obstacles[i].y++;

            if (obstacles[i].y >= MAP_HEIGHT) {
                obstacles[i].active = false;
                score++;
            } else if (obstacles[i].x == player_x && obstacles[i].y == PLAYER_Y) {
                game_over = true;
                return;
            }
        }
    }
}

void draw_game() {
    // Draw player
    riv_draw_rect_fill(player_x * TILE_SIZE, PLAYER_Y * TILE_SIZE, TILE_SIZE, TILE_SIZE, RIV_COLOR_LIGHTGREEN);

    // Draw obstacles
    for (int i = 0; i < MAX_OBSTACLES; i++) {
        if (obstacles[i].active) {
            riv_draw_rect_fill(obstacles[i].x * TILE_SIZE, obstacles[i].y * TILE_SIZE, TILE_SIZE, TILE_SIZE, RIV_COLOR_LIGHTRED);
        }
    }

    // Draw score
    char score_text[16];
    sprintf(score_text, "SCORE %d", score);
    riv_draw_text(score_text, RIV_SPRITESHEET_FONT_5X7, RIV_TOPLEFT, 2, 2, 1, RIV_COLOR_WHITE);
}

int main() {
    riv->width = 128;
    riv->height = 128;
    riv->target_fps = 15;
    reset_game();

    while (riv_present()) {
        if (!game_over) {
            update_game();
            riv_clear(RIV_COLOR_DARKSLATE);
            draw_game();
        } else {
            riv_clear(RIV_COLOR_DARKSLATE);
            riv_draw_text("GAME OVER", RIV_SPRITESHEET_FONT_5X7, RIV_CENTER, 64, 64, 2, RIV_COLOR_RED);
            riv_draw_text("PRESS ANY KEY", RIV_SPRITESHEET_FONT_5X7, RIV_CENTER, 64, 90, 1, RIV_COLOR_WHITE);
            if (riv->key_toggle_count > 0) {
                reset_game();
            }
        }
    }

    return 0;
}
