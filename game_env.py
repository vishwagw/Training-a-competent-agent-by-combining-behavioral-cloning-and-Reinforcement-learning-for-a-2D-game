import os
import random
import math
import numpy as np
import pygame

# Use a lightweight, gym-like environment wrapper around the existing game objects
# This module intentionally avoids opening a visible window by using the
# SDL_VIDEODRIVER='dummy' when render_mode is False. The observation is a
# compact float vector describing player, nearby enemies and bullets.

from main import Player, Bullet, Enemy, WIDTH, HEIGHT


class GameEnv:
    """A minimal environment wrapper around the game's core classes.

    Observation: fixed-size numpy float32 vector with:
      [player_x_norm, player_y_norm, health_norm, score_norm,
       enemies (up to N): dx_norm, dy_norm ...,
       bullets (up to M): dx_norm, dy_norm ...]

    Action (discrete):
      0: noop
      1: left
      2: right
      3: up
      4: down
      5: fire

    Reward:
      +1 for enemy killed
      -1 for player damage
      -0.001 per step to encourage progress

    Episode ends when player_health <= 0 or after max_steps.
    """

    def __init__(self, render_mode=False, max_steps=1000, n_enemies=4, n_bullets=6):
        self.render_mode = render_mode
        # For headless operation use the dummy driver to avoid creating a window
        if not render_mode:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()

        # Only create an actual window when rendering is requested
        self.screen = None
        if render_mode:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

        self.n_enemies = n_enemies
        self.n_bullets = n_bullets
        self.max_steps = max_steps
        self.action_space_n = 6

        # internal state
        self.player = None
        self.bullets = []
        self.enemies = []
        self.spawn_cooldown = 0
        self.spawn_time = 90
        self.score = 0
        self.player_health = 3
        self.step_count = 0

        self.reset()

    def reset(self):
        self.player = Player()
        self.bullets = []
        self.enemies = []
        self.spawn_cooldown = 0
        self.score = 0
        self.player_health = 3
        self.step_count = 0
        self.game_over = False
        return self._get_obs()

    def _get_obs(self):
        # Compact vector representation
        px = self.player.rect.centerx / float(WIDTH)
        py = self.player.rect.centery / float(HEIGHT)
        health = float(self.player_health) / 3.0
        score = float(self.score) / 1000.0

        # nearest enemies by distance
        enemies = sorted(self.enemies, key=lambda e: (e.rect.centerx - self.player.rect.centerx) ** 2 + (e.rect.centery - self.player.rect.centery) ** 2)
        e_vec = []
        for i in range(self.n_enemies):
            if i < len(enemies):
                dx = (enemies[i].rect.centerx - self.player.rect.centerx) / float(WIDTH)
                dy = (enemies[i].rect.centery - self.player.rect.centery) / float(HEIGHT)
            else:
                dx, dy = 0.0, 0.0
            e_vec.extend([dx, dy])

        # nearest bullets
        bullets = sorted(self.bullets, key=lambda b: (b.rect.centerx - self.player.rect.centerx) ** 2 + (b.rect.centery - self.player.rect.centery) ** 2)
        b_vec = []
        for i in range(self.n_bullets):
            if i < len(bullets):
                dx = (bullets[i].rect.centerx - self.player.rect.centerx) / float(WIDTH)
                dy = (bullets[i].rect.centery - self.player.rect.centery) / float(HEIGHT)
            else:
                dx, dy = 0.0, 0.0
            b_vec.extend([dx, dy])

        obs = [px, py, health, score] + e_vec + b_vec
        return np.asarray(obs, dtype=np.float32)

    def step(self, action):
        """Apply action and advance the game one tick/frame.

        Returns: obs, reward, done, info
        """
        reward = 0.0
        done = False

        # map action -> controls
        # first simple movement
        if action == 1:
            self.player.rect.x -= self.player.speed
        elif action == 2:
            self.player.rect.x += self.player.speed
        elif action == 3:
            self.player.rect.y -= self.player.speed
        elif action == 4:
            self.player.rect.y += self.player.speed
        elif action == 5:
            if self.player.can_fire():
                self.bullets.append(self.player.fire())

        # keep inside bounds
        self.player.rect.left = max(0, self.player.rect.left)
        self.player.rect.right = min(WIDTH, self.player.rect.right)
        self.player.rect.top = max(0, self.player.rect.top)
        self.player.rect.bottom = min(HEIGHT, self.player.rect.bottom)

        # update cooldowns
        self.player.update()

        # update bullets
        for b in self.bullets:
            b.update()
        self.bullets = [b for b in self.bullets if b.alive]

        # spawn enemies similar to main.py
        if self.spawn_cooldown <= 0:
            self.spawn_enemy()
            self.spawn_cooldown = self.spawn_time
        else:
            self.spawn_cooldown -= 1

        # update enemies
        for e in self.enemies:
            e.update(self.player.rect)
        self.enemies = [e for e in self.enemies if e.alive]

        # collisions
        killed = 0
        for b in list(self.bullets):
            for e in list(self.enemies):
                if b.rect.colliderect(e.rect):
                    b.alive = False
                    e.alive = False
                    self.score += 100
                    killed += 1

        if killed > 0:
            reward += float(killed) * 1.0

        # enemy hits player
        damage = 0
        for e in list(self.enemies):
            if e.rect.colliderect(self.player.rect):
                e.alive = False
                self.player_health -= 1
                damage += 1
                if self.player_health <= 0:
                    self.game_over = True

        if damage > 0:
            reward -= float(damage) * 1.0

        # small living penalty
        reward -= 0.001

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.game_over = True

        obs = self._get_obs()
        done = bool(self.game_over)
        info = {"score": self.score}
        return obs, float(reward), done, info

    def spawn_enemy(self):
        x = random.randint(20, WIDTH - 20)
        y = -40
        speed = random.randint(2, 4)
        self.enemies.append(Enemy(x, y, speed=speed))

    def render(self, mode="human"):
        # draws to the screen if render_mode True; otherwise returns an RGB array
        surf = pygame.Surface((WIDTH, HEIGHT))
        surf.fill((0, 0, 0))
        self.player.draw(surf)
        for b in self.bullets:
            b.draw(surf)
        for e in self.enemies:
            e.draw(surf)

        if self.render_mode and self.screen:
            self.screen.blit(surf, (0, 0))
            pygame.display.flip()
            return None
        # return a downscaled observation image (H, W, C)
        small = pygame.transform.smoothscale(surf, (160, 120))
        arr = pygame.surfarray.array3d(small)
        # convert from (W,H,C) to (H,W,C)
        arr = np.transpose(arr, (1, 0, 2))
        return arr

    def close(self):
        try:
            pygame.quit()
        except Exception:
            pass


if __name__ == "__main__":
    # quick manual smoke run
    env = GameEnv(render_mode=False)
    obs = env.reset()
    for _ in range(20):
        a = random.randint(0, env.action_space_n - 1)
        o, r, d, info = env.step(a)
        if d:
            break
    print("smoke run finished", o.shape)
    env.close()
