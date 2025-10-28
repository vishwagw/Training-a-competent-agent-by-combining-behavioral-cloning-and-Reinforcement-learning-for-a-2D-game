#!/usr/bin/env python3
"""Simple 2D shooter prototype using pygame.

Controls:
- Move: WASD or Arrow keys
- Fire: Space
- Quit: ESC or window close

This file contains a minimal Player, Bullet and Game loop to get started.
"""
import sys
import random
import math
import pygame
from pygame.locals import (
    QUIT,
    KEYDOWN,
    K_ESCAPE,
    K_SPACE,
    K_r,
    K_RETURN,
    K_a,
    K_d,
    K_w,
    K_s,
    K_LEFT,
    K_RIGHT,
    K_UP,
    K_DOWN,
)

WIDTH, HEIGHT = 800, 600
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 50, 50)
GREEN = (50, 220, 50)


class Player:
    def __init__(self, x=WIDTH // 2, y=HEIGHT - 60, w=40, h=20, speed=5):
        self.rect = pygame.Rect(0, 0, w, h)
        self.rect.centerx = x
        self.rect.bottom = y
        self.speed = speed
        self.cooldown = 0
        self.cooldown_time = 12  # frames between shots

    def handle_input(self, keys):
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.rect.x -= self.speed
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.rect.x += self.speed
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.rect.y -= self.speed
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.rect.y += self.speed

        # keep inside screen
        self.rect.left = max(0, self.rect.left)
        self.rect.right = min(WIDTH, self.rect.right)
        self.rect.top = max(0, self.rect.top)
        self.rect.bottom = min(HEIGHT, self.rect.bottom)

    def update(self):
        if self.cooldown > 0:
            self.cooldown -= 1

    def can_fire(self):
        return self.cooldown == 0

    def fire(self):
        self.cooldown = self.cooldown_time
        return Bullet(self.rect.centerx, self.rect.top)

    def draw(self, surf):
        pygame.draw.rect(surf, GREEN, self.rect)


class Bullet:
    def __init__(self, x, y, w=4, h=8, speed=-10):
        self.rect = pygame.Rect(0, 0, w, h)
        self.rect.centerx = x
        self.rect.bottom = y
        self.speed = speed
        self.alive = True

    def update(self):
        self.rect.y += self.speed
        if self.rect.bottom < 0:
            self.alive = False

    def draw(self, surf):
        pygame.draw.rect(surf, RED, self.rect)


class Enemy:
    def __init__(self, x, y, w=32, h=24, speed=2):
        self.rect = pygame.Rect(0, 0, w, h)
        self.rect.centerx = x
        self.rect.top = y
        self.speed = speed
        self.alive = True

    def update(self, player_rect=None):
        # simple kamikaze: move down, and slightly home toward player's x
        if player_rect:
            if self.rect.centerx < player_rect.centerx:
                self.rect.x += 1
            elif self.rect.centerx > player_rect.centerx:
                self.rect.x -= 1
        self.rect.y += self.speed
        if self.rect.top > HEIGHT:
            self.alive = False

    def draw(self, surf):
        pygame.draw.rect(surf, (200, 100, 50), self.rect)


class Particle:
    """Simple particle used for explosion effects."""
    def __init__(self, x, y, vx, vy, size=4, life=30, color=(255, 200, 80)):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.size = size
        self.life = life
        self.max_life = life
        self.color = color
        # pre-create a small surface for the particle (with per-pixel alpha)
        s = max(2, int(size * 2))
        surf = pygame.Surface((s, s), pygame.SRCALPHA)
        pygame.draw.circle(surf, color + (255,), (s // 2, s // 2), s // 2)
        self.surf = surf

    @property
    def alive(self):
        return self.life > 0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        # gravity-like pull
        self.vy += 0.15
        self.vx *= 0.99
        self.vy *= 0.99
        self.life -= 1

    def draw(self, target_surf):
        if self.life <= 0:
            return
        alpha = int(255 * (self.life / self.max_life))
        self.surf.set_alpha(alpha)
        pos = (int(self.x - self.surf.get_width() // 2), int(self.y - self.surf.get_height() // 2))
        target_surf.blit(self.surf, pos)


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("2D Shooter - Prototype")
        self.clock = pygame.time.Clock()
        self.player = Player()
        self.bullets = []
        self.enemies = []
        self.spawn_cooldown = 0
        self.spawn_time = 90  # frames between spawns (~1.5s)
        self.score = 0
        self.player_health = 3
        # font created after init; safe during runtime
        # Use Font(None, size) which uses a default font and is safe across platforms
        self.font = pygame.font.Font(None, 24)
        self.running = True
        # particle effects
        self.particles = []

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS)
            self.handle_events()
            # if game over, skip updates except for drawing the game over screen
            if not getattr(self, "game_over", False):
                self.update()
            self.draw()
        pygame.quit()
        sys.exit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    # quit from any mode
                    self.running = False
                # if game over, allow restart or quit
                elif getattr(self, "game_over", False):
                    if event.key in (K_r, K_RETURN):
                        self.restart()
                    elif event.key == K_ESCAPE:
                        self.running = False
                else:
                    if event.key == K_SPACE:
                        if self.player.can_fire():
                            self.bullets.append(self.player.fire())

        keys = pygame.key.get_pressed()
        # Only allow player movement if not game over
        if not getattr(self, "game_over", False):
            self.player.handle_input(keys)

    def update(self):
        self.player.update()
        for b in self.bullets:
            b.update()
        self.bullets = [b for b in self.bullets if b.alive]
        # spawn enemies
        if self.spawn_cooldown <= 0:
            self.spawn_enemy()
            self.spawn_cooldown = self.spawn_time
        else:
            self.spawn_cooldown -= 1

        # update enemies
        for e in self.enemies:
            e.update(self.player.rect)
        self.enemies = [e for e in self.enemies if e.alive]

        # collisions: bullets vs enemies
        for b in list(self.bullets):
            for e in list(self.enemies):
                if b.rect.colliderect(e.rect):
                    b.alive = False
                    e.alive = False
                    self.score += 100
                    # spawn explosion particles at enemy center
                    self.spawn_explosion(e.rect.centerx, e.rect.centery)

        # collisions: enemy vs player
        for e in list(self.enemies):
            if e.rect.colliderect(self.player.rect):
                e.alive = False
                self.player_health -= 1
                if self.player_health <= 0:
                    # set game over state instead of quitting immediately
                    self.game_over = True

        # update particles
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.alive]

    def draw(self):
        self.screen.fill(BLACK)
        self.player.draw(self.screen)
        for b in self.bullets:
            b.draw(self.screen)
        for e in self.enemies:
            e.draw(self.screen)

        # draw particles on top
        for p in self.particles:
            p.draw(self.screen)

        # HUD: score and health
        score_surf = self.font.render(f"Score: {self.score}", True, WHITE)
        hp_surf = self.font.render(f"Health: {self.player_health}", True, WHITE)
        self.screen.blit(score_surf, (8, 8))
        self.screen.blit(hp_surf, (8, 30))
        # If game over, draw overlay
        if getattr(self, "game_over", False):
            self.draw_game_over()
        pygame.display.flip()

    def draw_game_over(self):
        # translucent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        title = pygame.font.Font(None, 72).render("GAME OVER", True, WHITE)
        subtitle = pygame.font.Font(None, 28).render(
            f"Final Score: {self.score}    Press R or Enter to restart, ESC to quit", True, WHITE
        )
        tx = (WIDTH - title.get_width()) // 2
        ty = (HEIGHT - title.get_height()) // 2 - 30
        sx = (WIDTH - subtitle.get_width()) // 2
        sy = ty + title.get_height() + 16
        self.screen.blit(title, (tx, ty))
        self.screen.blit(subtitle, (sx, sy))

    def restart(self):
        # reset gameplay state
        self.player = Player()
        self.bullets = []
        self.enemies = []
        self.spawn_cooldown = 0
        self.score = 0
        self.player_health = 3
        self.game_over = False
        self.particles = []

    def spawn_explosion(self, x, y, count=12):
        for _ in range(count):
            angle = random.random() * 2 * 3.14159
            speed = random.uniform(1.5, 4.5)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            size = random.randint(2, 6)
            life = random.randint(18, 36)
            color = (255, random.randint(120, 220), random.randint(40, 120))
            self.particles.append(Particle(x, y, vx, vy, size=size, life=life, color=color))

    def spawn_enemy(self):
        x = random.randint(20, WIDTH - 20)
        y = -40
        speed = random.randint(2, 4)
        self.enemies.append(Enemy(x, y, speed=speed))


if __name__ == "__main__":
    Game().run()
