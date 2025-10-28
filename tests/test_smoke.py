"""Minimal smoke tests for project structure.

These tests avoid opening a display and only exercise the basic classes.
"""
import os
import sys
import pygame

# ensure project root is on sys.path so tests can import `main` when run by pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import Player, Bullet, Enemy
from game_env import GameEnv


def test_player_and_bullet_creation():
    # constructing Player and Bullet should not crash
    p = Player()
    assert p.rect.width > 0
    b = Bullet(p.rect.centerx, p.rect.top)
    assert b.rect.width > 0


def test_enemy_creation_and_overlap():
    p = Player()
    b = Bullet(p.rect.centerx, p.rect.top)
    e = Enemy(b.rect.centerx, b.rect.top)
    # rectangles should overlap when positioned at same coordinates
    assert b.rect.colliderect(e.rect)


def test_env_step():
    env = GameEnv(render_mode=False, max_steps=50)
    obs = env.reset()
    assert obs is not None
    # take a single noop step
    o, r, d, info = env.step(0)
    assert isinstance(o, (list, tuple, type(obs))) or o.shape[0] > 0
    env.close()
