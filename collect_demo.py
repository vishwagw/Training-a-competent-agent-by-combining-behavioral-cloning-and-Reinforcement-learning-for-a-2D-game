"""Record human gameplay to a demo dataset for behavioral cloning.

Run this script, play the game using the controls (WASD/arrow keys, Space to fire),
and press ESC or close window to stop. The script saves observations and actions to
`demos/demo.npz` (created automatically).
"""
import os
import argparse
import time
import numpy as np
import pygame

from game_env import GameEnv


KEY_TO_ACTION = {
    # noop handled when no key pressed
    pygame.K_a: 1,  # left
    pygame.K_LEFT: 1,
    pygame.K_d: 2,  # right
    pygame.K_RIGHT: 2,
    pygame.K_w: 3,  # up
    pygame.K_UP: 3,
    pygame.K_s: 4,  # down
    pygame.K_DOWN: 4,
    pygame.K_SPACE: 5,  # fire
}


def main(output="demos/demo.npz", render=True, max_steps=2000):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    env = GameEnv(render_mode=render, max_steps=max_steps)
    obs = env.reset()

    observations = []
    actions = []

    running = True
    clock = pygame.time.Clock()
    print("Starting demo recording. Play with WASD or arrows, Space to fire. ESC to quit.")
    while running:
        # read pygame events
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        keys = pygame.key.get_pressed()
        # pick first matching key in KEY_TO_ACTION
        found = False
        for k, a in KEY_TO_ACTION.items():
            if keys[k]:
                action = a
                found = True
                break
        # step environment
        next_obs, reward, done, info = env.step(action)
        observations.append(obs.copy())
        actions.append(action)
        obs = next_obs

        if done:
            print("Episode finished; resetting environment")
            obs = env.reset()

        clock.tick(60)

    env.close()
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    np.savez_compressed(output, obs=observations, actions=actions)
    print(f"Saved {len(actions)} transitions to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="demos/demo.npz", help="Output demo file (.npz)")
    parser.add_argument("--no-render", dest="render", action="store_false", help="Run without opening a window")
    args = parser.parse_args()
    main(output=args.out, render=args.render)
