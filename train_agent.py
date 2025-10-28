"""Small DQN training script for the compact GameEnv.

This script provides a minimal DQN implementation (MLP + replay buffer)
to demonstrate training an agent on the environment. It's intentionally
small and dependency-light (requires numpy and torch).

Run it for a quick smoke training run; tune hyperparameters for longer
training runs.
"""
import random
import math
import time
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game_env import GameEnv


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buf)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden=128, n_actions=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def train(
    env,
    episodes=50,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    buffer_capacity=5000,
    min_buffer=500,
    pretrained_path=None,
):
    obs = env.reset()
    input_dim = obs.shape[0]
    n_actions = env.action_space_n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = MLP(input_dim, hidden=128, n_actions=n_actions).to(device)
    target = MLP(input_dim, hidden=128, n_actions=n_actions).to(device)
    # if a pretrained bc checkpoint is provided, load it into the policy
    if pretrained_path is not None:
        try:
            sd = torch.load(pretrained_path, map_location=device)
            policy.load_state_dict(sd)
            target.load_state_dict(sd)
            print(f"Loaded pretrained weights from {pretrained_path}")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
    target.load_state_dict(policy.state_dict())
    opt = optim.Adam(policy.parameters(), lr=lr)

    buf = ReplayBuffer(capacity=buffer_capacity)

    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay = 500.0

    steps_done = 0

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0.0
        eps = epsilon_start
        for t in range(env.max_steps):
            eps = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1.0 * steps_done / epsilon_decay)
            steps_done += 1
            if random.random() < eps:
                action = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    s = torch.from_numpy(state).float().unsqueeze(0).to(device)
                    q = policy(s)
                    action = int(q.argmax().item())

            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            buf.push(state, action, reward, next_state, done)
            state = next_state

            if len(buf) >= min_buffer:
                batch = buf.sample(batch_size)
                states = torch.tensor(np.stack(batch.state)).float().to(device)
                actions = torch.tensor(batch.action).long().to(device)
                rewards = torch.tensor(batch.reward).float().to(device)
                next_states = torch.tensor(np.stack(batch.next_state)).float().to(device)
                dones = torch.tensor(batch.done).float().to(device)

                q_values = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target(next_states).max(1)[0]
                    target_q = rewards + (1.0 - dones) * gamma * next_q

                loss = nn.functional.mse_loss(q_values, target_q)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if steps_done % 200 == 0:
                target.load_state_dict(policy.state_dict())

            if done:
                break

        print(f"Episode {ep+1}/{episodes} reward={ep_reward:.2f} eps={eps:.3f}")

    return policy


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default=None, help="Path to BC pretrained checkpoint (.pth)")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    env = GameEnv(render_mode=False, max_steps=200)
    start = time.time()
    policy = train(env, episodes=args.episodes, pretrained_path=args.pretrained)
    dur = time.time() - start
    print("Training finished in %.1fs" % dur)
    env.close()


if __name__ == "__main__":
    main()
