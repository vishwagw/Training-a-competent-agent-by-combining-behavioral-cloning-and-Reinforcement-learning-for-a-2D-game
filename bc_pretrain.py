"""Behavioral cloning pretraining script.

Loads a demo `.npz` file produced by `collect_demo.py` and trains the same
MLP architecture as the RL script with a supervised cross-entropy loss to
predict actions from observations. Saves a checkpoint (PyTorch state_dict)
that can be loaded by `train_agent.py` before RL fine-tuning.
"""
import os
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from train_agent import MLP


def load_demo(path):
    data = np.load(path)
    obs = data["obs"]
    actions = data["actions"]
    return obs, actions


def train_bc(demo_path, out_path, epochs=10, batch_size=128, lr=1e-3):
    obs, actions = load_demo(demo_path)
    input_dim = obs.shape[1]
    n_actions = int(actions.max()) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim, hidden=128, n_actions=n_actions).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    idx = np.arange(len(obs))
    start = time.time()
    for ep in range(epochs):
        np.random.shuffle(idx)
        total_loss = 0.0
        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i : i + batch_size]
            s = torch.tensor(obs[batch_idx]).float().to(device)
            a = torch.tensor(actions[batch_idx]).long().to(device)
            logits = model(s)
            loss = loss_fn(logits, a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(batch_idx)
        avg = total_loss / len(obs)
        print(f"BC epoch {ep+1}/{epochs} avg_loss={avg:.4f}")
    dur = time.time() - start
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved BC model to {out_path} (trained {len(obs)} samples in {dur:.1f}s)")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("demo", help="Path to demo .npz file")
    p.add_argument("--out", default="checkpoints/bc_pretrained.pth", help="Output checkpoint path")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    args = p.parse_args()
    train_bc(args.demo, args.out, epochs=args.epochs, batch_size=args.batch)
