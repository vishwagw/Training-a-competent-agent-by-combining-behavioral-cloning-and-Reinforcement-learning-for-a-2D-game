# Pretraining workflow (behavioral cloning + RL)

This project includes scripts to record human demonstrations, pretrain a policy via
behavioral cloning (BC), and then fine-tune the policy with RL (DQN in the example).

Overview
- `collect_demo.py` — run and play the game; records (obs, action) pairs to `demos/demo.npz`.
- `bc_pretrain.py` — train an MLP policy on the demo dataset using supervised cross-entropy; saves checkpoint.
- `train_agent.py` — DQN trainer that can load the BC checkpoint for warm-starting and then fine-tune with RL.

Quick start (PowerShell)
```powershell
python -m pip install -r requirements.txt
# record demos (open a window unless --no-render)
python collect_demo.py --out demos/demo.npz
# pretrain with BC
python bc_pretrain.py demos/demo.npz --out checkpoints/bc_pretrained.pth --epochs 20
# fine-tune with RL starting from BC weights
python train_agent.py --pretrained checkpoints/bc_pretrained.pth --episodes 200
```

Notes
- The environment uses a compact observation vector (not raw pixels) for faster experimentation.
- The BC step is intentionally simple; it's useful as a pretraining initialization but should be complemented with careful RL tuning and reward shaping.

Next improvements
- Add gym.Env compatibility and `observation_space` / `action_space` objects.
- Add stronger RL algorithm (PPO or SAC) and use stable-baselines3 for faster iteration.
- Expand demo dataset, add multiple players/difficulties, and add data augmentation.
