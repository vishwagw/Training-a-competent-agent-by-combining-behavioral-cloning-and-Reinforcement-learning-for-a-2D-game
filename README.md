# 2D Shooter Prototype (pygame)

This is an initial prototype for a simple 2D shooter implemented in Python using pygame.

Features included in this scaffold:
- Window and game loop (800x600, 60 FPS)
- Player rectangle controlled by WASD or arrow keys
- Shooting bullets with Space (simple rate limit)
- Minimal smoke test

Controls
- Move: WASD or Arrow keys
- Fire: Space
- Quit: ESC or close window

Quick start (Windows PowerShell)

```powershell
python -m pip install -r requirements.txt
python main.py
```

Run tests

```powershell
python -m pip install -r requirements.txt
pytest -q
```

Notes
- This is a starting point. Next steps: enemies, collisions, scoring, audio, and basic UI.
