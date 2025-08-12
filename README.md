# Geometry Dash RL Bot

A reinforcement learning-based bot (**PPO**, `Stable-Baselines3`) for automatic gameplay in Geometry Dash via screen capture and key press emulation.

## 📂 Project Structure

project_root/
│
├── run.py # Unified CLI entry: train / resume / play
├── requirements.txt # Python dependencies
│
├── configs/
│ ├── init.py
│ └── capture_config.py # Dataclass with screen capture configuration
│
├── envs/
│ ├── init.py
│ ├── gd_env_soft.py # Fixed environment variant (recommended)
│ └── gd_env_strict.py # Original variant (with bug and logs)
│
├── training/
│ ├── init.py
│ ├── callbacks.py # AutoSaveCallback
│ ├── window.py # Geometry Dash window focusing function
│ ├── train.py # Train model from scratch
│ ├── resume.py # Continue training from a checkpoint
│ └── play.py # Run the game with the model (inference)
│
├── templates/
│ └── attempt.png # Template for death detection ("ATTEMPT")
│
└── models/ # Folder for saved models (.zip)

---

## ⚡ Installation

1. Install Python **3.9–3.13** (recommended).
2. Install dependencies:
    pip install -r requirements.txt
3. Ensure **Geometry Dash** is installed and runs in windowed mode (Recommended window size - width 225 height 127).

---

## 🛠 Environment Options

- **Soft** — fixed environment variant (**recommended**)
    Fixes the bug: after death detection, the template detector (CV) is completely disabled for 2 seconds and does not trigger again.

- **Strict** — historical environment variant
    Contains the original logic with frequent logging and the repeated ATTEMPT detection bug.

---

## 🚀 Running

### Train from scratch (Soft - Recommended)

python run.py --mode train --env soft --left x --top x --width x --height x --timesteps 300000 --fps 30

### Continue training from a checkpoint

python run.py --mode resume --env soft --left x --top x --width x --height x --timesteps 100000 --model models/gd_live_ppo.zip

### Play with the model

python run.py --mode play --env soft --left x --top x --width x --height x --model models/gd_live_ppo.zip

---

## 📌 Launch Arguments

| Parameter            | Description |
|----------------------|----------|
| `--mode`             | `train` — train from scratch, `resume` — continue, `play` — play |
| `--env`              | `soft` — bug fix, `strict` — original |
| `--left` / `--top`   | Offset of the capture window from the left/top edge of the screen |
| `--width` / `--height` | Capture area size |
| `--fps`              | Target capture FPS |
| `--timesteps`        | Number of steps for training/continuation |
| `--model`            | Path to the `.zip` model file |
| `--lr`               | Learning rate (for resume) |
| `--n_steps`          | Steps per batch (for resume) |
| `--batch_size`       | Batch size (for resume) |
| `--save_freq`        | Model autosave frequency |

---

## 📷 ATTEMPT Template Preparation (Not required if the window is launched with width 225 height 127 since such a template already exists)

1. Take a screenshot when the word `ATTEMPT` appears on the screen.
2. Save it as  `templates/attempt.png`.
3. The resolution must match what the code uses for template matching.

---

## ⚠ Важные замечания

- 📌 The **Soft** environment fixes the bug where, after death, the bot could immediately see `ATTEMPT` again after restart and loop.
- 🖥 On Windows, run the console as administrator for proper `keyboard` and `pyautogui` operation.
- 🎯 Disable frequent `print()` calls to increase FPS (strict has detailed logs; soft keeps logging minimal).
- 🧠 The model uses `PPO` from `Stable-Baselines3` with a CNN policy and a stack of the last 4 frames.

---

## 📜 License
MIT License — free use and modification.
