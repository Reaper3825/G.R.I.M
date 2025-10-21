import sys
import json
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import ReplayBuffer
from pathlib import Path
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = Path(r"D:\G.R.I.M\resources\models\ppo_model.zip")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_BUFFER = 512
RETRAIN_STEPS = 1024

# =====================================================
# MODEL & BUFFER SETUP
# =====================================================
try:
    model = PPO.load(MODEL_PATH, device=DEVICE)
    print(json.dumps({"status": "ready", "device": DEVICE}), flush=True)
except Exception:
    from train_ppo import model  # fallback: build new PPO model
    model.save(MODEL_PATH)
    print(json.dumps({"status": "new_model_created"}), flush=True)

# Dummy environment for PPO shape compatibility
class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.action_space = Discrete(8)
    def reset(self, *, seed=None, options=None):
        return np.zeros(5, dtype=np.float32), {}
    def step(self, action):
        return np.zeros(5, dtype=np.float32), 0.0, True, False, {}

env = DummyVecEnv([lambda: DummyEnv()])
buffer = ReplayBuffer(
    buffer_size=MAX_BUFFER,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=DEVICE,
    handle_timeout_termination=True
)

# =====================================================
# HELPERS
# =====================================================
def convert_state_dict(state_dict):
    """Convert GRIM CommandResult dict → numeric observation vector."""
    return np.array([
        hash(state_dict.get("command", "")) % 1e6 / 1e6,
        float(state_dict.get("success", False)),
        hash(state_dict.get("category", "")) % 1e6 / 1e6,
        hash(state_dict.get("mood", "")) % 1e6 / 1e6,
        0.0  # reserved for mean embedding
    ], dtype=np.float32)

def handle_batch(batch_data):
    """Process a batch of transitions for PPO fine-tuning."""
    for entry in batch_data:
        state = convert_state_dict(entry["state"])
        reward = float(entry.get("reward", 0.0))
        next_state = state
        done = False
        buffer.add(state, next_state, [0], [reward], [done], [0.0])
    maybe_retrain()

def maybe_retrain():
    """Retrain PPO when buffer full."""
    if buffer.size() < MAX_BUFFER:
        return
    model.policy.optimizer.zero_grad()
    model.learn(total_timesteps=RETRAIN_STEPS, progress_bar=False)
    model.save(MODEL_PATH)
    buffer.clear()
    print(json.dumps({"train": "updated_model"}), flush=True)

# =====================================================
# MAIN LOOP
# =====================================================
for line in sys.stdin:
    try:
        data = json.loads(line.strip())

        # --- Batch input from GRIM ---
        if "batch" in data:
            handle_batch(data["batch"])
            print(json.dumps({"status": f"processed_batch_{len(data['batch'])}"}), flush=True)
            continue

        # --- Single transition fallback ---
        if "transition" in data:
            handle_batch([data["transition"]])
            print(json.dumps({"status": "processed_single"}), flush=True)
            continue

        # --- Inference request ---
        obs = data.get("obs")
        commands = data.get("commands", [])
        embeddings = data.get("embeddings", {})

        if obs is not None and len(commands) > 0:
            # Dynamically resize action space
            n_actions = len(commands)
            model.policy.action_space = Discrete(n_actions)

            # Build observation vector
            obs_vec = np.array(list(obs.values()), dtype=np.float32).reshape(1, -1)

            # Add mean embedding if provided
            if embeddings:
                mean_emb = np.mean(list(embeddings.values()), dtype=np.float32)
                if obs_vec.shape[1] == 4:
                    obs_vec = np.concatenate([obs_vec, [[mean_emb]]], axis=1)
                else:
                    obs_vec[0, -1] = mean_emb

            # Predict action
            action, _ = model.predict(obs_vec, deterministic=True)
            idx = int(action)
            suggested = commands[idx] if 0 <= idx < len(commands) else None

            print(json.dumps({
                "action": idx,
                "suggested_command": suggested
            }), flush=True)
            continue

        # --- Invalid request handler ---
        print(json.dumps({"error": "invalid_request"}), flush=True)

    except Exception as e:
        print(json.dumps({"error": str(e)}), flush=True)
