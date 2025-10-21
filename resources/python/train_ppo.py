import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = Path(r"D:\G.R.I.M\resources\models\ppo_model.zip")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# GRIM Synthetic Environment (Dynamic)
# =====================================================
class GrimEnv(Env):
    """
    Synthetic RL environment emulating GRIM's reward space.

    Observation vector:
        [ command_hash, success_flag, category_hash, mood_hash, mean_embedding ]

    Action space:
        Discrete(n_actions)  # dynamically set

    Reward:
        Simulates gradient toward consistent command success.
    """
    def __init__(self, n_actions: int = 16):
        super().__init__()
        self.n_actions = n_actions
        self.observation_space = Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.action_space = Discrete(self.n_actions)
        self.state = np.zeros(5, dtype=np.float32)
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(-1, 1, size=5).astype(np.float32)
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        self.step_count += 1

        # Base reward — simulated success correlation
        base_reward = 1.0 - abs(self.state[0] - self.state[2])
        noise = np.random.uniform(-0.1, 0.1)
        reward = base_reward + noise

        # Update success flag proportionally
        self.state[1] = np.clip(self.state[1] + (reward * 0.05), -1.0, 1.0)

        # Minor embedding drift to emulate semantic diversity
        self.state[4] = np.clip(self.state[4] + np.random.uniform(-0.05, 0.05), -1.0, 1.0)

        done = self.step_count >= 32
        truncated = False
        info = {}
        return self.state, float(reward), done, truncated, info


# =====================================================
# TRAINING PIPELINE
# =====================================================
def main(n_actions: int = 16):
    """
    Trains PPO on the synthetic GRIM environment.
    You can override n_actions based on live command count.
    """
    env = DummyVecEnv([lambda: GrimEnv(n_actions=n_actions)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=DEVICE,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=128,
        ent_coef=0.01,
        n_epochs=10
    )

    print(f"🚀 Training PPO model on device: {DEVICE} with {n_actions} actions")
    model.learn(total_timesteps=100000)

    model.save(MODEL_PATH)
    print(f"✅ PPO model trained and saved to {MODEL_PATH}")


# =====================================================
# OPTIONAL ENTRY POINT
# =====================================================
if __name__ == "__main__":
    # You can pass custom action count here (e.g., 20 commands)
    main(n_actions=16)
