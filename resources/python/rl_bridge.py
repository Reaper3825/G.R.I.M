import sys
import json
import torch
from stable_baselines3 import PPO

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = r"D:\G.R.I.M\resources\models\ppo_model.zip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# LOAD MODEL
# =====================================================
try:
    model = PPO.load(MODEL_PATH, device=DEVICE)
    print(json.dumps({"status": "ready", "device": DEVICE}), flush=True)
except Exception as e:
    print(json.dumps({"error": str(e)}), flush=True)
    sys.exit(1)

# =====================================================
# MAIN LOOP
# =====================================================
for line in sys.stdin:
    try:
        data = json.loads(line.strip())
        obs = data.get("obs", None)
        if obs is None:
            print(json.dumps({"error": "missing obs"}), flush=True)
            continue

        action, _ = model.predict(obs, deterministic=True)
        print(json.dumps({"action": int(action)}), flush=True)

    except Exception as e:
        print(json.dumps({"error": str(e)}), flush=True)
