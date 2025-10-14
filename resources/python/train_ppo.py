from stable_baselines3 import PPO
import gymnasium as gym

# Create environment
env = gym.make("CartPole-v1")

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1, device="cpu")  # GPU optional

# Train model
model.learn(total_timesteps=50000)

# Save model to GRIM resources folder
model.save("D:/G.R.I.M/resources/models/ppo_model")

print("✅ PPO model trained and saved successfully.")
