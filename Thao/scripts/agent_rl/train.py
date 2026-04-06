import os
import json
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from custom_env import SmartCityGymEnv

current_file = Path(__file__).resolve().parent
project_root = current_file.parent.parent.parent
config_path = current_file / "config.json"

with open(config_path, 'r') as f:
    config = json.load(f)

env_cfg = config["environment"]
rl_cfg = config["agent_rl"]

steps_per_epoch = int(env_cfg["duration"] / env_cfg["step_duration"])
TOTAL_TIMESTEPS = rl_cfg["epochs"] * steps_per_epoch #epochs * steps

print(f"Configuration chargée : {rl_cfg['epochs']} epochs x {steps_per_epoch} steps = {TOTAL_TIMESTEPS} Timesteps.")
tensorboard_log_dir = str(current_file / "logs_tensorboard")
os.makedirs(tensorboard_log_dir, exist_ok=True)
env = make_vec_env(lambda: SmartCityGymEnv(config_path=str(config_path)), n_envs=1)
print("Création du modèle PPO...")
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=rl_cfg["learning_rate"],
    batch_size=rl_cfg["batch_size"],
    n_steps=rl_cfg["n_steps"],
    tensorboard_log=tensorboard_log_dir
)
print("Début de l'entraînement...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="PPO_SmartCity")
model_dir = current_file / "models" / "PPO"
os.makedirs(model_dir, exist_ok=True)
model.save(str(model_dir / "smart_city_ppo"))
print(f"Entraînement terminé ! Modèle sauvegardé dans {model_dir}")