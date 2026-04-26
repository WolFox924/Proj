import os
import json
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from custom_env_upconf import SmartCityGymEnv
from typing import Callable

# Diminution linéaire du learning rate au fil du temps
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

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
RUN_NAME = rl_cfg.get("run_name", "PPO_SmartCity")
print(f"Création du modèle PPO (LR={rl_cfg['learning_rate']})")
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=linear_schedule(rl_cfg["learning_rate"]),
    batch_size=rl_cfg["batch_size"],
    n_steps=rl_cfg["n_steps"],
    tensorboard_log=tensorboard_log_dir
)
print("Début de l'entraînement: Expérience '{RUN_NAME}'")
model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=RUN_NAME)
model_dir = current_file / "models" / "PPO"
os.makedirs(model_dir, exist_ok=True)
model.save(str(model_dir / f"smart_city_{RUN_NAME}"))
print(f"Entraînement terminé ! Modèle sauvegardé dans {model_dir} sous le nom smart_city_{RUN_NAME}.zip")