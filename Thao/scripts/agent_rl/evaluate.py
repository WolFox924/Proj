import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from custom_env import SmartCityGymEnv
from pathlib import Path
import os
import shutil

#configuration des chemins
current_file = Path(__file__).resolve().parent
project_root = current_file.parent.parent.parent
output_dir = project_root / "Thao" / "output" / "scripts_data" / "batch_RL_PPO"
#nettoyage pour être sûr
if output_dir.exists():
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

print("Chargement du modèle")
model_path = project_root / "Thao" / "scripts" / "models" / "PPO" / "smart_city_ppo.zip" # Adapte si besoin

#si le modèle n'existe pas
if not os.path.exists(model_path):
    print(f"Modèle introuvable à {model_path}. Vérifie le chemin.")
    exit()

#on ouvre le zip
model = PPO.load(model_path)
env = SmartCityGymEnv()
print("Lancement de l'évaluation")
obs, _ = env.reset()
done = False
steps = 0

#pour se souvenir des choix
rl_scaling_decisions = []
while not done:
    action, _ = model.predict(obs, deterministic=True)
    
    # on enregistre l'action avant de l'envoyer à l'environnement
    for i, act in enumerate(action):
        fname = env.functions[i]
        if act == 0:
            rl_scaling_decisions.append({'t': env.sim_env.now, 'function': fname, 'decision': 'scale_down', 'replicas_change': -1})
        elif act == 1:
            rl_scaling_decisions.append({'t': env.sim_env.now, 'function': fname, 'decision': 'does_nothing', 'replicas_change': 0})
        elif act == 2:
            rl_scaling_decisions.append({'t': env.sim_env.now, 'function': fname, 'decision': 'scale_up_low', 'replicas_change': 1})
        elif act == 3:
            rl_scaling_decisions.append({'t': env.sim_env.now, 'function': fname, 'decision': 'scale_up_medium', 'replicas_change': 2})
        elif act == 4:
            rl_scaling_decisions.append({'t': env.sim_env.now, 'function': fname, 'decision': 'scale_up_high', 'replicas_change': 3})
        elif act == 5:
            rl_scaling_decisions.append({'t': env.sim_env.now, 'function': fname, 'decision': 'scale_up_ultra', 'replicas_change': 4})

    obs, reward, done, truncated, info = env.step(action)
    steps += 1
    #DEBUG : on vérifie que le temps avance et montre les décisions de l'IA
    if steps % 5 == 0:
        print(f"   -> Simulation time: {env.sim_env.now:.1f}s / {env.duration}s | Actions IA : {action}")

print("Extraction des résultats")
sim = env.sim_env # On récupère l'objet sim interne


# extraction, invocation det exportation des résultats en csv
# Invocations
try:
    inv_df = sim.metrics.extract_dataframe("invocations")
    if inv_df is not None and not inv_df.empty:
        # Renommage des colonnes pour correspondre à tes scripts de plot
        if 'execution_time' in inv_df.columns: inv_df['t_exec'] = inv_df['execution_time']
        if 't' in inv_df.columns: inv_df['t_start'] = inv_df['t']
        
        inv_df.to_csv(output_dir / "invocations_df.csv", index=False)
        print(f"Invocations exportées ({len(inv_df)} lignes)")
    else:
        print("invocations_df est vide (tout éteint ?)")
except Exception as e:
    print(f"Erreur invocations: {e}")

# Power
try:
    pwr_df = sim.metrics.extract_dataframe("power")
    if pwr_df is not None and not pwr_df.empty:
        pwr_df.to_csv(output_dir / "power_df.csv", index=False)
        print(f"Power exporté ({len(pwr_df)} lignes)")
    else:
        print("power_df est vide")
except Exception as e:
    print(f"Erreur power: {e}")

# Energy
try:
    if pwr_df is not None and not pwr_df.empty:
        # On calcule l'énergie totale (en Joules) pour chaque nœud
        # Puisque le relevé se fait chaque seconde, Somme des Watts = Total des Joules
        energy_data = []
        for node in pwr_df['node'].unique():
            node_df = pwr_df[pwr_df['node'] == node]
            total_energy_joules = node_df['power_watts'].sum()
            node_type = node_df['node_type'].iloc[0] if 'node_type' in node_df.columns else "unknown"
            
            energy_data.append({
                'node': node, 
                'node_type': node_type,
                'energy_joules': total_energy_joules
            })
            
        energy_df = pd.DataFrame(energy_data)
        energy_df.to_csv(output_dir / "energy_df.csv", index=False)
        print(f"Energy exporté ({len(energy_df)} nœuds calculés depuis power_df)")
    else:
        print("Impossible de générer energy_df car power_df est vide")
except Exception as e:
    print(f"Erreur lors de la génération de l'énergie : {e}")

# Scaling
try:
    if len(rl_scaling_decisions) > 0:
        scale_df = pd.DataFrame(rl_scaling_decisions)
        scale_df.to_csv(output_dir / "scaling_decisions_df.csv", index=False)
        print(f"Scaling exporté ({len(scale_df)} décisions prises par l'IA)")
    else:
        print("Pas de décisions de scaling enregistrées")
except Exception as e:
    print(f"Erreur scaling: {e}")

print(f"Tout a été exporté et est prêt à être comparé avec les autres stratégies : {output_dir}")