import streamlit as st
import json
import subprocess
import os
from pathlib import Path

current_dir = Path(__file__).resolve().parent
CONFIG_FILE = current_dir / "config.json"
TRAIN_SCRIPT = current_dir / "train.py"
LOG_FILE = current_dir / "logs.txt"

def load_config():
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

# --- INTERFACE WEB ---
st.set_page_config(page_title="Smart City RL Control", layout="wide")
st.title("Smart City RL - Centre de Contrôle")

try:
    config = load_config()
except Exception as e:
    st.error(f"Impossible de lire config.json : {e}")
    st.stop()

# --- 1. ÉDITEUR DE CONFIGURATION ---
st.header("Configuration de l'environnement")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Simulation")
    config["environment"]["duration"] = st.number_input("Durée de l'Épisode (s)", value=config["environment"]["duration"], step=1)
    config["environment"]["step_duration"] = st.number_input("Durée du Step (s)", value=config["environment"]["step_duration"], step=1)
    config["environment"]["total_rps"] = st.number_input("Trafic (Total RPS)", value=config["environment"]["total_rps"], step=1)

with col2:
    st.subheader("Récompense (SLA vs Énergie)")
    config["reward"]["alpha_sla"] = st.number_input("Poids SLA (Alpha)", value=config["reward"]["alpha_sla"], step=0.1, format="%.2f")
    config["reward"]["beta_power"] = st.number_input("Poids Énergie (Beta)", value=config["reward"]["beta_power"], step=0.1, format="%.2f")
    config["reward"]["max_power_estimate"] = st.number_input("Estimation Puissance Max (W)", value=config["reward"]["max_power_estimate"], step=10.0)

with col3:
    st.subheader("Agent PPO (IA)")
    config["agent_rl"]["run_name"] = st.text_input("Nom de l'expérience (TensorBoard)", value=config["agent_rl"].get("run_name", "PPO_Run"))
    config["agent_rl"]["epochs"] = st.number_input("Nombre d'Épisodes (Epochs)", value=config["agent_rl"]["epochs"], step=10)
    config["agent_rl"]["learning_rate"] = st.number_input("Learning Rate", value=config["agent_rl"]["learning_rate"], format="%.5f")
    config["agent_rl"]["batch_size"] = st.number_input("Batch Size", value=config["agent_rl"]["batch_size"], step=16)

if st.button("Sauvegarder la configuration"):
    save_config(config)
    st.success("Configuration sauvegardée avec succès dans config.json !")

st.markdown("---")

# --- 2. CONTRÔLE DE L'ENTRAÎNEMENT ---
st.header("Contrôle du Serveur")
col_run, col_stop, col_tb = st.columns(3)

with col_run:
    if st.button("Lancer l'entraînement"):
        # On utilise sys.executable pour être sûr d'utiliser le bon environnement virtuel (.venv)
        import sys
        subprocess.Popen([sys.executable, str(TRAIN_SCRIPT)])
        st.success("Entraînement démarré en arrière-plan !")

with col_stop:
    if st.button("Arrêter l'entraînement"):
        os.system("pkill -f train.py")
        st.warning("Processus d'entraînement arrêté de force.")

with col_tb:
    st.info("Pour voir TensorBoard, ouvre le port 6006.")

st.markdown("---")

# --- 3. MONITORING EN DIRECT (LOGS) ---
st.header("Logs de la simulation (Rafraîchir pour mettre à jour)")

if st.button("Rafraîchir les logs"):
    pass 

if LOG_FILE.exists():
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-30:] # Affiche les 30 dernières lignes
        logs_text = "".join(lines)
        st.code(logs_text, language="text")
else:
    st.write("Aucun fichier logs.txt trouvé. Lancez un entraînement.")