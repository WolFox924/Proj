# Serveur
## Interface de contrôle (Streamlit) & Graphiques d'apprentissage (TensorBoard)
```bash
cd /home/thao/project/serverless-edge/Thao/scripts/agent_rl
source .venv/bin/activate.fish
nohup python -m streamlit run dashboard.py --server.address 0.0.0.0 --server.headless true > streamlit.log 2>&1 &
nohup python -m tensorboard.main --logdir ./logs_tensorboard/ --bind_all > tensorboard.log 2>&1 &
```

## Logs
```bash
cat streamlit.log
cat tensorboard.log
```

## Kill processes
```bash
pkill -f streamlit
pkill -f tensorboard
```