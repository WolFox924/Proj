# Serveur
## Interface de contrôle (Streamlit)
```bash
nohup python -m streamlit run dashboard.py --server.address 0.0.0.0 --server.headless true > streamlit.log 2>&1 &
```

## Graphiques d'apprentissage (TensorBoard)
```bash
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