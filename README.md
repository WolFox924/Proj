# Serverless Edge Simulation using faas-sim
Our project is to create and compare serverless egde strategies in application for a model of a smart city.

*In order to read faas-sim readme, you can find it at the root of the project as **README_faas.md***

## State of the art :
We compared our strategies to existing ones created by [Sayah Mahfoud](https://github.com/M4hf0d/faas-sim) (LPLT, HPST, K8s, STD_First) within [Philipp Raith](https://scholar.google.com/citations?user=foXpSlIAAAAJ&hl=en) simulations and Smart city topology.

## First semester reseaches :
We created our first strategy APE-H, our first compromise between satisfying services but with the lowest power consumption (*ext/mhfd/scaling/strategies/AdaptivePerformanceEnergyHybrid.py*).
And simulated it with the other created strategies in order to compare the throughout different plots using matplotlib.

### Usage :
From the root of the project : 
```shell
cd Thao
python -m strat_to_plot
```
Here you can type the number of the script you want to use for running the simulations. run_comp_raith is our last work, make sure to use it for an optimal experience.

It will then export the results as csv files directly inside *Thao/output/scripts_data* folder as another folder per strategy.

Then you can type another number to choose which creating plot script you want. 
- *plot_results_raith.py* creates the default plots (Energy, Latency and Scaling) saved in *Thao/output*
- *plot_results_raith2.py* creates the advanced plots (Cumulative Latency Distribution, Paretos ,Scaling decisions and Replicas over time) saved in *Thao/output/advanced_plots*



## Second semester reseaches :
Our goal was to create another strategy based on reinforcement learning to create a strategy which finds a compromise between satisfying services but with the lowest power consumption.

The RL agent (Reinforcement Learning) works with 3 different **.py files**:
- *Thao/scripts/agent_rl/train.py* 

which relies on stable_baselines3's PPO agent using our custom environment for learning.

- *Thao/scripts/agent_rl/custom_env.py* 

the environment with the same benchmark as previously, the same functions deployment, where the different steps are defined, reward and logging inside *Thao/scripts/logs.txt*

- *Thao/scripts/agent_rl/evaluates.py*

transforming the .zip file from the learning and export them as csv file for comparison like in the first semester.

### Steps: 

#### - Actions :

In *custom_env_1.py* we have 6 different steps :
- 0 : scales down
- 1 : does nothing
- 2 : scales up slowly
- 2 : scales up normaly
- 2 : scales up faster
- 2 : scales up very fast

In *custom_env.py* (in development) we have 6 different steps :
- -8 : scales ultra 5 (8 instances)
- -7 : scales ultra 4 (7 instances)
- -6 : scales ultra 3 (6 instances)
- -5 : scales ultra 2 (5 instances)
- -4 : scales ultra 1 (4 instances)
- -3 : scales high (3 instances)
- -2 : scales medium (2 instances)
- -1 : scales low (1 instances)
- 0 : does nothing
- 1 : scales up low (1 instances)
- 2 : scales up medium (2 instances)
- 3 : scales up high (3 instances)
- 4 : scales up ultra 1 (4 instances)
- 5 : scales up ultra 2 (5 instances)
- 6 : scales up ultra 3 (6 instances)
- 7 : scales up ultra 4 (7 instances)
- 8 : scales up ultra 5 (8 instances)

#### - Rewards :
- **Latency**

The **latency** works as an SLA score (Service level agreement), each function has an expected maximum latency. 
```python
EXPECTED_MAX = {'resnet50-inference': 2.5, 'speech-inference': 2.0, 'resnet50-preprocessing': 1.5, 'resnet50-training': 15.0, 'python-pi': 2.20, 'fio': 1.8}
```
We calculate a ratio between the actual amount of time to treat the function and the expected max. If the ratio is over 1 (time exceeded), the score = 1 - ratio, and if it's under the score is 1.
```python
ratio = float(row[exec_col]) / EXPECTED_MAX.get(base_func, 2.0)
# Dans les temps -> +1.0 | En retard -> Pénalité
if ratio <= 1.0 :
    score = 1.0 
else :
    score = (1.0 - ratio)
score = max(score, -1.0) 
# Limite basse de sécurité pour un peu centrer, on peut la mettre plus bas comme ça on lui inflige plus de pénalité. Donc pas vraiment centré et normalisé mais un peu pareil
sla_scores.append(score)
```

And after, inside sla_scores the final score is an average of all the previous scores.

- **Energy**

For the energy, only the current power of each node is extracted and we take the ratio between the maximum power estimate and the current power (capped by 1 for normalization).
```python
# Si 120 nœuds tournent à fond, on est autour de 600W-800W selon les CSV
MAX_POWER_ESTIMATE = 800.0 
normalized_power = min(current_power / MAX_POWER_ESTIMATE, 1.0)
```

For now the reward is 50% for latency and 50% for power 
```python
# Reward final (50% SLA / 50% Énergie)
ALPHA = 0.5
BETA = 0.5
reward = (ALPHA * avg_sla_score) - (BETA * normalized_power)
```
Calculated like that because an average sla score is good, but a high normalized power isn't.

### Usage :
From the root of the project : 
```shell
cd Thao/scripts
# can take a long depending on the number of timesteps set in the script
python agent_rl/train.py
```
After the simulations it creates *Thao/scripts/models/PPO/smart_city_ppo.zip* which has to be extracted into csv files for comparison with
```shell
python agent_rl/evaluate.py 
```