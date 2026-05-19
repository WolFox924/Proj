import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import random
import json
from pathlib import Path

#Arborescence pour qu'il s'y retrouve
current_file = Path(__file__).resolve().parent
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))

from sim.core import Environment
from sim.docker import ContainerRegistry
from sim.faas.system import DefaultFaasSystem
from sim.faassim import Simulation
from sim.logging import SimulatedClock, RuntimeLogger
from sim.metrics import Metrics, PowerMetrics
from sim.skippy import SimulationClusterContext
from skippy.core.scheduler import Scheduler
from skippy.core.storage import StorageIndex
from sim.faas import FunctionState

from ext.raith21.functionsim import AIPythonHTTPSimulatorFactory
from ext.raith21.generator import generate_devices
from ext.raith21.generators.edgeai import make_edgeai_settings
from ext.raith21.oracles import Raith21ResourceOracle, Raith21FetOracle
from ext.raith21.etherdevices import convert_to_ether_nodes
from ext.raith21.topology import urban_sensing_topology
from ext.raith21.util import vanilla
from ext.raith21.characterization import get_raith21_function_characterizations
from ext.raith21.fet import ai_execution_time_distributions
from ext.raith21.resources import ai_resources_per_node_image
from ext.raith21.predicates import CanRunPred, NodeHasAcceleratorPred, NodeHasFreeGpu, NodeHasFreeTpu
from ext.mhfd.deployments import create_smart_city_deployments
from ext.mhfd.scbenchmark import create_smart_city_constant_benchmark
from ext.mhfd.power import Raith21PowerOracle, DEVICE_POWER_PROFILES, power_monitoring_loop

class SmartCityGymEnv(gym.Env):

    #Initialisation avant chaque début de learning
    def __init__(self, config_path="config.json"):
        super(SmartCityGymEnv, self).__init__()
        
        # --- CHARGEMENT CONFIGURATION ---
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        env_cfg = self.config["environment"]
        self.num_devices = env_cfg["num_devices"]
        self.n_nuc = env_cfg["n_nuc"]
        self.duration = env_cfg["duration"]
        self.step_duration = env_cfg["step_duration"]
        self.total_rps = env_cfg["total_rps"] 
        
        self.device_settings = make_edgeai_settings(self.num_devices, self.n_nuc)
        self.functions = ['resnet50-inference', 'speech-inference', 'resnet50-preprocessing', 'resnet50-training', 'python-pi', 'fio']
        self.n_funcs = len(self.functions)
        
        # CORRECTION ACTION SPACE : 17 possibilités (de 0 à 16). Dans step, on fera action - 8 pour avoir de -8 à +8 !
        self.action_space = spaces.MultiDiscrete([17] * self.n_funcs)
        self.observation_space = spaces.Box(low=0, high=99999, shape=(1 + self.n_funcs,), dtype=np.float32) 
        self.sim_env = None
        self.sim_time = 0
        self.current_episode = 0
        self.max_episodes = self.config["agent_rl"]["epochs"]
        
        #Vide le fichier de logs au lancement de l'entraînement
        open('logs.txt', 'w', encoding='utf-8').close()

    def reset(self, seed=None, options=None):
        #Tiré de d'autres endroits pour la définition de l'environnement 
        super().reset(seed=seed)
        self.current_episode += 1        
        np.random.seed(1435)
        random.seed(1435)
        devices = generate_devices(self.num_devices, self.device_settings)
        ether_nodes = convert_to_ether_nodes(devices)
        storage_index = StorageIndex()
        topology = urban_sensing_topology(ether_nodes, storage_index)
        fet_oracle = Raith21FetOracle(ai_execution_time_distributions)
        resource_oracle = Raith21ResourceOracle(ai_resources_per_node_image)
        power_oracle = Raith21PowerOracle(DEVICE_POWER_PROFILES)
        predicates = [CanRunPred(fet_oracle, resource_oracle), NodeHasAcceleratorPred(), NodeHasFreeGpu(), NodeHasFreeTpu()]
        predicates.extend(Scheduler.default_predicates)
        sched_params = {"percentage_of_nodes_to_score": 100, "priorities": vanilla.get_priorities(), "predicates": predicates}
        #Environnement
        env = Environment()
        env.simulator_factory = AIPythonHTTPSimulatorFactory(get_raith21_function_characterizations(resource_oracle, fet_oracle))
        env.metrics = Metrics(env, log=RuntimeLogger(SimulatedClock(env)))
        env.topology = topology
        env.faas = DefaultFaasSystem(env)
        env.container_registry = ContainerRegistry()
        env.storage_index = storage_index
        env.cluster = SimulationClusterContext(env)
        env.scheduler = Scheduler(env.cluster, **sched_params)
        env.power_oracle = power_oracle
        env.power_metrics = PowerMetrics()
        env.background_processes.append(power_monitoring_loop)
        self.sim_env = env
        #Creation de la smart city
        create_smart_city_deployments(self.sim_env.faas, resource_oracle)
        #Benchmark
        benchmark = create_smart_city_constant_benchmark(999999, self.total_rps, "none")
        self.sim = Simulation(topology, benchmark, env=self.sim_env)
        self.sim.init_environment(self.sim_env)
        self.sim_env.process(self.sim_env.resource_monitor.run())
        benchmark.setup(self.sim_env)
        self.sim_env.faas.start()
        self.sim_env.process(benchmark.run(self.sim_env))
        
        #déploiement
        for fname in self.functions:
            if fname in self.sim_env.faas.functions_deployments:
                self.sim_env.process(self.sim_env.faas.scale_up(fname, 2))        
        self.sim_time = 0
        
        #mode append
        with open('logs.txt', 'a', encoding='utf-8') as fichier:
            print(f"\n======================================", file=fichier)
            print(f"=== SIMULATION (ÉPISODE) {self.current_episode:3} / {self.max_episodes} ===", file=fichier)
            print(f"======================================", file=fichier)
            
        return self._get_obs(), {}

    def step(self, actions):
        for i, act_raw in enumerate(actions):
            # Transformation de l'action [0, 16] en [-8, 8]
            act = act_raw - 8
            
            fname = self.functions[i]
            if fname not in self.sim_env.faas.functions_deployments:
                continue
            current_replicas = len(self.sim_env.faas.get_replicas(fname))
            try:
                # (Scale Down)
                if act < 0:
                    pods_to_remove = abs(act) # Transforme -3 en 3
                    
                    if current_replicas >= pods_to_remove:
                        # (Scale to Zero autorisé)
                        self.sim_env.process(self.sim_env.faas.scale_down(fname, pods_to_remove))
                    elif current_replicas > 0:
                        # S'il demande à enlever 8 pods mais qu'il n'en reste que 3, on enlève les 3 derniers
                        self.sim_env.process(self.sim_env.faas.scale_down(fname, current_replicas))
                # (Scale Up)
                elif act > 0:
                    # On s'assure de ne pas dépasser 15 pods maximum
                    if current_replicas + act <= 15:
                        self.sim_env.process(self.sim_env.faas.scale_up(fname, act))
                    elif current_replicas < 15:
                        # Si on demande trop, on remplit juste jusqu'à la limite de 15
                        pods_to_add = 15 - current_replicas
                        self.sim_env.process(self.sim_env.faas.scale_up(fname, pods_to_add))
                # Si act == 0, on ne fait rien (Maintien de l'état)
            except Exception:
                pass

        self.sim_time += self.step_duration
        self.sim_env.run(until=self.sim_time)

        # --------------------------------REWARD--------------------------------
        rew_cfg = self.config["reward"]
        EXPECTED_MAX = rew_cfg["expected_max"]
        metrics = self.sim_env.metrics.extract_dataframe('invocations')
        requests_processed = 0
        total_reqs = 0
        
        total_actual_latency = 0.0
        total_expected_latency = 0.0
        if metrics is not None and not metrics.empty:
            total_reqs = len(metrics)
            t_col = 't_start' if 't_start' in metrics.columns else ('t' if 't' in metrics.columns else None)
            exec_col = 'execution_time' if 'execution_time' in metrics.columns else ('t_exec' if 't_exec' in metrics.columns else None)
            if t_col:
                recent = metrics[metrics[t_col] >= (self.sim_env.now - self.step_duration)]
                requests_processed = len(recent)
                if requests_processed > 0 and exec_col:
                    for _, row in recent.iterrows():
                        func_name = row.get('function', row.get('deployment', 'python-pi'))
                        base_func = next((f for f in self.functions if f in func_name), 'python-pi')
                        total_actual_latency += float(row[exec_col])
                        total_expected_latency += EXPECTED_MAX.get(base_func, 2.0)

        q_reqs = len(self.sim_env.faas.request_queue.items) if hasattr(self.sim_env.faas, 'request_queue') else 0
        if requests_processed > 0 and total_expected_latency > 0:
            if total_actual_latency <= total_expected_latency:
                avg_sla_score = 1.0 # plus rapide que le budget autorisé
            else:
                # Si on dépasse, la note baisse doucement (ex: budget 10s, réel 15s -> score 10/15 = 0.66)
                avg_sla_score = total_expected_latency / total_actual_latency
        elif requests_processed == 0:
            if q_reqs == 0:
                avg_sla_score = 1.0 # Pas de requêtes, pas de pénalité
            else:
                avg_sla_score = 0.0 # Requêtes bloquées, note de zéro

        # calcul du Score Énergie
        current_power = 0.0
        exact_max_cluster_power = 0.0
        
        pwr_df = self.sim_env.metrics.extract_dataframe('power')
        if pwr_df is not None and not pwr_df.empty:
            if 'power_watts' in pwr_df.columns and 'timestamp' in pwr_df.columns:
                last_time = pwr_df['timestamp'].max()
                derniere_seconde_df = pwr_df[pwr_df['timestamp'] == last_time]
                current_power = float(derniere_seconde_df['power_watts'].sum()) 
                if 'node_type' in derniere_seconde_df.columns and "hardware_max_power" in self.config:
                    hw_limits = self.config["hardware_max_power"]
                    for node_type in derniere_seconde_df['node_type']:
                        exact_max_cluster_power += hw_limits.get(node_type, 5.0)            
            elif 'power_watts' in pwr_df.columns:
                current_power = float(pwr_df['power_watts'].iloc[-self.num_devices:].sum())
        if exact_max_cluster_power == 0.0:
            exact_max_cluster_power = rew_cfg.get("max_power_estimate", 800.0)
        # Normalisation Symétrique de l'Énergie (0 = Conso Max, 1 = Conso Zéro)
        power_ratio = min(current_power / exact_max_cluster_power, 1.0)
        power_score = 1.0 - power_ratio
        current_total_pods = sum(len(self.sim_env.faas.get_replicas(f)) for f in self.functions if f in getattr(self.sim_env.faas, 'functions_deployments', {}))
        active_pods = max(1, current_total_pods)
        efficiency = requests_processed / active_pods
        efficiency_score = min(efficiency / 5.0, 1.0) 

        ALPHA = rew_cfg.get("alpha_sla", 0.4)
        BETA = rew_cfg.get("beta_power", 0.4)
        GAMMA = rew_cfg.get("gamma_efficiency", 0.2) 

        reward = (ALPHA * avg_sla_score) + (BETA * power_score) + (GAMMA * efficiency_score)
        
        # =======================================================

        # Monitoring logs qu'on peut mettre dans un grafana après en vrai
        all_pods = 0
        ready_pods = 0
        for f in self.functions:
            if f in self.sim_env.faas.functions_deployments:
                all_pods += len(self.sim_env.faas.get_replicas(f))
                ready_pods += len(self.sim_env.faas.get_replicas(f, FunctionState.RUNNING))

        with open('logs.txt', 'a', encoding='utf-8') as fichier:
            print(
                f"- t={self.sim_time:3.0f}s | "
                f"reqs={requests_processed:3}/{total_reqs:4} | "
                f"q_reqs={q_reqs:4} | "
                f"pods={ready_pods:2}/{all_pods:2} | "
                f"sla={avg_sla_score:5.2f} | "
                f"pwr_norm={power_score:4.2f} | "
                f"efficiency={efficiency_score:4.2f} | "
                f"r={reward:6.2f}", 
                file=fichier
            )
        
        #fini ou pas
        terminated = self.sim_time >= self.duration
        return self._get_obs(), reward, terminated, False, {}

    #obtenir toutes les invocations
    def _get_obs(self):
        metrics = self.sim_env.metrics.extract_dataframe('invocations')
        avg_lat = 0.0
        #pareil que pour récupérer la latence (vraiment copié collé)
        if metrics is not None and not metrics.empty:
            t_col = 't_start' if 't_start' in metrics.columns else ('t' if 't' in metrics.columns else None)
            exec_col = 'execution_time' if 'execution_time' in metrics.columns else ('t_exec' if 't_exec' in metrics.columns else None)
            if t_col and exec_col:
                recent = metrics[metrics[t_col] >= (self.sim_env.now - 15)]
                if not recent.empty:
                    avg_lat = recent[exec_col].mean()
        
        avg_lat = 0.0 if np.isnan(avg_lat) else avg_lat
        replicas_counts = []
        for f in self.functions:
            count = len(self.sim_env.faas.get_replicas(f, FunctionState.RUNNING)) if hasattr(self.sim_env, 'faas') and f in self.sim_env.faas.functions_deployments else 0
            replicas_counts.append(count)
        return np.array([float(avg_lat)] + replicas_counts, dtype=np.float32)