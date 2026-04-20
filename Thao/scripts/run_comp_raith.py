#!/usr/bin/env python
# coding: utf-8
"""
FaaS Simulation - Batch Runner for All Strategies
Lance toutes les stratégies séquentiellement et sauvegarde les données brutes.
"""

import sys
import os
import logging
import random
import numpy as np
import pandas as pd
from pathlib import Path

# ================= CORRECTION PATH =================
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent.parent
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
# ===================================================

# Imports Simulation
from sim.core import Environment
from sim.docker import ContainerRegistry
from sim.faas.system import DefaultFaasSystem
from sim.faassim import Simulation
from sim.logging import SimulatedClock, RuntimeLogger
from sim.metrics import Metrics, PowerMetrics
from sim.skippy import SimulationClusterContext
from skippy.core.scheduler import Scheduler
from skippy.core.storage import StorageIndex

# Imports Raith21 & MHFD
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
from ext.mhfd.deployments import create_smart_city_deployments, create_custom_smart_city_deployments
from ext.mhfd.scbenchmark import create_smart_city_constant_benchmark
from ext.mhfd.power import Raith21PowerOracle, DEVICE_POWER_PROFILES, power_monitoring_loop
from ext.mhfd.autoscaler import create_heterogeneous_edge_autoscaler
from ext.mhfd.scaling.strategies.AdaptivePerformanceEnergyHybrid import AdaptivePerformanceEnergyHybrid

# --- CONFIGURATION DES STRATÉGIES À LANCER ---
STRATEGIES_TO_RUN = [
    ("Standard_First", "basic"),          
    ("Standard_K8s", "kubernetes"),  
    ("LPLT_Energy", "power"),             
    ("APE_H_Hybrid", "compromis"),       
    ("HPST_Performance", "performance")         
]

class SimulationConfig:
    """Configuration dynamique"""
    def __init__(self, strategy_name, internal_strategy_id):
        self.num_devices = 120
        self.n_nuc = 3
        self.device_settings = make_edgeai_settings(self.num_devices, self.n_nuc)
        self.duration = 200  
        self.total_rps = 35
        self.scenario = "none"
        
        self.strategy_name = strategy_name
        self.internal_strategy_id = internal_strategy_id
        self.settings_id = f"batch_{strategy_name}" 
        
        self.custom_counts = {
            "resnet50-inference": 1, "speech-inference": 1,
            "resnet50-preprocessing": 1, "resnet50-training": 1,
            "python-pi": 1, "fio": 1,
        }
        self.percentage_of_nodes_to_score = 100
        
        # Chemins de sortie
        base_dir = current_script_path.parent.parent  
        self.output_root = base_dir / "output"
        self.data_dir_base = self.output_root / "scripts_data"
        
        self.numpy_seed = 1435
        self.random_seed = 1435

def setup_logging():
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def setup_environment(topology, storage_index, sched_params, fet_oracle, resource_oracle, power_oracle, config):
    env = Environment()
    env.simulator_factory = AIPythonHTTPSimulatorFactory(
        get_raith21_function_characterizations(resource_oracle, fet_oracle))
    env.metrics = Metrics(env, log=RuntimeLogger(SimulatedClock(env)))
    env.topology = topology
    env.faas = DefaultFaasSystem(env)
    env.container_registry = ContainerRegistry()
    env.storage_index = storage_index
    env.cluster = SimulationClusterContext(env)
    env.scheduler = Scheduler(env.cluster, **sched_params)
    power_metrics = PowerMetrics()
    env.power_oracle = power_oracle
    env.power_metrics = power_metrics
    
    # SÉLECTION DE LA STRATÉGIE
    strat_id = config.internal_strategy_id
    print(f" ⚙️ Injection Stratégie : {config.strategy_name} (ID: {strat_id})")
    
    if strat_id == 'compromis':
        autoscaler = AdaptivePerformanceEnergyHybrid(env, env.faas, power_oracle)
    else:
        autoscaler = create_heterogeneous_edge_autoscaler(
            env, env.faas, power_oracle, strategy=strat_id
        )
    
    env.background_processes.append(power_monitoring_loop)
    env.background_processes.append(lambda env: autoscaler.run())
    return env

def run_single_strategy(strat_name, strat_id):
    """Exécute une simulation complète pour une stratégie donnée"""
    print(f"\n{'=' * 60}")
    print(f"DÉMARRAGE SIMULATION : {strat_name}")
    print(f"{'=' * 60}")
    
    config = SimulationConfig(strat_name, strat_id)
    setup_logging()
    
    np.random.seed(config.numpy_seed)
    random.seed(config.random_seed)
    
    # Génération TopologieQ
    devices = generate_devices(config.num_devices, config.device_settings)
    ether_nodes = convert_to_ether_nodes(devices)
    storage_index = StorageIndex()
    topology = urban_sensing_topology(ether_nodes, storage_index)
    
    fet_oracle = Raith21FetOracle(ai_execution_time_distributions)
    resource_oracle = Raith21ResourceOracle(ai_resources_per_node_image)
    power_oracle = Raith21PowerOracle(DEVICE_POWER_PROFILES)
    
    predicates = [CanRunPred(fet_oracle, resource_oracle), NodeHasAcceleratorPred(), NodeHasFreeGpu(),
                  NodeHasFreeTpu()]
    predicates.extend(Scheduler.default_predicates)
    sched_params = {"percentage_of_nodes_to_score": 100, "priorities": vanilla.get_priorities(),
                    "predicates": predicates}
    
    env = setup_environment(topology, storage_index, sched_params, fet_oracle, resource_oracle, power_oracle, config)
    
    print(" Déploiement des fonctions...")
    if config.scenario == "custom":
        create_custom_smart_city_deployments(env.faas, config.custom_counts.keys(), resource_oracle)
    else:
        create_smart_city_deployments(env.faas, resource_oracle)
    
    benchmark = create_smart_city_constant_benchmark(config.duration, config.total_rps, config.scenario)
    
    print(" ⏱Exécution...")
    sim = Simulation(topology, benchmark, env=env)
    sim.run()
    
    print(" Sauvegarde des résultats bruts...")
    dfs = {}
    for mt in ["invocations", "power", "energy", "scaling_decisions"]:
        try:
            dfs[f"{mt}_df"] = sim.env.metrics.extract_dataframe(mt)
        except:
            pass
            
    raw_dir = config.data_dir_base / f"{config.settings_id}"
    os.makedirs(raw_dir, exist_ok=True)
    
    for name, df in dfs.items():
        if df is not None: 
            df.to_csv(raw_dir / f"{name}.csv", index=False)
    
    print(f" ✅ Données sauvegardées dans : {raw_dir}")

def main():
    Path(__file__).resolve().parent.parent.joinpath("output").mkdir(exist_ok=True)
    print(f"Planification de {len(STRATEGIES_TO_RUN)} simulations")
    for name, internal_id in STRATEGIES_TO_RUN:
        run_single_strategy(name, internal_id)

if __name__ == "__main__":
    main()

