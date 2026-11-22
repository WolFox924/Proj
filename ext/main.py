import os
import random
import pandas as pd
import importlib

from ether.core import Node as EtherNode
from ether.scenarios.urbansensing import UrbanSensingScenario
from ether.topology import Topology
from faas.system.core import Environment, FunctionRequest


def build_topology() -> Topology:
    topology = Topology()
    scenario = UrbanSensingScenario()
    scenario.materialize(topology)
    return topology


def request_generator(env: Environment, fn_name: str, edge_nodes: list[str], rps: float = 2.0, duration: int = 120):
    rng = random.Random(42)
    while env.now < duration:
        yield env.timeout(rng.expovariate(rps))
        origin_node = rng.choice(edge_nodes)
        req = FunctionRequest(name=fn_name, client=origin_node)
        env.process(env.faas.invoke(req))


def main():
    output_dir = os.path.join("data", "decentralized_s1")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Construction de l'environnement et de la topologie Ether
    env = Environment()
    env.topology = build_topology()

    # 2. Activation de l'ordonnancement décentralisé (1 scheduler par nœud)
    try:
        setup_decentralized_schedulers = importlib.import_module(
            "ext.decentralized.system"
        ).setup_decentralized_schedulers
    except ModuleNotFoundError:
        import sys

        decentralized_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "decentralized")
        )
        if decentralized_dir not in sys.path:
            sys.path.insert(0, decentralized_dir)
        setup_decentralized_schedulers = importlib.import_module(
            "system"
        ).setup_decentralized_schedulers

    schedulers = setup_decentralized_schedulers(
        env=env,
        max_hops=3,
        offload_prob=0.5,
        seed=42,
    )

    # 3. Déploiement des fonctions (reprend les fonctions standard de examples/basic/main.py)
    from examples.basic.main import create_deployments
    deployments = create_deployments()
    for dep in deployments:
        env.faas.deploy(dep)

    # 4. Sélection des nœuds Edge d'entrée pour injecter le trafic
    edge_nodes = [
        name for name in schedulers.keys()
        if "rpi" in name or "nuc" in name or "edge" in name
    ] or list(schedulers.keys())

    target_fn = deployments[0].name
    env.process(request_generator(env, fn_name=target_fn, edge_nodes=edge_nodes, rps=5.0, duration=120))

    # 5. Exécution de la simulation SimPy
    env.run(until=130)

    # 6. Export des métriques au même format CSV que data/experiment_1_kubernetes_baseline/
    for metric_name in ["invocations", "schedule", "fets", "network", "energy", "power"]:
        try:
            df = env.metrics.extract_dataframe(metric_name)
            if df is not None and not df.empty:
                df.to_csv(os.path.join(output_dir, f"{metric_name}_df.csv"), index=False)
        except Exception:
            pass

    # Export des statistiques locales par nœud (warm hits, cold starts, sauts)
    stats_rows = [{"node": node, **sched.stats} for node, sched in schedulers.items()]
    pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, "local_schedulers_stats.csv"), index=False)
    print(f"Simulation terminée. Résultats exportés dans {output_dir}/")


if __name__ == "__main__":
    main()