import sys
import pandas as pd
import logging
import random
import numpy as np
from pathlib import Path

# ================= SETUP =================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))

try:
    from sim.faassim import Simulation
    from sim.core import Environment
    from sim.faas import FaasSystem
    from ext.mhfd.power import Raith21PowerOracle, DEVICE_POWER_PROFILES
    
    # Stratégies
    from ext.mhfd.scaling.strategies.khpa import KubernetesStyleFirstFitBinPacker
    from ext.mhfd.scaling.strategies.standard_first_fit_packer import StandardFirstFitBinPacker
    from ext.mhfd.scaling.strategies.hpst import HighPerformanceShortTimeBinPacker
    from ext.mhfd.scaling.strategies.lplt import LowPowerLongTimeBinPacker
    from ext.mhfd.scaling.strategies.AdaptivePerformanceEnergyHybrid import AdaptivePerformanceEnergyHybrid
    from ext.mhfd.scaling.base_autoscaler import BaseAutoscaler

except ImportError as e:
    print(f"❌ Erreur Import: {e}"); sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ==========================================================
# 1. PATCHS & UTILITAIRES DE CHARGE
# ==========================================================
# Dictionnaire global pour simuler la charge CPU partagée entre le Système et les Stratégies
GLOBAL_NODE_LOAD = {} 

def get_simulated_cpu(node_name):
    """Retourne l'utilisation CPU simulée (0.0 à 1.0) basée sur le nombre de requêtes actives."""
    active_reqs = GLOBAL_NODE_LOAD.get(node_name, 0)
    # Disons qu'une requête prend 15% de CPU
    util = active_reqs * 0.15
    return min(util, 1.0)

def patch_strategies():
    """Patch pour que TOUTES les stratégies voient la charge simulée."""
    
    # Patch générique pour _safe_node_util ou get_node_utilization
    def patched_get_util(self, node):
        node_name = getattr(node, 'name', str(node))
        cpu = get_simulated_cpu(node_name)
        return {'cpu': cpu, 'memory': cpu * 0.8, 'gpu': 0.0, 'network': 0.0}

    # Application aux classes
    LowPowerLongTimeBinPacker.get_node_utilization = patched_get_util
    HighPerformanceShortTimeBinPacker.get_node_utilization = patched_get_util
    StandardFirstFitBinPacker.get_node_utilization = patched_get_util
    KubernetesStyleFirstFitBinPacker.get_node_utilization = patched_get_util
    AdaptivePerformanceEnergyHybrid.get_node_utilization = patched_get_util
    
    # Patchs spécifiques LPLT (méthodes manquantes)
    def lplt_get_nodes(self):
        nodes = set()
        if hasattr(self, 'faas_system') and hasattr(self.faas_system, 'replicas'):
            for r_list in self.faas_system.replicas.values():
                for r in r_list: nodes.add(r.node)
        return list(nodes)
    LowPowerLongTimeBinPacker.get_nodes_with_any_replicas = lplt_get_nodes
    LowPowerLongTimeBinPacker.has_any_replicas = lambda self, n: n in lplt_get_nodes(self)

patch_strategies()

# ==========================================================
# 3. SYSTÈME SIMULÉ INTELLIGENT (AVEC AUTOSCALING)
# ==========================================================
class ConcreteFaasSystem(FaasSystem):
    def __init__(self, env):
        self.env = env; self.strategy = None; self.replicas = {}    
        # Reset de la charge globale à chaque nouvelle simu
        GLOBAL_NODE_LOAD.clear()

    def get_replicas(self, fn_name): return self.replicas.get(fn_name, [])

    def invoke(self, request):
        fn_name = request['image']
        
        # --- 1. AUTOSCALING LOGIC ---
        # On regarde si on a besoin de plus de puissance
        current_reps = self.replicas.get(fn_name, [])
        
        # Calcul de la charge moyenne sur les réplicas existants
        avg_load = 0
        if current_reps:
            total_load = sum(GLOBAL_NODE_LOAD.get(r.node.name, 0) for r in current_reps)
            avg_load = total_load / len(current_reps)
        
        # SEUIL DE SCALING : Si > 4 requêtes en cours par réplica, on Scale UP
        need_scale = not current_reps or avg_load >= 4.0
        
        if need_scale:
            # On demande un nouveau nœud à la stratégie
            self.scale_up(fn_name, 1)
            # Petit délai de "Provisioning" pour simuler le démarrage
            yield self.env.timeout(0.2)
        
        # --- 2. LOAD BALANCING ---
        # On rafraîchit la liste (au cas où scale_up a marché)
        my_replicas = self.replicas.get(fn_name, [])
        if not my_replicas: return # Échec total

        # On choisit le réplica sur le nœud le MOINS chargé
        my_replicas.sort(key=lambda r: GLOBAL_NODE_LOAD.get(r.node.name, 0))
        replica_node = my_replicas[0].node
        
        # --- 3. EXÉCUTION & PHYSIQUE ---
        node_name = replica_node.name
        
        # On augmente la charge (Concurrence)
        GLOBAL_NODE_LOAD[node_name] = GLOBAL_NODE_LOAD.get(node_name, 0) + 1
        
        try:
            # Simulation physique
            base_duration = 0.5
            speed = 1.0; watts = 5.0
            name = node_name.lower()

            if 'xeon' in name:   speed = 0.15; watts = 120.0
            elif 'nuc' in name:  speed = 0.35; watts = 25.0
            elif 'nx' in name:   speed = 0.4;  watts = 15.0
            elif 'tx2' in name:  speed = 0.5;  watts = 8.0 
            elif 'nano' in name: speed = 0.8;  watts = 5.0
            elif 'rpi4' in name: speed = 1.4;  watts = 4.0 
            elif 'rpi3' in name: speed = 2.0;  watts = 2.0 

            # Pénalité de saturation : Si CPU > 100%, ça ralentit fort
            cpu_load = get_simulated_cpu(node_name)
            congestion = 1.0
            if cpu_load > 1.0: congestion = cpu_load * 1.5 # Pénalité exponentielle

            actual_duration = base_duration * speed * congestion
            
            yield self.env.timeout(actual_duration)
            
            # Logs
            if hasattr(self.env, 'metrics'):
                self.env.metrics.log('response_time', {'timestamp': self.env.now, 'function': fn_name, 'node': node_name, 'duration': actual_duration})
                self.env.metrics.log('energy_consumption', {'timestamp': self.env.now, 'node': node_name, 'joules': watts * actual_duration})
                
        finally:
            # On libère la charge à la fin (Try/Finally assure que c'est fait même si crash)
            GLOBAL_NODE_LOAD[node_name] = max(0, GLOBAL_NODE_LOAD.get(node_name, 0) - 1)

    def scale_up(self, fn_name, count=1):
        if fn_name not in self.replicas: self.replicas[fn_name] = []
        for _ in range(count):
            selected = None
            if self.strategy:
                try: selected = self.strategy.select_node_for_scaling(fn_name)
                except: pass
            
            # Fallback
            if not selected and self.env.topology.nodes: 
                # On prend un au hasard pour éviter de toujours prendre le premier
                import random
                selected = random.choice(self.env.topology.nodes)
            
            if selected:
                class MockReplica:
                    def __init__(self, n): self.node = n
                self.replicas[fn_name].append(MockReplica(selected))

    # Méthodes bouchons
    def deploy(self, f): pass
    def remove(self, f): pass
    def discover(self, f): return []
    def get_deployments(self): return []
    def get_function_index(self): return {}
    def poll_available_replica(self, f): return None
    def scale_down(self, f, c=1): pass
    def suspend(self, f): pass

# ================= 4. INFRASTRUCTURE & SCÉNARIO =================
def create_complex_topology():
    class Node:
        def __init__(self, name, cpu, mem): 
            self.name=name; self.capacity={'cpu':cpu,'memory':mem}; self.labels={}; self.resource=None
    
    # 3 clusters distincts pour bien voir la diversité
    nodes_list = [
        # Cluster RPi (LPLT)
        Node('rpi3-1', 4, 1024),
        Node('rpi4-1', 4, 4096),
        # Cluster Edge (APE-H)
        Node('tx2-1', 6, 8192), Node('nx-1', 6, 8192),
        # Cluster Cloud (HPST)
        Node('nuc-1', 8, 16384), Node('xeon-1', 32, 64000)
    ]
    class Topo: 
        @property
        def nodes(self): return nodes_list
        def get_nodes(self): return nodes_list
    return Topo()

def create_bursty_trace():
    # Scénario : 85 requêtes très rapprochées (Burst)
    # Cela va forcer la saturation des premiers nœuds
    t = np.linspace(0, 10, 85) 
    fn = ['resnet50-inference'] * 85
    return pd.DataFrame({'timestamp': t, 'function_name': fn})

class BenchmarkWrapper:
    def __init__(self, df): self.df = df
    def setup(self, env): pass
    def run(self, env):
        print(f"   ⏱️  Début Scénario Stress ({len(self.df)} reqs)...")
        for _, row in self.df.iterrows():
            wait = row['timestamp'] - env.now
            if wait > 0: yield env.timeout(wait)
            env.process(env.faas_system.invoke({'image': row['function_name']}))
        yield env.timeout(10.0) # Attente fin
        print("   🏁 Scénario terminé.")

def run_simulation(StrategyClass, name):
    print(f"\n⚡ --- SCÉNARIO : {name} ---")
    trace_df = create_bursty_trace()
    env = Environment()
    env.topology = create_complex_topology()
    env.power_oracle = Raith21PowerOracle(DEVICE_POWER_PROFILES)
    
    class Metrics:
        def __init__(self): self.data = []
        def log(self, m, v): 
            if isinstance(v, dict): v['metric']=m; self.data.append(v)
        def record_power_sample(self, *a, **k): return {}
        def get_df(self): return pd.DataFrame(self.data)
    
    env.metrics = Metrics(); env.power_metrics = Metrics(); env.power_monitoring_interval = 100 
    sys = ConcreteFaasSystem(env); env.faas_system = sys
    
    try:
        strat = StrategyClass(env=env, faas_system=sys, power_oracle=env.power_oracle)
        sys.strategy = strat
    except Exception as e: print(f"❌ Erreur Init Stratégie: {e}"); return

    sim = Simulation(env.topology, BenchmarkWrapper(trace_df), env, timeout=100)
    sim.scheduler = strat
    
    try:
        sim.run()
        res = env.metrics.get_df()
        out = Path(project_root) / 'Thao' / 'output' / f"results_{name}.csv"
        out.parent.mkdir(exist_ok=True)
        
        if not res.empty:
            res.to_csv(out, index=False)
            node_counts = res[res['metric']=='response_time']['node'].value_counts()
            top_node = node_counts.idxmax()
            lat = res[res['metric']=='response_time']['duration'].mean()
            nrg = res[res['metric']=='energy_consumption']['joules'].sum()
            
            print(f"   ✅ Sauvegardé : {out}")
            print(f"   📊 RÉSULTATS {name}:")
            print(f"      - Nœud Dominant   : {top_node} ({node_counts[top_node]} reqs)")
            print(f"      - Latence Moyenne : {lat:.3f} s")
            print(f"      - Energie Totale  : {nrg:.1f} J")
            
            # Affichage de la diversité (Combien de noeuds utilisés ?)
            used = list(node_counts.index)
            print(f"      - Diversité ({len(used)}) : {used[:4]}...")
        else:
            print("   ⚠️  Pas de données.")
            
    except Exception as e:
        print(f"   ❌ Erreur Runtime: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_simulation(StandardFirstFitBinPacker, "Standard_First")
    run_simulation(KubernetesStyleFirstFitBinPacker, "Standard_K8s")
    run_simulation(HighPerformanceShortTimeBinPacker, "HPST_Performance")
    run_simulation(LowPowerLongTimeBinPacker, "LPLT_Energy")
    run_simulation(AdaptivePerformanceEnergyHybrid, "APE_H_Hybrid")