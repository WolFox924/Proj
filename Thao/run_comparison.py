import sys
import pandas as pd
import logging
import random
from pathlib import Path

# ================= SETUP =================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path: 
    sys.path.insert(0, str(project_root))

from sim.faassim import Simulation
from sim.core import Environment
from sim.faas import FaasSystem
from ext.mhfd.power import Raith21PowerOracle, DEVICE_POWER_PROFILES    
from ext.mhfd.scaling.strategies.standard_first_fit_packer import StandardFirstFitBinPacker
from ext.mhfd.scaling.strategies.hpst import HighPerformanceShortTimeBinPacker
from ext.mhfd.scaling.strategies.lplt import LowPowerLongTimeBinPacker
from ext.mhfd.scaling.strategies.khpa import KubernetesStyleFirstFitBinPacker

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#Aide Gemini
def patch_lplt_strategy():
    #Injecte les méthodes manquantes dans LPLT pour éviter les crashs.
    def get_nodes_with_any_replicas_fixed(self):
        nodes = set()
        if hasattr(self, 'faas_system') and hasattr(self.faas_system, 'replicas'):
            for replicas_list in self.faas_system.replicas.values():
                for replica in replicas_list:
                    if hasattr(replica, 'node'): nodes.add(replica.node)
        return list(nodes)

    def has_any_replicas_fixed(self, node):
        #Vérifie si un nœud héberge déjà quelque chose.
        if hasattr(self, 'faas_system') and hasattr(self.faas_system, 'replicas'):
            for replicas_list in self.faas_system.replicas.values():
                for replica in replicas_list:
                    if hasattr(replica, 'node') and replica.node == node:
                        return True
        return False

    # Application des patchs
    LowPowerLongTimeBinPacker.get_nodes_with_any_replicas = get_nodes_with_any_replicas_fixed
    LowPowerLongTimeBinPacker.has_any_replicas = has_any_replicas_fixed
    print("Méthodes injectées dans LowPowerLongTimeBinPacker.")
patch_lplt_strategy()


# ================= FAAS SYSTEM =================
# ================= 1. LE SYSTÈME INTELLIGENT (CORRIGÉ) =================
class ConcreteFaasSystem(FaasSystem):
    def __init__(self, env):
        self.env = env
        self.strategy = None  
        self.replicas = {}    

    def get_replicas(self, fn_name):
        return self.replicas.get(fn_name, [])
    
    # Méthode utilitaire pour LPLT
    def get_all_nodes_with_replicas(self):
        nodes = set()
        for r_list in self.replicas.values():
            for r in r_list: nodes.add(r.node)
        return list(nodes)

    def invoke(self, request):
        fn_name = request['image']
        
        # 1. Scaling (Cold Start)
        # Si la fonction n'existe pas ou n'a pas de réplicas, on doit scaler
        if fn_name not in self.replicas or not self.replicas[fn_name]:
            self.scale_up(fn_name, 1)
            # On simule le temps de démarrage du conteneur
            yield self.env.timeout(0.5) 
        
        # --- CORRECTION ICI : Vérification de sécurité après le délai ---
        if fn_name not in self.replicas or not self.replicas[fn_name]:
            # Si malgré le scale_up, on n'a rien (ex: stratégie a échoué), on abandonne cette requête
            # Cela évite le Crash KeyError
            logging.warning(f"⚠️ Échec invocation : Pas de réplica disponible pour {fn_name}")
            return 
        # ---------------------------------------------------------------
        
        # 2. Sélection du nœud
        my_replicas = self.replicas[fn_name]
        
        # Round Robin pour l'exécution
        replica_node = my_replicas[random.randint(0, len(my_replicas)-1)].node
        
        # 3. Physique Simulée
        base_duration = 0.5
        speed_factor = 1.0
        power_watts = 5.0

        if 'xeon' in replica_node.name: 
            speed_factor = 0.2; power_watts = 100.0
        elif 'nx' in replica_node.name: 
            speed_factor = 0.4; power_watts = 15.0
        elif 'nano' in replica_node.name: 
            speed_factor = 0.8; power_watts = 5.0
        elif 'rpi4' in replica_node.name: 
            speed_factor = 1.4; power_watts = 4.0
        elif 'rpi3' in replica_node.name: 
            speed_factor = 1.6; power_watts = 2.0
        elif 'rpi' in replica_node.name: 
             speed_factor = 1.5; power_watts = 3.0

        actual_duration = base_duration * speed_factor
        
        yield self.env.timeout(actual_duration)
        
        if hasattr(self.env, 'metrics'):
            self.env.metrics.log('response_time', {
                'timestamp': self.env.now,
                'function': fn_name,
                'node': replica_node.name,
                'duration': actual_duration
            })
            
            energy_joules = power_watts * actual_duration
            self.env.metrics.log('energy_consumption', {
                'timestamp': self.env.now,
                'node': replica_node.name,
                'joules': energy_joules
            })

    def scale_up(self, fn_name, count=1):
        if fn_name not in self.replicas: self.replicas[fn_name] = []
        
        for _ in range(count):
            selected_node = None
            if self.strategy and hasattr(self.strategy, 'select_node_for_scaling'):
                try:
                    selected_node = self.strategy.select_node_for_scaling(fn_name)
                except Exception as e:
                    logging.error(f"Erreur stratégie lors du choix du nœud: {e}")
            
            # Fallback si la stratégie ne renvoie rien
            if not selected_node and self.env.topology.nodes:
                # On prend le premier nœud par défaut pour éviter le crash
                selected_node = self.env.topology.nodes[0]
            
            if selected_node:
                class MockReplica:
                    def __init__(self, n): self.node = n
                self.replicas[fn_name].append(MockReplica(selected_node))
            else:
                 logging.error(f"❌ Impossible de scaler {fn_name}: Aucun nœud disponible.")

    # Méthodes abstraites bouchonnées
    def deploy(self, f): pass
    def remove(self, f): pass
    def discover(self, f): return []
    def get_deployments(self): return []
    def get_function_index(self): return {}
    def poll_available_replica(self, f): return None
    def scale_down(self, f, c=1): pass
    def suspend(self, f): pass
# ================= INFRASTRUCTURE & WORKLOAD =================

def create_topology():
    class Node:
        def __init__(self, name, cpu, mem): 
            self.name=name; self.capacity={'cpu':cpu,'memory':mem}; self.labels={}
            self.resource=None
    
    # Ordre important pour tester les préférences
    nodes_list = [
        Node('rpi3-1', 4, 1024), 
        Node('rpi4-1', 4, 4096),
        Node('nano-1', 4, 4096), 
        Node('nx-1', 6, 8192),
        Node('xeon-1', 32, 64000)
    ]
    class Topo: 
        @property
        def nodes(self): return nodes_list
        def get_nodes(self): return nodes_list
    return Topo()

class BenchmarkWrapper:
    def __init__(self, df): self.df = df
    def setup(self, env): pass
    def run(self, env):
        print(f"   ⏱️  Début Benchmark ({len(self.df)} reqs)...")
        for _, row in self.df.iterrows():
            wait = row['timestamp'] - env.now
            if wait > 0: yield env.timeout(wait)
            env.process(env.faas_system.invoke({'image': row['function_name']}))
        yield env.timeout(2.0)
        print("   🏁 Benchmark terminé.")

# ================= MOTEUR / SIMULATION =================
def run_simulation(StrategyClass, name):
    print(f"\n⚡ --- SCÉNARIO : {name} ---")
    
    # 50 requêtes pour avoir des stats fiables toutes les 0.2 secondes
    data = {'timestamp': [i*0.2 for i in range(50)], 
            'function_name': ['resnet50-inference']*50}
    trace_df = pd.DataFrame(data)

    env = Environment()
    env.topology = create_topology()
    env.power_oracle = Raith21PowerOracle(DEVICE_POWER_PROFILES)
    
    class Metrics:
        def __init__(self): self.data = []
        def log(self, m, v): 
            if isinstance(v, dict): v['metric']=m; self.data.append(v)
        def record_power_sample(self, *a, **k): return {}
        def get_df(self): return pd.DataFrame(self.data)
    
    env.metrics = Metrics(); env.power_metrics = Metrics()
    env.power_monitoring_interval = 100 

    sys = ConcreteFaasSystem(env)
    env.faas_system = sys
    
    try:
        strat = StrategyClass(env=env, faas_system=sys, power_oracle=env.power_oracle)
        sys.strategy = strat
    except Exception as e:
        print(f"❌ Erreur Init Stratégie: {e}"); return

    sim = Simulation(env.topology, BenchmarkWrapper(trace_df), env, timeout=30)
    sim.scheduler = strat
    
    try:
        sim.run()
        res = env.metrics.get_df()
        out = Path(project_root) / 'Thao' / 'output' / f"results_{name}.csv"
        out.parent.mkdir(exist_ok=True)
        
        if not res.empty:
            res.to_csv(out, index=False)
            print(f"   ✅ Données sauvegardées : {out}")
            
            node_counts = res[res['metric']=='response_time']['node'].value_counts()
            top_node = node_counts.idxmax() if not node_counts.empty else "None"
            
            lat = res[res['metric']=='response_time']['duration']
            nrg = res[res['metric']=='energy_consumption']['joules']
            
            print(f"   📊 RÉSULTATS {name}:")
            print(f"      - Nœud Préféré    : {top_node}")
            print(f"      - Latence Moyenne : {lat.mean():.3f} s")
            print(f"      - Energie Totale  : {nrg.sum():.1f} J")
        else:
            print("   ⚠️  Pas de données.")
            
    except Exception as e:
        print(f"   ❌ Erreur Runtime: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_simulation(KubernetesStyleFirstFitBinPacker, "Standard_K8s")
    run_simulation(StandardFirstFitBinPacker, "Standard_First")
    run_simulation(HighPerformanceShortTimeBinPacker, "HPST_Performance")
    run_simulation(LowPowerLongTimeBinPacker, "LPLT_Energy")

#run_simulation ->