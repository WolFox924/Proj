
import logging
import math
from typing import Optional, Any, List, Dict, Tuple

from sim.faas.core import Node  # type: ignore
from ext.mhfd.scaling.base_autoscaler import BaseAutoscaler
from ext.mhfd.power import get_current_utilization  # function: (env, node_name) -> Dict[str,float]

logger = logging.getLogger(__name__)


class AdaptivePerformanceEnergyHybrid(BaseAutoscaler):
    """
    Adaptive Performance-Energy Hybrid (APE-H) autoscaler

    Principes:
    - Combine indicateurs de latence (response time) et d'efficacité énergétique (W/RPS).
    - Comportements profilés: latency_sensitive, energy_sensitive, balanced.
    - Priorise consolidation (ajouter sur nœud qui a déjà des replicas) quand possible.
    - Défensif: gère absences d'APIs, valeurs manquantes, et renvoie des choix par défaut sûrs.
    - Compromis entre HPST et LPLT
    """

    def __init__(self, env, faas_system, power_oracle):
        # strategy_name lisible (hérité)
        super().__init__(env, faas_system, power_oracle, strategy_name="APE-H")

        # Paramètres adaptatifs (modifiable facilement)
        self.scale_up_threshold = getattr(self, "scale_up_threshold", 15.0)    # RPS
        self.scale_down_threshold = getattr(self, "scale_down_threshold", 5.0) # RPS
        self.response_time_threshold = getattr(self, "response_time_threshold", 600.0)  # ms
        self.latency_weight = 0.6    # poids de la latence dans la décision hybride
        self.energy_weight = 0.4     # poids de l'énergie (W/RPS) dans la décision hybride
        self.min_delta_to_scale = 0.10  # 10% changement nécessaire pour scaler
        self.cooldown = 5.0  # secondes (sim time) entre scales par déploiement
        self._last_scale_time = {}

        # classement simple des types de noeuds (plus élevé => plus performant)
        self.performance_rank = {
            'xeongpu': 20, 'xeoncpu': 18, 'nuc': 15, 'nx': 14, 'tx2': 13,
            'rpi4': 8, 'rockpi': 7, 'nano': 6, 'coral': 5, 'rpi3': 4, 'unknown': 1
        }

        logger.info("Initialized Adaptive Performance-Energy Hybrid (APE-H)")

    # -------------------------
    # Public strategy methods
    # -------------------------
    def make_scaling_decision(self, deployment_name: str, current_replicas: int,
                              current_load: float, avg_response_time: float) -> str:
        """
        Décision de scaling:
        - Si load > up_threshold => scale_up
        - Si load < down_threshold => scale_down
        - Sinon, calcule metric hybride (latence normalisée + énergie normalisée).
          Si metric > 1 + min_delta_to_scale => scale_up
          Si metric < 1 - min_delta_to_scale => scale_down
          else no_action
        """

        # Respect cooldown par deployment
        last = self._last_scale_time.get(deployment_name, -9999)
        if (self.env.now - last) < self.cooldown:
            logger.debug(f"Cooldown active for {deployment_name}: {(self.env.now-last):.1f}s < {self.cooldown}s")
            return "no_action"

        # Edge cases defensifs
        load = max(0.0, float(current_load or 0.0))
        rtime = float(avg_response_time or 0.0)

        # Rule-of-thumb thresholds first
        if load > self.scale_up_threshold and current_replicas < self.max_replicas:
            logger.debug(f"{deployment_name}: direct scale_up (load {load:.2f} > {self.scale_up_threshold})")
            return "scale_up"
        if load < self.scale_down_threshold and current_replicas > self.min_replicas:
            logger.debug(f"{deployment_name}: direct scale_down (load {load:.2f} < {self.scale_down_threshold})")
            return "scale_down"

        # Compute energy-per-RPS for this deployment (lower = better)
        energy_per_rps = self._estimate_energy_per_rps(deployment_name)
        # Normalize latence: 1.0 => exactly at threshold
        latency_score = (rtime / max(1.0, self.response_time_threshold))
        # Normalize energy: map to ~1.0 baseline (choose 1.0 when energy_per_rps near baseline)
        # baseline chosen empirically; avoid division by 0
        baseline_energy = 1.0  # joules per rps baseline — ajuster si nécessaire
        energy_score = energy_per_rps / max(1e-6, baseline_energy)

        # Combined metric: >1 => worse than baseline (need upscale)
        combined_metric = self.latency_weight * latency_score + self.energy_weight * energy_score

        logger.debug(f"APE-H DECISION [{deployment_name}] load={load:.2f} rtime={rtime:.1f}ms "
                     f"energy_per_rps={energy_per_rps:.3f} => combined={combined_metric:.3f}")

        if combined_metric > 1.0 + self.min_delta_to_scale and current_replicas < self.max_replicas:
            return "scale_up"
        if combined_metric < 1.0 - self.min_delta_to_scale and current_replicas > self.min_replicas:
            return "scale_down"
        return "no_action"

    def select_node_for_scaling(self, deployment_name: str) -> Optional[Node]:
        """
        Sélection du nœud:
        - Tenter consolidation: nœuds qui hébergent déjà des replicas de ce déploiement (si safe)
        - Autrement, classer par score hybride (performance vs power efficiency)
        - Retourne le premier nœud avec ressources suffisantes
        """

        # 1) Try consolidation on existing nodes
        try:
            existing_nodes = self.get_nodes_with_replicas(deployment_name)
        except Exception:
            existing_nodes = []

        for node in existing_nodes:
            util = self._safe_node_util(node)
            if util and util.get('cpu', 0.0) < 0.85 and util.get('memory', 0.0) < 0.85:
                logger.info(f"APE-H: consolidating on existing node {node.name}")
                return node

        # 2) Candidate nodes
        candidates = self.get_available_nodes()
        if not candidates:
            logger.warning("APE-H: no available nodes found")
            return None

        # Build hybrid score for each node
        scored: List[Tuple[Node, float]] = []
        for node in candidates:
            util = self._safe_node_util(node)
            if util is None:
                # If we can't read utilization, assume light usage
                util = {'cpu': 0.0, 'memory': 0.0, 'gpu': 0.0, 'network': 0.0}

            perf = float(self.performance_rank.get(self.extract_node_type(node.name), 1))
            # prefer nodes with lower current cpu utilization and higher perf
            cpu_headroom = max(0.0, 1.0 - util.get('cpu', 1.0))
            mem_headroom = max(0.0, 1.0 - util.get('memory', 1.0))
            headroom_score = 0.7 * cpu_headroom + 0.3 * mem_headroom

            # Combined score: favor high-perf & headroom
            score = 0.6 * perf + 0.4 * (headroom_score * 100.0)
            scored.append((node, score))

        if not scored:
            logger.warning("APE-H: no scored nodes")
            return None

        scored.sort(key=lambda x: -x[1])  # descending
        # Pick first that has sufficient resources for this deployment
        for node, score in scored:
            if self.has_sufficient_resources(node, deployment_name):
                logger.info(f"APE-H Selected node {node.name} (score={score:.2f}) for {deployment_name}")
                return node

        logger.warning("APE-H: no node with sufficient resources found after scoring")
        return None

    # -------------------------
    # Internal helpers
    # -------------------------
    def get_available_nodes(self) -> List[Node]:
        """Return list of nodes usable for scheduling; fallback to topology if faas lacks method."""
        # Prefer faas.get_nodes() if present
        if hasattr(self.faas, 'get_nodes'):
            try:
                nodes = list(self.faas.get_nodes())
                return nodes
            except Exception as e:
                logger.debug(f"faas.get_nodes() failed: {e}")

        # Fallback: use environment topology if available
        if hasattr(self.env, 'topology') and hasattr(self.env.topology, 'get_nodes'):
            try:
                return list(self.env.topology.get_nodes())
            except Exception as e:
                logger.debug(f"env.topology.get_nodes() failed: {e}")

        logger.warning("No node list available from faas or env.topology")
        return []

    def _estimate_energy_per_rps(self, deployment_name: str) -> float:
        """
        Estime l'énergie (watts) par RPS pour le déploiement en sommant consommation
        des replicas et divisant par charge actuelle (RPS).
        """
        try:
            replicas = self.faas.get_replicas(deployment_name)
        except Exception:
            replicas = []

        if not replicas:
            return 0.0

        current_load = max(0.1, float(self.get_current_load(deployment_name)))
        total_power = 0.0
        for replica in replicas:
            node = getattr(replica, 'node', None)
            if node is None:
                continue
            util = self._safe_node_util(node)
            # power_oracle may have predict_power(node_type, cpu, gpu, network, memory)
            node_type = self.extract_node_type(node.name)
            try:
                if hasattr(self.power_oracle, 'predict_power'):
                    p = float(self.power_oracle.predict_power(node_type,
                                                             util.get('cpu', 0.0),
                                                             util.get('gpu', 0.0),
                                                             util.get('network', 0.0),
                                                             util.get('memory', 0.0)))
                else:
                    # fallback: simple heuristic: perf_rank * cpu_util * cte
                    p = float(self.performance_rank.get(node_type, 1)) * util.get('cpu', 0.0) * 2.0
            except Exception:
                p = 0.0
            total_power += p

        energy_per_rps = total_power / current_load
        return energy_per_rps

    def _safe_node_util(self, node: Any) -> Optional[Dict[str, float]]:
        """Wrapper around get_current_utilization with env and node name handling."""
        try:
            # the power.get_current_utilization signature is (env, node_name)
            util = get_current_utilization(self.env, node.name)
            if util is None:
                return {'cpu': 0.0, 'memory': 0.0, 'gpu': 0.0, 'network': 0.0}
            # ensure keys exist
            return {
                'cpu': float(util.get('cpu', 0.0)),
                'memory': float(util.get('memory', 0.0)),
                'gpu': float(util.get('gpu', 0.0)),
                'network': float(util.get('network', 0.0))
            }
        except Exception as e:
            logger.debug(f"Could not read utilization for {getattr(node, 'name', node)}: {e}")
            return {'cpu': 0.0, 'memory': 0.0, 'gpu': 0.0, 'network': 0.0}

    # -------------------------
    # Override record_scaling_decision to update cooldown
    # -------------------------
    def record_scaling_decision(self, deployment_name: str, action: str,
                                new_replica_count: int, selected_node: Any,
                                current_load: float = 0.0, avg_response_time: float = 0.0,
                                reason: str = ""):
        super().record_scaling_decision(deployment_name, action, new_replica_count,
                                        selected_node, current_load, avg_response_time, reason)
        if action in ("scale_up", "scale_down"):
            self._last_scale_time[deployment_name] = self.env.now

    # -------------------------
    # Utility: wrappers to reuse base methods if necessary
    # -------------------------
    def get_nodes_with_replicas(self, deployment_name: str) -> List[Any]:
        """Return nodes that already host a replica of deployment_name (unique list)."""
        nodes = []
        try:
            for r in self.faas.get_replicas(deployment_name):
                if getattr(r, 'node', None) and r.node not in nodes:
                    nodes.append(r.node)
        except Exception:
            return []
        return nodes

