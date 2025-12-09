import logging
from typing import Optional, List, Dict
from typing import Optional, Any, List, Tuple
from sim.faas.core import FunctionDeployment, Node
from ext.mhfd.scaling.base_autoscaler import BaseAutoscaler
from ext.mhfd.power import get_current_utilization
logger = logging.getLogger(__name__)

class AdaptivePerformanceEnergyHybrid(BaseAutoscaler):
    """
    Adaptive Performance-Energy Hybrid (APE-H) autoscaler.
    Uses both performance (RPS, response times) and energy efficiency
    to make scaling decisions.
    compromis entre HPST et LPLT
    """

    def __init__(self, env, faas_system, power_oracle):
        super().__init__(env, faas_system, power_oracle, strategy_name="APE-H")
        self.last_scale_time = 0

    # -----------------------------
    # Helper: list available nodes
    # -----------------------------
    def get_available_nodes(self) -> List[Node]:
        """Return all compute nodes available for scaling"""
        if hasattr(self.faas, 'get_nodes'):
            return [node for node in self.faas.get_nodes() if getattr(node, 'node_type', None) == 'compute']
        else:
            logger.warning("FaaS system does not have get_nodes(), returning empty list")
            return []

    # -----------------------------
    # Main scaling logic
    # -----------------------------
    def make_scaling_decision(self, deployment_name: str, current_replicas: int, 
                              current_load: float, avg_response_time: float) -> str:
        """Decide whether to scale up, scale down, or do nothing"""
        if current_load > self.scale_up_threshold:
            return "scale_up"
        elif current_load < self.scale_down_threshold:
            return "scale_down"
        else:
            return "no_action"

    def select_node_for_scaling(self, deployment_name: str) -> Optional[Node]:
        """Select the best node to place a new replica"""
        available_nodes = self.get_available_nodes()
        if not available_nodes:
            return None

        # Rank nodes by lowest utilization
        min_util = float("inf")
        selected_node = None
        for node in available_nodes:
            util = self.get_node_utilization(node)
            if util['cpu'] < min_util:
                min_util = util['cpu']
                selected_node = node
        return selected_node

    # -----------------------------
    # Node utilization helper
    # -----------------------------
    def get_node_utilization(self, node: Node) -> Dict[str, float]:
        """Compute utilization for a node using power_oracle or defaults"""
        try:
            return self.power_oracle.get_current_utilization(self.env, node.name)
        except Exception as e:
            logger.error(f"Failed to get utilization for {node.name}: {e}")
            return {'cpu': 0.0, 'memory': 0.0, 'gpu': 0.0, 'network': 0.0}

