import random
from typing import Dict, Optional

from ether.core import Node as EtherNode
from faas.system.core import Environment, FunctionRequest
from faas.system.default import DefaultFaasSystem  # type: ignore[import-not-found]

from .scheduler import LocalNodeScheduler


class DecentralizedFaasSystem(DefaultFaasSystem):
    """
    Remplace le routage centralisé de DefaultFaasSystem en délégant chaque requête
    entrante au LocalNodeScheduler de son nœud d'origine.
    """

    def __init__(
        self, env: Environment, default_ingress_node: Optional[str] = None
    ) -> None:
        super().__init__(env)
        self.default_ingress_node = default_ingress_node

    def invoke(self, request: FunctionRequest):
        schedulers: Dict[str, LocalNodeScheduler] = getattr(
            self.env, "node_schedulers", {}
        )
        if not schedulers:
            return super().invoke(request)

        origin = getattr(request, "client", None)
        if origin not in schedulers:
            origin = (
                self.default_ingress_node
                if self.default_ingress_node in schedulers
                else next(iter(schedulers))
            )

        return schedulers[origin].handle_request(request)


def setup_decentralized_schedulers(
    env: Environment,
    max_hops: int = 3,
    offload_prob: float = 0.5,
    seed: int = 42,
) -> Dict[str, LocalNodeScheduler]:
    """
    Instancie un LocalNodeScheduler indépendant par nœud de calcul de la topologie Ether
    et bascule env.faas en mode décentralisé.
    """
    rng = random.Random(seed)
    worker_nodes = [
        node
        for node in env.topology.nodes
        if isinstance(node, EtherNode) and node.name != "registry"
    ]

    env.node_schedulers = {
        node.name: LocalNodeScheduler(
            env=env,
            node_name=node.name,
            max_hops=max_hops,
            offload_prob=offload_prob,
            rng=random.Random(rng.randint(0, 2**32 - 1)),
        )
        for node in worker_nodes
    }

    default_ingress = worker_nodes[0].name if worker_nodes else None
    env.faas = DecentralizedFaasSystem(env, default_ingress_node=default_ingress)
    return env.node_schedulers