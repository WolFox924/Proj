from collections import deque
import random
from typing import Dict, List, Optional, Set

import simpy
from ether.core import Node as EtherNode
from ether.topology import Topology
from skippy.core.model import Node as SkippyNode, Pod, SchedulingResult

from faas.system.core import (
    Environment,
    FunctionDeployment,
    FunctionReplica,
    FunctionRequest,
    FunctionState,
    NodeState,
)


class LocalNodeScheduler:
    """
    Ordonnanceur strictement local attaché à un unique nœud physique.
    Visibilité limitée à l'état du nœud local et aux identifiants de ses voisins directs.
    """

    def __init__(
        self,
        env: Environment,
        node_name: str,
        max_hops: int = 3,
        offload_prob: float = 0.5,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.env: Environment = env
        self.node_name: str = node_name
        self.max_hops: int = max_hops
        self.offload_prob: float = offload_prob
        self.rng: random.Random = rng or random.Random()

        self.node_state: NodeState = env.get_node_state(node_name)
        self.ether_node: EtherNode = self.node_state.ether_node
        self.skippy_node: SkippyNode = self.node_state.skippy_node

        self.neighbors: List[str] = self._discover_direct_neighbors(env.topology)

        self.stats: Dict[str, int] = {
            "warm_hits": 0,
            "cold_starts": 0,
            "forwarded": 0,
            "dropped": 0,
        }

    def _discover_direct_neighbors(self, topology: Topology) -> List[str]:
        direct_neighbors: List[str] = []
        visited: Set[object] = {self.ether_node}
        queue = deque([self.ether_node])

        while queue:
            current = queue.popleft()
            for nbr in topology.neighbors(current):
                if nbr in visited:
                    continue
                visited.add(nbr)

                if isinstance(nbr, EtherNode):
                    if nbr.name not in (self.node_name, "registry"):
                        direct_neighbors.append(nbr.name)
                else:
                    queue.append(nbr)

        return direct_neighbors

    def schedule(self, pod: Pod) -> SchedulingResult:
        if self._pod_fits_locally(pod):
            needed_images = [
                c.image
                for c in pod.spec.containers
                if c.image not in self.skippy_node.images
            ]
            return SchedulingResult(self.skippy_node, needed_images, [self.skippy_node])
        return SchedulingResult(None, [], [])

    def _get_local_replicas(
        self, fn_name: str, states: Optional[Set[FunctionState]] = None
    ) -> List[FunctionReplica]:
        target_states = states or {FunctionState.RUNNING}
        return [
            r
            for r in self.env.faas.replicas.get(fn_name, [])
            if r.node is not None
            and r.node.name == self.node_name
            and r.state in target_states
        ]

    def _pod_fits_locally(self, pod: Pod) -> bool:
        req_cpu = 0
        req_mem = 0
        node_arch = self.skippy_node.labels.get("beta.kubernetes.io/arch") or (
            self.ether_node.arch.name.lower()
            if getattr(self.ether_node, "arch", None)
            else None
        )

        for container in pod.spec.containers:
            requests = getattr(container.resources, "requests", {}) or {}
            req_cpu += int(requests.get("cpu", 0))
            req_mem += int(requests.get("memory", 0))

            if node_arch and ":" in container.image:
                tag = container.image.split(":")[-1].lower()
                if tag in ("arm32", "aarch64", "x86", "amd64") and tag != node_arch:
                    if not (tag in ("x86", "amd64") and node_arch in ("x86", "amd64")):
                        return False

        alloc = self.skippy_node.allocatable
        return alloc.cpu_millis >= req_cpu and alloc.memory >= req_mem

    def _can_host_locally(self, deployment: FunctionDeployment) -> bool:
        pod = self.env.faas.create_pod(deployment)
        return self._pod_fits_locally(pod)

    def handle_request(
        self,
        request: FunctionRequest,
        ttl: Optional[int] = None,
        prev_node: Optional[str] = None,
    ) -> simpy.events.Event:
        return self.env.process(self._process_request(request, ttl, prev_node))

    def _process_request(
        self,
        request: FunctionRequest,
        ttl: Optional[int],
        prev_node: Optional[str],
    ):
        current_ttl = self.max_hops if ttl is None else ttl
        deployment: Optional[FunctionDeployment] = self.env.faas.deployments.get(
            request.name
        )
        if deployment is None:
            self.stats["dropped"] += 1
            return None

        warm_replicas = self._get_local_replicas(request.name, {FunctionState.RUNNING})
        if warm_replicas:
            self.stats["warm_hits"] += 1
            chosen_replica = self.rng.choice(warm_replicas)
            return (
                yield from self.env.faas.simulate_function_invocation(
                    self.env, chosen_replica, request
                )
            )

        can_host = self._can_host_locally(deployment)
        candidates = [n for n in self.neighbors if n != prev_node] or list(self.neighbors)

        if current_ttl <= 0 or not candidates:
            if can_host:
                return (yield from self._cold_start_and_execute(deployment, request))
            self.stats["dropped"] += 1
            return None

        should_offload = (not can_host) or (self.rng.random() < self.offload_prob)

        if should_offload:
            target_node_name = self.rng.choice(candidates)
            self.stats["forwarded"] += 1
            yield from self._simulate_hop_delay(target_node_name)

            target_scheduler: LocalNodeScheduler = self.env.node_schedulers[
                target_node_name
            ]
            return (
                yield from target_scheduler._process_request(
                    request, ttl=current_ttl - 1, prev_node=self.node_name
                )
            )

        return (yield from self._cold_start_and_execute(deployment, request))

    def _cold_start_and_execute(
        self, deployment: FunctionDeployment, request: FunctionRequest
    ):
        self.stats["cold_starts"] += 1
        replica = FunctionReplica(
            deployment.fn, self.node_state, FunctionState.CONCEIVED
        )
        self.env.faas.replicas.setdefault(deployment.name, []).append(replica)

        previous_scheduler = self.env.scheduler
        self.env.scheduler = self
        try:
            yield from self.env.faas.simulate_function_start(deployment, replica)
        finally:
            self.env.scheduler = previous_scheduler

        if replica.state != FunctionState.RUNNING:
            self.stats["dropped"] += 1
            return None

        return (
            yield from self.env.faas.simulate_function_invocation(
                self.env, replica, request
            )
        )

    def _simulate_hop_delay(self, target_node_name: str):
        target_state = self.env.get_node_state(target_node_name)
        route = self.env.topology.route(self.ether_node, target_state.ether_node)
        if route is not None and getattr(route, "rtt", 0) > 0:
            one_way_sec = (route.rtt / 2.0) / 1000.0
            if one_way_sec > 0:
                yield self.env.timeout(one_way_sec)