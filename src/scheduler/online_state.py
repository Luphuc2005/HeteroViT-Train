"""Online Runtime State per Worker and Cluster-wide Telemetry."""
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List


class NodeRuntimeState:
    """Dynamic online runtime telemetry state for a single cluster node/worker."""

    def __init__(
        self,
        rank: int,
        node_id: str,
        current_batch: int = 0,
        compute_ms: float = 0.0,
        compute_ema_ms: float = 0.0,
        runtime_std_ms: float = 0.0,
        comm_ms: float = 0.0,
        idle_ms: float = 0.0,
        active: Optional[bool] = None,
        sample_count: int = 0,
        alpha: float = 0.2,
    ):
        self.rank = int(rank)
        self.node_id = str(node_id)
        self.current_batch = int(current_batch)
        self.compute_ms = float(compute_ms)
        self.compute_ema_ms = float(compute_ema_ms)
        self.runtime_std_ms = float(runtime_std_ms)
        self.comm_ms = float(comm_ms)
        self.idle_ms = float(idle_ms)
        self.active = bool(active if active is not None else (self.current_batch > 0))
        self.sample_count = int(sample_count)
        self.alpha = float(alpha)

    def update(
        self,
        observed_compute_ms: Optional[float] = None,
        observed_comm_ms: Optional[float] = None,
        observed_idle_ms: Optional[float] = None,
        compute_ms: Optional[float] = None,
        comm_ms: Optional[float] = None,
        idle_ms: Optional[float] = None,
        current_batch: Optional[int] = None,
        runtime_std_ms: float = 0.0,
        alpha: Optional[float] = None,
    ) -> None:
        """Updates the state with a new observation using Exponential Moving Average."""
        c_ms = observed_compute_ms if observed_compute_ms is not None else compute_ms
        if c_ms is not None:
            self.compute_ms = float(c_ms)

        cm_ms = observed_comm_ms if observed_comm_ms is not None else comm_ms
        if cm_ms is not None:
            self.comm_ms = float(cm_ms)

        i_ms = observed_idle_ms if observed_idle_ms is not None else idle_ms
        if i_ms is not None:
            self.idle_ms = float(i_ms)

        if current_batch is not None:
            self.current_batch = int(current_batch)
            self.active = (self.current_batch > 0)

        self.runtime_std_ms = float(runtime_std_ms)
        self.sample_count += 1
        a = float(alpha if alpha is not None else self.alpha)

        if self.active and self.compute_ms > 0.0:
            if self.compute_ema_ms <= 0.0:
                self.compute_ema_ms = self.compute_ms
            else:
                self.compute_ema_ms = float(a * self.compute_ms + (1.0 - a) * self.compute_ema_ms)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "node_id": self.node_id,
            "current_batch": self.current_batch,
            "compute_ms": self.compute_ms,
            "compute_ema_ms": self.compute_ema_ms,
            "runtime_std_ms": self.runtime_std_ms,
            "comm_ms": self.comm_ms,
            "idle_ms": self.idle_ms,
            "active": self.active,
            "sample_count": self.sample_count,
            "alpha": self.alpha,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeRuntimeState":
        return cls(
            rank=int(data["rank"]),
            node_id=str(data["node_id"]),
            current_batch=int(data["current_batch"]),
            compute_ms=float(data.get("compute_ms", 0.0)),
            compute_ema_ms=float(data.get("compute_ema_ms", 0.0)),
            runtime_std_ms=float(data.get("runtime_std_ms", 0.0)),
            comm_ms=float(data.get("comm_ms", 0.0)),
            idle_ms=float(data.get("idle_ms", 0.0)),
            active=bool(data.get("active", int(data["current_batch"]) > 0)),
            sample_count=int(data.get("sample_count", 0)),
            alpha=float(data.get("alpha", 0.2)),
        )


class OnlineClusterState:
    """Maintains collection of NodeRuntimeStates across all ranks in the cluster."""

    def __init__(self, ema_alpha: float = 0.2):
        self.ema_alpha = float(ema_alpha)
        self.nodes: Dict[str, NodeRuntimeState] = {}
        self.rank_to_node: Dict[int, str] = {}

    def register_node(self, rank: int, node_id: str, initial_batch: int) -> NodeRuntimeState:
        state = NodeRuntimeState(
            rank=rank,
            node_id=node_id,
            current_batch=initial_batch,
            active=(initial_batch > 0),
            alpha=self.ema_alpha,
        )
        self.nodes[node_id] = state
        self.rank_to_node[rank] = node_id
        return state

    def update_node(
        self,
        node_id: str,
        compute_ms: float,
        comm_ms: float,
        idle_ms: float,
        current_batch: int,
        runtime_std_ms: float = 0.0,
        rank: Optional[int] = None,
    ) -> NodeRuntimeState:
        if node_id not in self.nodes:
            r = rank if rank is not None else len(self.nodes)
            self.register_node(r, node_id, current_batch)

        state = self.nodes[node_id]
        state.update(
            compute_ms=compute_ms,
            comm_ms=comm_ms,
            idle_ms=idle_ms,
            current_batch=current_batch,
            runtime_std_ms=runtime_std_ms,
            alpha=self.ema_alpha,
        )
        return state

    def get_node(self, node_id: str) -> Optional[NodeRuntimeState]:
        return self.nodes.get(node_id)

    def get_node_by_rank(self, rank: int) -> Optional[NodeRuntimeState]:
        node_id = self.rank_to_node.get(rank)
        return self.nodes.get(node_id) if node_id else None

    def get_active_nodes(self) -> List[str]:
        return [node_id for node_id, s in self.nodes.items() if s.active]

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {node_id: s.to_dict() for node_id, s in self.nodes.items()}
