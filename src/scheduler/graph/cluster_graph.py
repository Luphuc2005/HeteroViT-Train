"""Cluster Graph G=(V, E) representation for distributed heterogeneous training.

Represents:
- V: Cluster nodes (heterogeneous compute resources: GPU/CPU, available batch sizes, compute profiles)
- E: Communication edges (pairwise latency, payload-dependent bandwidths, switch topology)
"""
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class NodeVertex:
    """Represents a computing node v in V."""
    node_id: str
    rank: int
    hostname: str
    device_type: str
    available_batches: List[int]
    max_batch: int
    empirical_compute: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active: bool = True
    runtime_std_ms: float = 0.0
    runtime_ema_ms: Optional[float] = None

    def get_profile_for_batch(self, batch_size: int) -> Optional[Dict[str, Any]]:
        return self.empirical_compute.get(str(batch_size))


@dataclass
class EdgeLink:
    """Represents a network communication link e in E."""
    edge_id: str
    node_u: str
    node_v: str
    rank_u: int
    rank_v: int
    latency_ms: float
    bandwidth_10mb_mb_s: float
    bandwidth_by_payload_mb_s: Dict[str, float] = field(default_factory=dict)
    is_bidirectional: bool = True


class ClusterGraph:
    """Cluster Graph G = (V, E) encapsulating compute vertices and network edges."""

    def __init__(
        self,
        nodes: Optional[Dict[str, NodeVertex]] = None,
        edges: Optional[Dict[Tuple[str, str], EdgeLink]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.nodes: Dict[str, NodeVertex] = nodes or {}
        self.edges: Dict[Tuple[str, str], EdgeLink] = edges or {}
        self.metadata: Dict[str, Any] = metadata or {}

    @classmethod
    def load_from_json(cls, json_path: str) -> "ClusterGraph":
        """Loads ClusterGraph from a cluster_graph.json file."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Cluster graph JSON not found at {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = {}
        for n_data in data.get("nodes", []):
            node = NodeVertex(
                node_id=n_data["node_id"],
                rank=int(n_data["rank"]),
                hostname=n_data["hostname"],
                device_type=n_data["device_type"],
                available_batches=[int(b) for b in n_data.get("available_batches", [])],
                max_batch=int(n_data.get("max_batch", max(n_data.get("available_batches", [128])))),
                empirical_compute=n_data.get("empirical_compute", {}),
                active=bool(n_data.get("active", True)),
                runtime_std_ms=float(n_data.get("runtime_std_ms", 0.0)),
                runtime_ema_ms=float(n_data["runtime_ema_ms"]) if n_data.get("runtime_ema_ms") is not None else None,
            )
            nodes[node.node_id] = node

        edges = {}
        for e_data in data.get("edges", []):
            edge = EdgeLink(
                edge_id=e_data["edge_id"],
                node_u=e_data["node_u"],
                node_v=e_data["node_v"],
                rank_u=int(e_data["rank_u"]),
                rank_v=int(e_data["rank_v"]),
                latency_ms=float(e_data["latency_ms"]),
                bandwidth_10mb_mb_s=float(e_data["bandwidth_10mb_mb_s"]),
                bandwidth_by_payload_mb_s={
                    k: float(v) for k, v in e_data.get("bandwidth_by_payload_mb_s", {}).items()
                },
                is_bidirectional=bool(e_data.get("is_bidirectional", True)),
            )
            edges[(edge.node_u, edge.node_v)] = edge
            if edge.is_bidirectional:
                edges[(edge.node_v, edge.node_u)] = edge

        return cls(nodes=nodes, edges=edges, metadata=data.get("metadata", {}))

    def get_node(self, node_id: str) -> Optional[NodeVertex]:
        return self.nodes.get(node_id)

    def get_node_by_rank(self, rank: int) -> Optional[NodeVertex]:
        for n in self.nodes.values():
            if n.rank == rank:
                return n
        return None

    def get_active_nodes(self) -> List[NodeVertex]:
        return [n for n in self.nodes.values() if n.active]

    def get_edge(self, u: str, v: str) -> Optional[EdgeLink]:
        return self.edges.get((u, v))

    def update_node_runtime(
        self,
        node_id: str,
        runtime_ema_ms: Optional[float] = None,
        runtime_std_ms: Optional[float] = None,
        active: Optional[bool] = None,
    ) -> None:
        """Updates dynamic runtime attributes of a node vertex."""
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found in cluster graph.")
        node = self.nodes[node_id]
        if runtime_ema_ms is not None:
            node.runtime_ema_ms = float(runtime_ema_ms)
        if runtime_std_ms is not None:
            node.runtime_std_ms = float(runtime_std_ms)
        if active is not None:
            node.active = bool(active)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes ClusterGraph to dictionary."""
        unique_edges = []
        seen_edges = set()
        for edge in self.edges.values():
            if edge.edge_id not in seen_edges:
                seen_edges.add(edge.edge_id)
                unique_edges.append({
                    "edge_id": edge.edge_id,
                    "node_u": edge.node_u,
                    "node_v": edge.node_v,
                    "rank_u": edge.rank_u,
                    "rank_v": edge.rank_v,
                    "latency_ms": edge.latency_ms,
                    "bandwidth_10mb_mb_s": edge.bandwidth_10mb_mb_s,
                    "bandwidth_by_payload_mb_s": edge.bandwidth_by_payload_mb_s,
                    "is_bidirectional": edge.is_bidirectional,
                })

        return {
            "metadata": self.metadata,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "rank": n.rank,
                    "hostname": n.hostname,
                    "device_type": n.device_type,
                    "available_batches": n.available_batches,
                    "max_batch": n.max_batch,
                    "empirical_compute": n.empirical_compute,
                    "active": n.active,
                    "runtime_std_ms": n.runtime_std_ms,
                    "runtime_ema_ms": n.runtime_ema_ms,
                }
                for n in self.nodes.values()
            ],
            "edges": unique_edges,
        }
