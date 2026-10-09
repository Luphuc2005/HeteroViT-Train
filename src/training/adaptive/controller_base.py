"""Base Interface and Telemetry Records for Adaptive Local SGD Controllers.

Design principles:
  - Pure separation of concerns: Controller decides next H only.
  - Controller does NOT perform MPI collectives or modify model/optimizer states.
  - One authoritative rank decides next H, then broadcasts to all workers.
  - Supports state serialization and cold-start recovery.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from .consensus_metrics import ConsensusMetrics


@dataclass
class SyncRecord:
    """Telemetry recorded at each model synchronization boundary."""
    round_idx: int
    epoch: int
    global_step: int
    sync_reason: str               # 'H_REACHED', 'EPOCH_END', 'TRAIN_END'
    is_forced: bool
    actual_local_steps: int        # h_t
    planned_h: int                 # H executed for this block
    samples_since_sync: int        # Local worker samples n_i
    total_samples_since_sync: int  # Cluster-wide samples sum_j n_j
    metrics: ConsensusMetrics      # v_t, q_t, m_t, etc.
    compute_time_sec: float        # Local compute time since last sync
    sync_time_sec: float           # Model averaging wall-clock time
    overhead_sec: float            # Measurement/controller overhead time
    communicated_bytes: int        # Bytes transmitted during model sync


@dataclass
class AdaptiveDecision:
    """Decision produced by an adaptive controller for the subsequent block."""
    next_h: int
    decision: str                  # 'KEEP', 'INCREASE', 'DECREASE', 'SKIP_PARTIAL', 'FALLBACK_MIN', etc.
    reason: str                    # Detailed explanation / feasibility rationale
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptiveHControllerBase(ABC):
    """Abstract Base Class for Synchronization Period Controllers."""

    def __init__(
        self,
        candidates: List[int],
        initial_h: int,
        steps_per_epoch: int = 150,
        adjacent_step_only: bool = True,
    ):
        if not candidates or any(c <= 0 for c in candidates):
            raise ValueError(f"Candidates must be non-empty positive integers: {candidates}")
        if candidates != sorted(set(candidates)):
            raise ValueError(f"Candidates must be strictly sorted without duplicates: {candidates}")
        if initial_h not in candidates:
            raise ValueError(f"initial_h ({initial_h}) must be in candidates ({candidates})")

        self.candidates = [int(c) for c in candidates]
        self.initial_h = int(initial_h)
        self.current_h = int(initial_h)
        self.steps_per_epoch = int(steps_per_epoch)
        self.adjacent_step_only = bool(adjacent_step_only)
        self.sync_rounds = 0
        self.last_record: Optional[SyncRecord] = None
        self.last_decision: Optional[AdaptiveDecision] = None

    def clamp_transition(self, candidate_h: int) -> int:
        """Restricts transition to at most one adjacent candidate step if enabled."""
        if not self.adjacent_step_only:
            return candidate_h
        curr_idx = self.candidates.index(self.current_h)
        target_idx = self.candidates.index(candidate_h)
        if target_idx > curr_idx:
            return self.candidates[curr_idx + 1]
        elif target_idx < curr_idx:
            return self.candidates[curr_idx - 1]
        return self.current_h

    @abstractmethod
    def observe(self, record: SyncRecord) -> None:
        """Observes telemetry from a completed synchronization block."""
        pass

    @abstractmethod
    def propose_next_h(self, steps_remaining_in_epoch: Optional[int] = None) -> AdaptiveDecision:
        """Computes the recommended synchronization interval for the next block."""
        pass

    def process_sync_event(
        self,
        record: SyncRecord,
        steps_remaining_in_epoch: Optional[int] = None,
    ) -> AdaptiveDecision:
        """Unified lifecycle method: records observation and proposes next H."""
        self.sync_rounds += 1
        self.last_record = record
        self.observe(record)

        # Standard policy: do not adapt controller policy on forced partial blocks
        # to ensure fair comparability with fixed baselines and baseline V1
        if record.is_forced and record.actual_local_steps < record.planned_h:
            decision = AdaptiveDecision(
                next_h=self.current_h,
                decision="SKIP_PARTIAL",
                reason=(
                    f"Partial block (actual_steps={record.actual_local_steps} < "
                    f"planned={record.planned_h}). Preserving current H."
                ),
                metadata={"partial_block": True, "controller_type": self.__class__.__name__},
            )
        else:
            decision = self.propose_next_h(steps_remaining_in_epoch=steps_remaining_in_epoch)

        self.last_decision = decision
        self.current_h = decision.next_h
        return decision

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Serializes internal controller state."""
        pass

    @abstractmethod
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restores internal controller state."""
        pass

    def reset(self) -> None:
        """Resets controller to initial condition."""
        self.current_h = self.initial_h
        self.sync_rounds = 0
        self.last_record = None
        self.last_decision = None

