"""Online profiling and candidate-search scheduler for heterogeneous workers.

The scheduler is deliberately framework agnostic: it only consumes worker timing
profiles and returns an allocation.  TensorFlow/process synchronization stays in
the trainer integration.
"""
from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class WorkerRuntimeProfile:
    """Smoothed runtime observation for one worker at one batch size."""

    worker_id: str
    device_type: str
    batch_size: int
    raw_time_ms: float
    ema_time_ms: float
    throughput_img_per_ms: float
    sample_count: int


@dataclass(frozen=True)
class CandidatePrediction:
    """Predicted worker runtimes and objective values for an allocation."""

    allocation: Dict[str, int]
    worker_times_ms: Dict[str, float]
    delta_ms: float
    critical_ms: float


@dataclass(frozen=True)
class SchedulerDecision:
    """Auditable result produced at one scheduler decision window."""

    step: int
    current_allocation: Dict[str, int]
    target_allocation: Dict[str, int]
    current_prediction: CandidatePrediction
    target_prediction: CandidatePrediction
    profiles: Dict[str, WorkerRuntimeProfile]
    action: str
    reason: str
    improvement_pct: float
    cooldown_remaining: int


class PerformanceHistory:
    """Per-worker, per-batch empirical EMA history and runtime estimator."""

    def __init__(self, ema_alpha: float):
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        self.ema_alpha = float(ema_alpha)
        self._profiles: Dict[str, Dict[int, WorkerRuntimeProfile]] = {}
        self._latest: Dict[str, WorkerRuntimeProfile] = {}

    @staticmethod
    def _valid_observation(batch_size: int, raw_time_ms: float) -> bool:
        return (
            isinstance(batch_size, int)
            and batch_size > 0
            and math.isfinite(raw_time_ms)
            and raw_time_ms > 0.0
        )

    def observe(
        self,
        worker_id: str,
        device_type: str,
        batch_size: int,
        raw_time_ms: float,
    ) -> Optional[WorkerRuntimeProfile]:
        """Records a valid timing, returning ``None`` for invalid observations."""
        batch_size = int(batch_size)
        raw_time_ms = float(raw_time_ms)
        if not self._valid_observation(batch_size, raw_time_ms):
            return None

        worker_profiles = self._profiles.setdefault(worker_id, {})
        previous = worker_profiles.get(batch_size)
        if previous is None:
            ema_time_ms = raw_time_ms
            sample_count = 1
        else:
            ema_time_ms = (
                self.ema_alpha * raw_time_ms
                + (1.0 - self.ema_alpha) * previous.ema_time_ms
            )
            sample_count = previous.sample_count + 1

        profile = WorkerRuntimeProfile(
            worker_id=str(worker_id),
            device_type=str(device_type),
            batch_size=batch_size,
            raw_time_ms=raw_time_ms,
            ema_time_ms=ema_time_ms,
            throughput_img_per_ms=float(batch_size) / ema_time_ms,
            sample_count=sample_count,
        )
        worker_profiles[batch_size] = profile
        self._latest[worker_id] = profile
        return profile

    def latest_profiles(
        self, worker_ids: Optional[Iterable[str]] = None
    ) -> List[WorkerRuntimeProfile]:
        if worker_ids is None:
            return list(self._latest.values())
        return [self._latest[w] for w in worker_ids if w in self._latest]

    def profile_at(self, worker_id: str, batch_size: int) -> Optional[WorkerRuntimeProfile]:
        return self._profiles.get(worker_id, {}).get(int(batch_size))

    def predict_time_ms(
        self,
        worker_id: str,
        batch_size: int,
        current_profile: Optional[WorkerRuntimeProfile] = None,
    ) -> float:
        """Predicts runtime using exact EMA, interpolation, then local throughput."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("candidate batch size must be positive")

        worker_profiles = self._profiles.get(worker_id, {})
        exact = worker_profiles.get(batch_size)
        if exact is not None:
            return exact.ema_time_ms

        observed_sizes = sorted(worker_profiles)
        lower = [size for size in observed_sizes if size < batch_size]
        upper = [size for size in observed_sizes if size > batch_size]
        if lower and upper:
            lo_size = lower[-1]
            hi_size = upper[0]
            lo_time = worker_profiles[lo_size].ema_time_ms
            hi_time = worker_profiles[hi_size].ema_time_ms
            ratio = float(batch_size - lo_size) / float(hi_size - lo_size)
            prediction = lo_time + ratio * (hi_time - lo_time)
            if math.isfinite(prediction) and prediction > 0.0:
                return prediction

        reference = current_profile or self._latest.get(worker_id)
        if reference is None or reference.throughput_img_per_ms <= 0.0:
            raise ValueError(f"no valid performance profile for worker {worker_id!r}")
        prediction = float(batch_size) / reference.throughput_img_per_ms
        if not math.isfinite(prediction) or prediction <= 0.0:
            raise ValueError(f"invalid runtime prediction for worker {worker_id!r}")
        return prediction


class DynamicWorkloadScheduler:
    """EMA-based candidate search with improvement and cooldown guards."""

    def __init__(
        self,
        candidates: Sequence[Mapping[str, int]],
        global_batch: int,
        worker_device_types: Mapping[str, str],
        fallback_allocation: Mapping[str, int],
        enabled: bool = True,
        apply_changes: bool = False,
        ema_alpha: float = 0.15,
        decision_interval_steps: int = 20,
        cooldown_steps: int = 20,
        delta_threshold_ms: float = 5.0,
        min_improvement_pct: float = 1.0,
        warmup_steps: int = 10,
        settling_steps: int = 2,
        stability_windows: int = 2,
    ):
        self.enabled = bool(enabled)
        self.apply_changes = bool(apply_changes)
        self.global_batch = int(global_batch)
        self.worker_device_types = dict(worker_device_types)
        self.worker_ids = tuple(self.worker_device_types)
        self.fallback_allocation = {
            worker_id: int(batch_size)
            for worker_id, batch_size in fallback_allocation.items()
        }
        self.decision_interval_steps = max(1, int(decision_interval_steps))
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.delta_threshold_ms = max(0.0, float(delta_threshold_ms))
        self.min_improvement_pct = max(0.0, float(min_improvement_pct))
        self.warmup_steps = max(0, int(warmup_steps))
        self.settling_steps = max(0, int(settling_steps))
        self.stability_windows = max(1, int(stability_windows))
        self.history = PerformanceHistory(ema_alpha=float(ema_alpha))
        self.last_switch_step: Optional[int] = None
        self.decision_count = 0
        self.switch_count = 0
        self._pending_target: Optional[Tuple[int, ...]] = None
        self._pending_target_windows = 0

        if self.global_batch <= 0:
            raise ValueError("global_batch must be positive")
        if not self.worker_ids:
            raise ValueError("at least one worker is required")
        self._validate_allocation(self.fallback_allocation)

        valid_candidates: List[Dict[str, int]] = []
        seen: set = set()
        for candidate in candidates:
            try:
                normalized = {
                    worker_id: int(candidate[worker_id]) for worker_id in self.worker_ids
                }
                self._validate_allocation(normalized)
            except (KeyError, TypeError, ValueError):
                continue
            key = tuple(normalized[worker_id] for worker_id in self.worker_ids)
            if key not in seen:
                valid_candidates.append(normalized)
                seen.add(key)

        fallback_key = tuple(self.fallback_allocation[w] for w in self.worker_ids)
        if fallback_key not in seen:
            valid_candidates.append(dict(self.fallback_allocation))
        self.candidates = valid_candidates

    @classmethod
    def from_config(
        cls,
        scheduler_config: Mapping[str, Any],
        initial_allocation: Mapping[str, int],
    ) -> "DynamicWorkloadScheduler":
        cfg = dict(scheduler_config)
        worker_device_types = dict(
            cfg.get(
                "workers",
                {"lab01_gpu": "gpu", "lab01_cpu": "cpu"},
            )
        )
        global_batch = int(cfg.get("global_batch", sum(initial_allocation.values())))
        fallback = dict(cfg.get("fallback_allocation", initial_allocation))
        candidates = cfg.get("candidates", [initial_allocation])
        return cls(
            candidates=candidates,
            global_batch=global_batch,
            worker_device_types=worker_device_types,
            fallback_allocation=fallback,
            enabled=bool(cfg.get("enabled", True)),
            apply_changes=bool(cfg.get("apply_changes", False)),
            ema_alpha=float(cfg.get("ema_alpha", 0.15)),
            decision_interval_steps=int(cfg.get("decision_interval_steps", 20)),
            cooldown_steps=int(cfg.get("cooldown_steps", 20)),
            delta_threshold_ms=float(cfg.get("delta_threshold_ms", 5.0)),
            min_improvement_pct=float(cfg.get("min_improvement_pct", 1.0)),
            warmup_steps=int(cfg.get("warmup_steps", 10)),
            settling_steps=int(cfg.get("settling_steps", 2)),
            stability_windows=int(cfg.get("stability_windows", 2)),
        )

    def _validate_allocation(self, allocation: Mapping[str, int]) -> None:
        if set(allocation) != set(self.worker_ids):
            raise ValueError("allocation worker IDs do not match configured workers")
        batches = [int(allocation[w]) for w in self.worker_ids]
        if any(batch <= 0 for batch in batches):
            raise ValueError("all worker batch sizes must be positive")
        if sum(batches) != self.global_batch:
            raise ValueError(
                f"allocation sum {sum(batches)} does not match global batch {self.global_batch}"
            )

    def observe(
        self,
        worker_id: str,
        batch_size: int,
        raw_time_ms: float,
    ) -> Optional[WorkerRuntimeProfile]:
        if worker_id not in self.worker_device_types:
            return None
        return self.history.observe(
            worker_id=worker_id,
            device_type=self.worker_device_types[worker_id],
            batch_size=int(batch_size),
            raw_time_ms=float(raw_time_ms),
        )

    def solve(
        self,
        profiles: List[WorkerRuntimeProfile],
        global_batch: int,
    ) -> CandidatePrediction:
        """Returns the best configured allocation for the supplied worker profiles."""
        if int(global_batch) != self.global_batch:
            raise ValueError("solve global_batch does not match scheduler global_batch")
        profile_map = {profile.worker_id: profile for profile in profiles}
        if any(worker_id not in profile_map for worker_id in self.worker_ids):
            raise ValueError("missing current worker profile")

        predictions = [self._predict(candidate, profile_map) for candidate in self.candidates]
        within_delta = [
            prediction
            for prediction in predictions
            if prediction.delta_ms <= self.delta_threshold_ms
        ]
        pool = within_delta or predictions
        return min(
            pool,
            key=lambda prediction: (
                prediction.critical_ms,
                prediction.delta_ms,
                tuple(prediction.allocation[w] for w in self.worker_ids),
            ),
        )

    def _predict(
        self,
        allocation: Mapping[str, int],
        profile_map: Mapping[str, WorkerRuntimeProfile],
    ) -> CandidatePrediction:
        worker_times = {
            worker_id: self.history.predict_time_ms(
                worker_id,
                int(allocation[worker_id]),
                current_profile=profile_map[worker_id],
            )
            for worker_id in self.worker_ids
        }
        times = list(worker_times.values())
        return CandidatePrediction(
            allocation={worker_id: int(allocation[worker_id]) for worker_id in self.worker_ids},
            worker_times_ms=worker_times,
            delta_ms=max(times) - min(times),
            critical_ms=max(times),
        )

    def cooldown_remaining(self, step: int) -> int:
        if self.last_switch_step is None:
            return 0
        elapsed = max(0, int(step) - self.last_switch_step)
        return max(0, self.cooldown_steps - elapsed)

    def should_decide(self, step: int) -> bool:
        return (
            self.enabled
            and int(step) > self.warmup_steps
            and int(step) % self.decision_interval_steps == 0
        )

    def should_observe(self, step: int) -> bool:
        """Rejects global and post-switch graph/device warm-up observations."""
        step = int(step)
        if step <= self.warmup_steps:
            return False
        if self.last_switch_step is None:
            return True
        return step - self.last_switch_step > self.settling_steps

    def maybe_decide(
        self,
        step: int,
        current_allocation: Mapping[str, int],
    ) -> Optional[SchedulerDecision]:
        if not self.should_decide(step):
            return None

        current = {worker_id: int(current_allocation[worker_id]) for worker_id in self.worker_ids}
        self._validate_allocation(current)
        profiles = self.history.latest_profiles(self.worker_ids)
        if len(profiles) != len(self.worker_ids):
            return None
        profile_map = {profile.worker_id: profile for profile in profiles}

        target_prediction = self.solve(profiles, self.global_batch)
        current_prediction = self._predict(current, profile_map)
        improvement_pct = (
            (current_prediction.critical_ms - target_prediction.critical_ms)
            / current_prediction.critical_ms
            * 100.0
        )
        remaining = self.cooldown_remaining(step)
        same_allocation = target_prediction.allocation == current

        if not self.apply_changes:
            action = "OBSERVE_ONLY"
            reason = "predicted_keep" if same_allocation else "predicted_better_allocation"
        elif same_allocation:
            action = "KEEP"
            reason = "current_allocation_is_best"
            self._reset_pending_target()
        elif remaining > 0:
            action = "KEEP"
            reason = "cooldown_active"
            self._reset_pending_target()
        elif improvement_pct <= self.min_improvement_pct:
            action = "KEEP"
            reason = "improvement_below_threshold"
            self._reset_pending_target()
        else:
            target_key = tuple(
                target_prediction.allocation[worker_id] for worker_id in self.worker_ids
            )
            if target_key == self._pending_target:
                self._pending_target_windows += 1
            else:
                self._pending_target = target_key
                self._pending_target_windows = 1

            if self._pending_target_windows < self.stability_windows:
                action = "KEEP"
                reason = (
                    f"target_stability_{self._pending_target_windows}/"
                    f"{self.stability_windows}"
                )
            else:
                action = "SWITCH"
                reason = "predicted_critical_time_improved"
                self.last_switch_step = int(step)
                remaining = self.cooldown_steps
                self.switch_count += 1
                self._reset_pending_target()

        self.decision_count += 1
        return SchedulerDecision(
            step=int(step),
            current_allocation=current,
            target_allocation=dict(target_prediction.allocation),
            current_prediction=current_prediction,
            target_prediction=target_prediction,
            profiles=profile_map,
            action=action,
            reason=reason,
            improvement_pct=improvement_pct,
            cooldown_remaining=remaining,
        )

    def _reset_pending_target(self) -> None:
        self._pending_target = None
        self._pending_target_windows = 0
