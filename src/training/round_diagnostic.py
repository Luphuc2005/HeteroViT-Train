"""Buffered round telemetry and root-referenced monotonic clock calibration.

Calibration runs only at startup and existing epoch boundaries, never inside a
model-sync round. Raw timestamps stay on each rank's perf_counter clock; offline
analysis interpolates the saved root-minus-local clock corrections.
"""
import csv
import json
import time
from pathlib import Path


class RoundDiagnostic:
    FIELDS = [
        'epoch', 'global_step', 'sync_round', 'rank', 'local_batch',
        'local_start_timestamp', 'local_done_timestamp',
        'gather_enter_timestamp', 'gather_exit_timestamp',
        'sync_enter_timestamp', 'sync_exit_timestamp',
        'compute_ms', 'optimizer_ms', 'gather_ms', 'model_sync_ms',
        'samples_processed', 'local_steps', 'compute_optimizer_ms',
        'data_ms', 'round_elapsed_ms',
    ]

    def __init__(self, comm, rank, run_dir, probes=9):
        self.comm, self.rank = comm, rank
        self.size = comm.Get_size()
        self.path = Path(run_dir)
        self.probes = probes
        self.rounds = []
        self.epochs = []
        self.calibrations = []
        self.calibration_elapsed_seconds = 0.0
        self.write_elapsed_seconds = 0.0

    def calibrate(self, epoch):
        """NTP-style four timestamps, minimum network-delay sample per peer.

        The half network RTT is an offset uncertainty estimate under arbitrary
        path asymmetry. Interpolation does not bound unobserved clock-rate changes.
        """
        started = time.perf_counter()
        records = []
        if self.rank == 0:
            root_now = time.perf_counter()
            records.append(dict(rank=0, epoch=epoch, local_timestamp=root_now,
                                root_timestamp=root_now, root_minus_local_seconds=0.0,
                                uncertainty_seconds=0.0, probes=[]))
        for peer in range(1, self.size):
            probes = []
            for probe in range(self.probes):
                if self.rank == 0:
                    t0 = time.perf_counter()
                    self.comm.send(probe, dest=peer, tag=27180)
                    t1, t2 = self.comm.recv(source=peer, tag=27181)
                    t3 = time.perf_counter()
                    delay = max(0.0, (t3 - t0) - (t2 - t1))
                    probes.append(dict(root_send=t0, worker_receive=t1,
                                       worker_send=t2, root_receive=t3,
                                       network_rtt_seconds=delay))
                elif self.rank == peer:
                    self.comm.recv(source=0, tag=27180)
                    t1 = time.perf_counter()
                    t2 = time.perf_counter()
                    self.comm.send((t1, t2), dest=0, tag=27181)
            if self.rank == 0:
                best = min(probes, key=lambda p: p['network_rtt_seconds'])
                local_mid = (best['worker_receive'] + best['worker_send']) / 2
                root_mid = (best['root_send'] + best['root_receive']) / 2
                records.append(dict(rank=peer, epoch=epoch, local_timestamp=local_mid,
                                    root_timestamp=root_mid,
                                    root_minus_local_seconds=root_mid - local_mid,
                                    uncertainty_seconds=best['network_rtt_seconds'] / 2,
                                    probes=probes))
        records = self.comm.bcast(records if self.rank == 0 else None, root=0)
        self.calibrations.append(next(r for r in records if r['rank'] == self.rank))
        self.calibration_elapsed_seconds += time.perf_counter() - started
        self.flush()

    def record_round(self, **row):
        self.rounds.append(dict(rank=self.rank, **row))

    def record_epoch(self, epoch, elapsed_training_seconds, val_accuracy, val_loss,
                     validation_complete_timestamp):
        self.epochs.append(dict(epoch=epoch, rank=self.rank,
                                elapsed_training_seconds=elapsed_training_seconds,
                                validation_complete_timestamp=validation_complete_timestamp,
                                val_accuracy=val_accuracy, val_loss=val_loss))

    def flush(self):
        started = time.perf_counter()
        round_path = self.path / ('round_diagnostic_rank%d.csv' % self.rank)
        with round_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self.rounds)
        with (self.path / ('epoch_elapsed_rank%d.csv' % self.rank)).open('w', newline='') as handle:
            fields = ['epoch', 'rank', 'elapsed_training_seconds', 'validation_complete_timestamp',
                      'val_accuracy', 'val_loss']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.epochs)
        metadata = dict(
            timestamp_clock='time.perf_counter, raw local seconds', reference_rank=0,
            alignment='piecewise linear root-minus-local offsets using adjacent epoch markers',
            calibration_probes=self.probes, calibration_elapsed_seconds=self.calibration_elapsed_seconds,
            write_elapsed_seconds_before_this_flush=self.write_elapsed_seconds,
            optimizer_timer='host perf_counter callbacks around unchanged apply_gradients; graph dependencies ensure optimizer completion; max replica envelope on rank0; deterministic ops remain enabled',
            compute_timer='local train-step wall time minus optimizer envelope; includes graph/host overhead',
            legacy_compute_timer='step_timing_breakdown.compute_ms still includes optimizer',
            epoch_elapsed_timer='rank0 training-loop elapsed immediately after validation, before checkpoint saving',
            calibrations=self.calibrations)
        (self.path / ('clock_calibration_rank%d.json' % self.rank)).write_text(json.dumps(metadata, indent=2) + '\n')
        self.write_elapsed_seconds += time.perf_counter() - started
