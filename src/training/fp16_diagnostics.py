"""Compare FP16 representations on the same inputs along an FP32 trajectory."""
import csv
from pathlib import Path

import numpy as np

from src.training.fp16_communicator import FP16Communicator
from src.training.fp16_delta_communicator import FP16DeltaCommunicator


class SameInputFP16Diagnostic:
    """Shadow collectives only; the caller applies the actual FP32 reference."""

    def __init__(self, comm, anchor):
        self.whole = FP16Communicator(comm, collective='allgather')
        self.delta = FP16DeltaCommunicator(comm, collective='allgather')
        self.delta.set_anchor(anchor)
        self.round = 0

    def compare(self, local_weights, fp32_reference, samples, policy='sample_weighted'):
        # The trainer has already computed/applied fp32_reference from exactly
        # these local_weights. Both shadow methods start from the previous
        # applied FP32 model, never from their own shadow result.
        anchor = self.delta.global_prev.copy()
        whole, _, whole_stats = self.whole.synchronize(local_weights, samples, policy)
        delta, _, delta_stats = self.delta.synchronize(local_weights, samples, policy)
        update = fp32_reference.astype(np.float64) - anchor.astype(np.float64)
        update_norm = float(np.linalg.norm(update))
        record = dict(round=self.round + 1, samples=int(samples),
                      num_elements=int(local_weights.size),
                      reference_update_l2=update_norm,
                      local_delta_l2=float(np.linalg.norm(
                          local_weights.astype(np.float64) - anchor.astype(np.float64))))
        for name, result, stats in [('whole', whole, whole_stats), ('delta', delta, delta_stats)]:
            error = result.astype(np.float64) - fp32_reference.astype(np.float64)
            error_norm = float(np.linalg.norm(error))
            record.update({
                name + '_error_l2': error_norm,
                name + '_error_relative_to_update': error_norm / (update_norm + 1e-12),
                name + '_max_absolute_error': float(np.max(np.abs(error))),
                name + '_encode_ms': stats.get('encode_ms', stats['cast_to_fp16_ms']),
                name + '_decode_ms': stats.get('decode_ms', stats['cast_accumulation_ms']),
                name + '_payload_collective_ms': stats['communication_ms'],
                name + '_payload_bytes': stats['payload_bytes'],
            })
        record['delta_lower_error'] = int(record['delta_error_l2'] < record['whole_error_l2'])
        # Also collectively validate the reference before making it the next anchor.
        self.delta.set_anchor(fp32_reference)
        self.round += 1
        return record


def install_same_input_diagnostic(trainer):
    """Wrap one trainer after normal MPI/device/TF initialization is complete."""
    if (trainer.comm_precision != 'fp32' or trainer.max_steps is None
            or trainer.sync_mode not in ('local_sgd', 'periodic_averaging')):
        raise ValueError('Same-input diagnosis requires FP32 Local SGD and --max-steps')

    def flat_weights():
        return np.concatenate([w.ravel() for w in trainer.model.get_weights()]).astype(np.float32)

    trainer.fp16_shadow = SameInputFP16Diagnostic(trainer.comm, flat_weights())
    trainer.fp16_shadow_rows = []
    original_sync = trainer.synchronize_model_weights

    def diagnostic_sync(policy='sample_weighted', samples_since_sync=0, verify_weights=False):
        local = flat_weights()
        result = original_sync(policy=policy, samples_since_sync=samples_since_sync,
                               verify_weights=verify_weights)
        # Only original_sync applies weights. The shadows cannot mutate either
        # the trainer model or its optimizer; they receive copied NumPy arrays.
        row = trainer.fp16_shadow.compare(local, flat_weights(), samples_since_sync, policy)
        trainer.fp16_shadow_rows.append(row)
        path = Path(trainer.logger.run_dir) / ('fp16_same_inputs_rank%d.csv' % trainer.rank)
        with path.open('a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if len(trainer.fp16_shadow_rows) == 1:
                writer.writeheader()
            writer.writerow(row)
        return result

    trainer.synchronize_model_weights = diagnostic_sync
