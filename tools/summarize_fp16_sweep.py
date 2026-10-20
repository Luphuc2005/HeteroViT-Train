"""Validate matched configs and summarize selected-method sweep reports.

The original whole-weight FP16 CSV has two rows per sync and an incomplete
header. Only trainer rows containing total_sync_ms enter timing aggregates.
"""
import argparse
import copy
import csv
import json
import math
import re
from pathlib import Path

METHODS = {
    'fp32': ('local_sgd_h40_fp32_40e.yaml', 'fp32', 'allreduce'),
    'fp16_whole_allgather': ('local_sgd_h40_fp16_allgather_40e.yaml', 'fp16', 'allgather'),
    'fp16_whole_allreduce': ('local_sgd_h40_fp16_40e.yaml', 'fp16', 'allreduce'),
    'fp16_delta_allgather': ('local_sgd_h40_fp16_delta_allgather_40e.yaml', 'fp16_delta_allgather', 'allgather'),
    'fp16_delta_allgather_ef': ('local_sgd_h40_fp16_delta_allgather_ef_40e.yaml', 'fp16_delta_allgather', 'allgather'),
    'fp16_delta_allreduce': ('local_sgd_h40_fp16_delta_allreduce_40e.yaml', 'fp16_delta_allreduce', 'allreduce'),
}


def check_configs(root, methods=None):
    import yaml
    methods = list(METHODS) if methods is None else methods
    common = None
    for method in methods:
        filename, precision, collective = METHODS[method]
        config = yaml.safe_load((root / 'configs/mpi' / filename).read_text())
        training = config['training']
        if (training.get('comm_precision', 'fp32') != precision
                or training.get('comm_collective', 'allreduce') != collective
                or training.get('comm_error_feedback', False) != (method == 'fp16_delta_allgather_ef')
                or training.get('comm_verify_reference', False)
                or training.get('verify_sync_every_round', False)):
            raise ValueError('Invalid communication settings for ' + method)
        if (config['seed'] != 42 or training['epochs'] != 40
                or training['local_sgd_h'] != 40 or training['sync_mode'] != 'local_sgd'
                or training['rank_batch_sizes'] != [266, 6, 6, 6, 16]):
            raise ValueError('Invalid fixed experiment settings for ' + method)
        normalized = copy.deepcopy(config)
        normalized.pop('experiment')
        for field in ('comm_precision', 'comm_collective', 'comm_verify_reference', 'verify_sync_every_round', 'comm_error_feedback'):
            normalized['training'].pop(field, None)
        if common is None:
            common = normalized
        elif normalized != common:
            raise ValueError('Unmatched model/training/runtime settings for ' + method)
    print('PASS: %d matched base configs; H=40, base seed=42, reference=false' % len(methods))


def communication_rows(path):
    with path.open(newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = []
        for values in reader:
            if not values:
                continue
            fields = header
            if len(values) == len(header) + 1 and 'total_sync_ms' not in header:
                fields = header + ['total_sync_ms']
            elif len(values) != len(header):
                raise ValueError('Unexpected telemetry schema in ' + str(path))
            row = dict(zip(fields, values))
            if 'total_sync_ms' not in row:
                # Communicator-only duplicate; trainer emits the complete row.
                continue
            for key in ('communication_ms', 'total_sync_ms', 'cast_to_fp16_ms', 'cast_accumulation_ms', 'payload_bytes'):
                number = float(row[key])
                if not math.isfinite(number) or number < 0:
                    raise ValueError('Invalid telemetry value in ' + str(path))
            rows.append(row)
    if not rows:
        raise ValueError('Missing complete synchronization telemetry: ' + str(path))
    return rows


def summarize(root, methods=None):
    methods = list(METHODS) if methods is None else methods
    summaries = []
    for method in methods:
        rank_metrics = []
        for rank in range(5):
            path = root / 'ranks' / method / ('rank%d' % rank) / ('weight_communication_rank%d.csv' % rank)
            if not path.exists():
                break  # Partial results are summarized after each completed method.
            rows = communication_rows(path)
            rank_metrics.append(dict(rank=rank, synchronization_rounds=len(rows),
                communication_total_ms=sum(float(r['communication_ms']) for r in rows),
                sync_total_ms=sum(float(r['total_sync_ms']) for r in rows),
                encode_total_ms=sum(float(r.get('encode_ms', r['cast_to_fp16_ms'])) for r in rows),
                decode_total_ms=sum(float(r.get('decode_ms', r['cast_accumulation_ms'])) for r in rows),
                payload_bytes=int(rows[0]['payload_bytes'])))
        if len(rank_metrics) != 5:
            continue
        if len({r['synchronization_rounds'] for r in rank_metrics}) != 1:
            raise ValueError('Synchronization round counts differ across ranks: ' + method)
        text = (root / (method + '.log')).read_text(errors='replace')
        accuracy = re.search(r'Final Test Evaluation -> Loss: [0-9.eE+-]+, Accuracy: ([0-9.eE+-]+)%', text)
        duration = re.search(r'MPI Local SGD Training finished in ([0-9.eE+-]+)s', text)
        if duration is None:
            raise ValueError('Missing training completion: ' + method)
        summaries.append(dict(method=method, test_accuracy_percent=float(accuracy.group(1)) if accuracy else None,
            training_time_seconds=float(duration.group(1)), synchronization_rounds=rank_metrics[0]['synchronization_rounds'],
            max_rank_communication_total_ms=max(r['communication_total_ms'] for r in rank_metrics),
            max_rank_sync_total_ms=max(r['sync_total_ms'] for r in rank_metrics),
            max_rank_encode_total_ms=max(r['encode_total_ms'] for r in rank_metrics),
            max_rank_decode_total_ms=max(r['decode_total_ms'] for r in rank_metrics),
            payload_bytes_per_rank=rank_metrics[0]['payload_bytes'], rank_metrics=rank_metrics))
    (root / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    with (root / 'summary.csv').open('w', newline='') as handle:
        fields = [k for k in summaries[0] if k != 'rank_metrics'] if summaries else ['method']
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summaries)
    print('Summarized %d/%d methods: %s' % (len(summaries), len(methods), root / 'summary.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-configs', action='store_true')
    parser.add_argument('--methods', nargs='+', choices=list(METHODS), default=None)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    if args.check_configs:
        check_configs(args.root, args.methods)
    else:
        from results_layout import resolve_result_path
        summarize(resolve_result_path(args.root), args.methods)
