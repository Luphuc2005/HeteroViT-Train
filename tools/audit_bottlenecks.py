"""Audit existing FP16 runs without changing logs or launching training.

Run from the repository: python tools/audit_bottlenecks.py
Uses only the standard library and the existing telemetry-schema reader.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

from summarize_fp16_sweep import communication_rows
from results_layout import find_result


SWEEPS = [
    (42, 'fp16_sweep_uydYAoMC'),
    (43, 'fp16_sweep_seed43_RIHAAkW2'),
    (44, 'fp16_sweep_seed44_swTaOOce'),
]
BATCHES = [266, 6, 6, 6, 16]
FIELDS = ['data_ms', 'compute_ms', 'optimizer_ms', 'barrier_ms',
          'allreduce_ms', 'other_ms', 'step_ms']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def close(actual, expected, context, tolerance=0.02):
    require(abs(actual - expected) <= tolerance,
            '%s: %r != %r' % (context, actual, expected))


def audit_run(repo, sweep, seed, summary, paired=False):
    method = summary['method']
    base = find_result(repo, sweep)
    log_path = base / (method + '.log')
    log = log_path.read_text(errors='replace')
    completion = re.findall(r'MPI Local SGD Training finished in ([\d.]+)s', log)
    test = re.findall(r'Final Test Evaluation -> Loss: [\d.eE+-]+, Accuracy: ([\d.]+)%', log)
    require(len(completion) == len(test) == 1, 'Missing/duplicate completion: ' + str(log_path))
    require(not re.search(r'Traceback \(most recent call last\)|\[SYNC FAILED\]|Non-finite', log),
            'Failure marker: ' + str(log_path))
    duration, accuracy = float(completion[0]), float(test[0])
    close(duration, float(summary['training_time_seconds']), 'loop vs summary')
    close(accuracy, float(summary['test_accuracy_percent']), 'test vs summary', 0.001)
    result = dict(seed=seed, paired=paired, sweep=sweep, method=method,
                  log=str(log_path.relative_to(repo)), training_loop_seconds=duration,
                  test_accuracy_percent=accuracy, ranks=[])
    source = Path((base / 'source_stage.txt').read_text().strip())
    result['source_snapshot'] = str(source)
    result['source_sha256'] = {}
    for filename in ['trainer_mpi.py', 'fp16_communicator.py',
                     'fp16_delta_communicator.py', 'fp16_delta_ef_communicator.py']:
        path = source / 'src/training' / filename
        if path.exists():
            result['source_sha256'][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    step_records = []
    for rank in range(5):
        rank_dir = base / 'ranks' / method / ('rank%d' % rank)
        trace_path = rank_dir / ('step_timing_breakdown.csv' if rank == 0
                                 else 'step_timing_breakdown_rank%d.csv' % rank)
        trace = rows(trace_path)
        expected_steps = [(epoch, step) for epoch in range(1, 41) for step in range(1, 151)]
        require([(int(r['epoch']), int(r['step'])) for r in trace] == expected_steps,
                'Incomplete/out-of-order step trace: ' + str(trace_path))
        require(all(int(r['rank']) == rank and int(r['batch']) == BATCHES[rank] for r in trace),
                'Unexpected rank or batch: ' + str(trace_path))
        for row in trace:
            require(all(math.isfinite(float(row[k])) and float(row[k]) >= 0 for k in FIELDS),
                    'Invalid timing: ' + str(trace_path))
            close(sum(float(row[k]) for k in FIELDS if k != 'step_ms'),
                  float(row['step_ms']), 'step component sum', 0.001)
        totals = {k.replace('_ms', '_seconds'): sum(float(r[k]) for r in trace) / 1000
                  for k in FIELDS}
        comm_path = rank_dir / ('weight_communication_rank%d.csv' % rank)
        comm = communication_rows(comm_path)
        require(len(comm) == 160, 'Sync count: ' + str(comm_path))
        sync_steps = [int(r['step']) for r in trace if float(r['allreduce_ms']) > 0]
        require(sync_steps == [40, 80, 120, 150] * 40, 'Unexpected sync positions')
        payloads = {int(r['payload_bytes']) for r in comm}
        expected_payload = 10774312 if method == 'fp32' else 5387156
        require(payloads == {expected_payload}, 'Unexpected payload')
        comm_totals = dict(
            payload_call_seconds=sum(float(r['communication_ms']) for r in comm) / 1000,
            model_sync_seconds=sum(float(r['total_sync_ms']) for r in comm) / 1000,
            encode_seconds=sum(float(r.get('encode_ms', r['cast_to_fp16_ms'])) for r in comm) / 1000,
            decode_seconds=sum(float(r.get('decode_ms', r['cast_accumulation_ms'])) for r in comm) / 1000,
        )
        with comm_path.open(newline='') as handle:
            raw_count = sum(1 for _ in csv.reader(handle)) - 1
        result['ranks'].append(dict(rank=rank, trace=str(trace_path.relative_to(repo)),
                                  communication_csv=str(comm_path.relative_to(repo)),
                                  steps=len(trace), raw_communication_rows=raw_count,
                                  syncs=len(comm), batch=BATCHES[rank],
                                  payload_bytes_per_rank_sync=expected_payload,
                                  step_totals=totals, communication_totals=comm_totals))
        step_records.append(trace)
    close(max(r['communication_totals']['payload_call_seconds'] for r in result['ranks']),
          float(summary['max_rank_communication_total_ms']) / 1000, 'payload max vs summary')
    close(max(r['communication_totals']['model_sync_seconds'] for r in result['ranks']),
          float(summary['max_rank_sync_total_ms']) / 1000, 'sync max vs summary')
    timeline_path = base / 'ranks' / method / 'rank0/local_sgd_timeline.csv'
    timeline = rows(timeline_path)
    require(len(timeline) == 160, 'Timeline count')
    for row in timeline:
        count = int(row['total_samples_since_sync'])
        require(count == sum(int(row['rank%d_samples' % i]) for i in range(5)), 'Timeline samples')
        steps = 30 if int(row['step']) == 150 else 40
        require([int(row['rank%d_samples' % i]) for i in range(5)] == [b * steps for b in BATCHES],
                'Timeline per-rank samples')
    result['processed_samples'] = sum(int(r['total_samples_since_sync']) for r in timeline)
    require(result['processed_samples'] == 1800000, 'Sample budget')
    # Local work per window helps compare load, but does not establish actual
    # arrival order: ranks can leave previous syncs at different wall times.
    windows = []
    busiest = Counter()
    for epoch in range(1, 41):
        start = 0
        for end in [40, 80, 120, 150]:
            work = [sum(float(r['data_ms']) + float(r['compute_ms'])
                        for r in trace[(epoch - 1) * 150 + start:(epoch - 1) * 150 + end]) / 1000
                    for trace in step_records]
            maximum_rank = max(range(5), key=lambda i: work[i])
            busiest[maximum_rank] += 1
            windows.append(dict(epoch=epoch, end_step=end, local_work_seconds=work,
                                largest_local_work_rank=maximum_rank))
            start = end
    result['local_work_windows'] = windows
    result['largest_local_work_window_counts'] = dict(busiest)
    train_path = base / 'ranks' / method / 'rank0/train.csv'
    epochs = rows(train_path)
    require([int(r['epoch']) for r in epochs] == list(range(1, 41)), 'Epoch sequence')
    require({float(r['learning_rate']) for r in epochs} == {0.001}, 'Learning rate')
    result['train_csv'] = str(train_path.relative_to(repo))
    result['epoch_train_seconds'] = sum(float(r['train_time']) for r in epochs)
    result['validation_seconds'] = sum(float(r['val_time']) for r in epochs)
    result['sum_epoch_seconds'] = sum(float(r['epoch_time']) for r in epochs)
    result['loop_minus_sum_epoch_seconds'] = duration - result['sum_epoch_seconds']
    result['loop_throughput'] = result['processed_samples'] / duration
    result['epoch_train_throughput'] = result['processed_samples'] / result['epoch_train_seconds']
    result['best_val_percent'] = 100 * max(float(r['val_accuracy']) for r in epochs)
    result['val_percent_e10_e20_e30_e40'] = [100 * float(epochs[i - 1]['val_accuracy']) for i in [10, 20, 30, 40]]
    result['retrospective_targets'] = {}
    for target in [65, 66, 67]:
        elapsed = 0.0
        hit = None
        for row in epochs:
            elapsed += float(row['epoch_time'])
            if float(row['val_accuracy']) * 100 >= target - 1e-9:
                hit = dict(epoch=int(row['epoch']), sum_epoch_seconds=elapsed)
                break
        result['retrospective_targets'][str(target)] = hit
    return result


def audit(repo):
    runs = []
    for seed, sweep in SWEEPS + [(43, 'fp16_sweep_seed43_2tyCNQ2H')]:
        paired = sweep == 'fp16_sweep_seed43_2tyCNQ2H'
        for summary in rows(find_result(repo, sweep) / 'summary.csv'):
            runs.append(audit_run(repo, sweep, seed, summary, paired))
    aggregates = []
    for method in dict.fromkeys(r['method'] for r in runs if not r['paired']):
        selected = [r for r in runs if r['method'] == method and not r['paired']]
        mean_totals = {k: statistics.mean(r['ranks'][0]['step_totals'][k] for r in selected)
                       for k in selected[0]['ranks'][0]['step_totals']}
        aggregate = dict(method=method, n_seeds=len(selected),
                         test_mean_percent=statistics.mean(r['test_accuracy_percent'] for r in selected),
                         test_sample_sd_pp=statistics.stdev(r['test_accuracy_percent'] for r in selected),
                         loop_mean_seconds=statistics.mean(r['training_loop_seconds'] for r in selected),
                         rank0_step_mean_seconds=mean_totals)
        aggregate['rank0_step_fraction_percent'] = {k: 100 * v / mean_totals['step_seconds']
                                                   for k, v in mean_totals.items() if k != 'step_seconds'}
        aggregate['per_rank_mean'] = []
        for rank in range(5):
            aggregate['per_rank_mean'].append(dict(
                rank=rank,
                step_totals={k: statistics.mean(r['ranks'][rank]['step_totals'][k] for r in selected)
                             for k in mean_totals},
                communication_totals={k: statistics.mean(r['ranks'][rank]['communication_totals'][k] for r in selected)
                                      for k in selected[0]['ranks'][rank]['communication_totals']},
            ))
        aggregate['largest_local_work_window_counts'] = dict(sum(
            (Counter(r['largest_local_work_window_counts']) for r in selected), Counter()))
        aggregates.append(aggregate)
    fp32 = next(a for a in aggregates if a['method'] == 'fp32')
    for a in aggregates:
        a['loop_time_reduction_vs_fp32_percent'] = 100 * (1 - a['loop_mean_seconds'] / fp32['loop_mean_seconds'])
    legacy_base = find_result(repo, 'allreduce_baseline_300_20e_20261006_153923')
    legacy_trace = [r for r in rows(legacy_base / 'step_timing_breakdown.csv') if int(r['rank']) == 0]
    legacy = dict(source=str(legacy_base.relative_to(repo)), rank0_steps=len(legacy_trace),
                  rank0_totals={k.replace('_ms', '_seconds'): sum(float(r[k]) for r in legacy_trace) / 1000
                                for k in FIELDS})
    legacy['rank0_payload_call_fraction_of_steps_percent'] = (
        100 * legacy['rank0_totals']['allreduce_seconds'] / legacy['rank0_totals']['step_seconds'])
    return dict(date='2026-10-18', scope='17 existing full runs, 85 rank traces; no training launched',
                passed=True, semantics=dict(
                    compute='Forward/backward plus local optimizer in Local SGD',
                    wait='Elapsed sample-count gather; not a standalone barrier or pure network timer',
                    model_sync='Complete model sync in step trace, regardless of collective',
                    payload_call='Elapsed payload MPI call; includes waiting, not pure wire time',
                    local_work_windows='Data plus compute duration only; not synchronized timestamps or arrival order',
                    retrospective_targets='Sum epoch_time until first validation crossing; excludes some loop overhead, targets selected retrospectively; not exact wall time',
                    rank_maxima='Maximum of per-rank totals; never added to another rank or to a maximum of another metric'),
                aggregates=aggregates, legacy_allreduce=legacy, runs=runs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('docs/bottleneck_audit_20261018.json'))
    args = parser.parse_args()
    repo = args.repo.resolve()
    result = audit(repo)
    output = args.output if args.output.is_absolute() else repo / args.output
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print('PASS: audited %d runs / %d rank traces -> %s' % (len(result['runs']),
          sum(len(r['ranks']) for r in result['runs']), output))
