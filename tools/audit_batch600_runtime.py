"""Reconstruct batch600 runtime and cross-rank readiness from immutable run artifacts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import difflib

import yaml
from index_results import SavedConfigLoader
from paired_workload_analysis import align, stats
from results_layout import find_result, resolve_result_path

RUN = 'batch600_40e_seed42_20261019_160951_jbd2gj9b'
ALLOCATIONS = {'gpu_only': [600], 'cluster5': [532, 12, 12, 12, 32]}
TIMESTAMPS = ['local_start_timestamp', 'local_done_timestamp', 'gather_enter_timestamp',
              'gather_exit_timestamp', 'sync_enter_timestamp', 'sync_exit_timestamp']


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        found = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                found.append(dict(field=prefix + key, gpu=a.get(key), cluster=b.get(key)))
            else:
                found.extend(differences(a[key], b[key], prefix + key + '.'))
        return found
    return [] if a == b else [dict(field=prefix.rstrip('.'), gpu=a, cluster=b)]


def audit_variant(root, variant, allocation):
    text = (root / (variant + '.log')).read_text(errors='replace')
    assert not re.search(r'Traceback \(most recent call last\)|\[SYNC FAILED\]|Non-finite|ResourceExhaustedError', text)
    duration = float(re.findall(r'Total Training Time\s*:\s*([\d.]+)s', text)[0])
    ranks, events = [], []
    for rank, batch in enumerate(allocation):
        folder = root / 'ranks' / variant / ('rank%d' % rank)
        raw = rows(folder / ('round_diagnostic_rank%d.csv' % rank))
        step = rows(folder / ('step_timing_breakdown.csv' if rank == 0 else 'step_timing_breakdown_rank%d.csv' % rank))
        comm = rows(folder / ('weight_communication_rank%d.csv' % rank))
        clock = json.loads((folder / ('clock_calibration_rank%d.json' % rank)).read_text())
        assert len(raw) == len(comm) == 160 and len(step) == 3000
        assert len(clock['calibrations']) == 41
        assert [(int(r['epoch']), int(r['step'])) for r in step] == [(e, s) for e in range(1, 41) for s in range(1, 76)]
        for s in step:
            assert abs(sum(float(s[k]) for k in ['data_ms', 'compute_ms', 'optimizer_ms',
                'barrier_ms', 'allreduce_ms', 'other_ms']) - float(s['step_ms'])) < .001
        assert {int(r['payload_bytes']) for r in comm} == {10774312}
        for index, record in enumerate(raw):
            row = {k: float(v) for k, v in record.items()}
            assert all(math.isfinite(v) for v in row.values())
            assert row['rank'] == rank and row['local_batch'] == batch and row['sync_round'] == index + 1
            epoch, position = divmod(index, 4)
            assert row['epoch'] == epoch + 1 and row['global_step'] == epoch * 75 + [20, 40, 60, 75][position]
            assert row['local_steps'] == [20, 20, 20, 15][position]
            assert row['samples_processed'] == batch * row['local_steps']
            for field in TIMESTAMPS:
                row['aligned_' + field], error = align(row[field], clock['calibrations'])
                if field == 'local_done_timestamp':
                    row['clock_uncertainty_seconds'] = error
            row['variant'] = variant
            row['block_elapsed_ms'] = 1000 * (row['local_done_timestamp'] - row['local_start_timestamp'])
            events.append(row)
        steady = [float(s['compute_ms']) for s in step if int(s['epoch']) > 1]
        metadata = json.loads((folder / 'metadata.json').read_text())
        record = dict(rank=rank, host=metadata['hostname'], batch=batch,
            tensorflow_version=metadata['tensorflow_version'], cpu_affinity_at_metadata=metadata.get('cpu_affinity'),
            samples_per_epoch=batch * 75, processed_samples=batch * 3000,
            local_compute_optimizer_seconds=sum(float(s['compute_ms']) for s in step) / 1000,
            step_compute_steady_ms=stats(steady),
            steady_samples_per_second=batch * 1000 / statistics.median(steady),
            steady_ms_per_sample=statistics.median(steady) / batch,
            gather_seconds=sum(float(s['gather_ms']) for s in raw) / 1000,
            outer_model_sync_seconds=sum(float(s['model_sync_ms']) for s in raw) / 1000,
            communication_interval_seconds=sum(float(s['communication_ms']) for s in comm) / 1000,
            clock_max_uncertainty_ms=1000 * max(m['uncertainty_seconds'] for m in clock['calibrations']),
            calibration_seconds=clock['calibration_elapsed_seconds'],
            telemetry_flush_seconds=clock['write_elapsed_seconds_before_this_flush'],
            step_totals_seconds={field: sum(float(s[field]) for s in step) / 1000 for field in
                ['data_ms', 'compute_ms', 'optimizer_ms', 'barrier_ms', 'allreduce_ms', 'other_ms', 'step_ms']},
            block_compute_optimizer_ms=stats([float(s['compute_optimizer_ms']) for s in raw]),
            block_elapsed_ms=stats([1000 * (float(s['local_done_timestamp']) - float(s['local_start_timestamp'])) for s in raw]),
            optimizer_envelope_seconds=sum(float(s['optimizer_ms']) for s in raw) / 1000,
            inner_sync_seconds=sum(float(s['total_sync_ms']) for s in comm) / 1000)
        ranks.append(record)
    round_records = []
    for round_id in range(1, 161):
        group = sorted([r for r in events if r['sync_round'] == round_id], key=lambda r: r['rank'])
        last = max(group, key=lambda r: r['aligned_local_done_timestamp'])
        latest = last['aligned_local_done_timestamp']
        first = min(r['aligned_local_done_timestamp'] for r in group)
        latest_enter = max(r['aligned_gather_enter_timestamp'] for r in group)
        last_lower = max(r['aligned_local_done_timestamp'] - r['clock_uncertainty_seconds'] for r in group)
        candidates = [r for r in group if r['aligned_local_done_timestamp'] + r['clock_uncertainty_seconds'] >= last_lower]
        for row in group:
            row['waiting_until_last_done_ms'] = 1000 * max(0, latest - row['aligned_local_done_timestamp'])
            uncertainty = last['clock_uncertainty_seconds'] + row['clock_uncertainty_seconds']
            row['waiting_lower_ms'] = max(0, row['waiting_until_last_done_ms'] - 1000 * uncertainty)
            row['waiting_upper_ms'] = row['waiting_until_last_done_ms'] + 1000 * uncertainty
            row['waiting_until_last_gather_enter_ms'] = 1000 * max(0, latest_enter - row['aligned_gather_enter_timestamp'])
        round_records.append(dict(variant=variant, epoch=int(group[0]['epoch']), sync_round=round_id,
            global_step=int(group[0]['global_step']), step_in_epoch=(int(group[0]['global_step']) - 1) % 75 + 1,
            critical_rank=int(last['rank']), critical_rank_ambiguous=len(candidates) != 1,
            readiness_spread_ms=1000 * (latest - first),
            local_start_spread_ms=1000 * (max(r['aligned_local_start_timestamp'] for r in group) - min(r['aligned_local_start_timestamp'] for r in group)),
            rank0_wait_ms=group[0]['waiting_until_last_done_ms'], rank0_gather_ms=group[0]['gather_ms'],
            rank0_wait_minus_gather_ms=group[0]['waiting_until_last_done_ms'] - group[0]['gather_ms'],
            rank0_outer_sync_ms=group[0]['model_sync_ms'],
            round_envelope_ms=1000 * (max(r['aligned_sync_exit_timestamp'] for r in group) - min(r['aligned_local_start_timestamp'] for r in group))))
    for rank, record in enumerate(ranks):
        selected = [row for row in events if row['rank'] == rank]
        record['waiting_until_last_done_ms'] = stats([row['waiting_until_last_done_ms'] for row in selected])
        record['waiting_until_last_done_seconds'] = sum(row['waiting_until_last_done_ms'] for row in selected) / 1000
        record['nominal_critical_rounds'] = sum(row['critical_rank'] == rank for row in round_records)
    folder = root / 'ranks' / variant / 'rank0'
    epochs = rows(folder / 'train.csv')
    assert len(epochs) == 40 and all(float(e['learning_rate']) == 0.001 for e in epochs)
    assert sum(r['processed_samples'] for r in ranks) == 1800000
    timeline = rows(folder / 'local_sgd_timeline.csv')
    assert [int(r['step']) for r in timeline] == [20, 40, 60, 75] * 40
    assert [int(r['is_forced']) for r in timeline] == [0, 0, 0, 1] * 40
    assert sum(int(r['total_samples_since_sync']) for r in timeline) == 1800000
    s = ranks[0]['step_totals_seconds']
    train = sum(float(r['train_time']) for r in epochs)
    validation = sum(float(r['val_time']) for r in epochs)
    comm = ranks[0]['communication_interval_seconds']
    inner = ranks[0]['inner_sync_seconds']
    components = dict(local_compute_optimizer=s['compute_ms'], data_fetch=s['data_ms'],
        sample_gather_elapsed=s['barrier_ms'], communication_interval=comm,
        sync_outside_communication_interval=inner - comm,
        sync_csv_log_wrapper=s['allreduce_ms'] - inner, step_other=s['other_ms'],
        unclassified_epoch_time_outside_steps=train - s['step_ms'], validation=validation,
        unclassified_loop_time_outside_epoch_train_val=duration - train - validation)
    assert abs(sum(components.values()) - duration) < 1e-6
    elapsed = rows(folder / 'epoch_elapsed_rank0.csv')
    first_target = next((r for r in elapsed if float(r['val_accuracy']) >= .65), None)
    result = dict(variant=variant, total_seconds=duration,
        test_accuracy_percent=float(re.findall(r'Final Test Evaluation -> Loss: [\d.]+, Accuracy: ([\d.]+)%', text)[0]),
        components_seconds=components, ranks=ranks,
        rank0_wait_ms=stats([r['rank0_wait_ms'] for r in round_records]),
        wait_minus_gather_ms=stats([r['rank0_wait_minus_gather_ms'] for r in round_records]),
        ambiguous_rounds=sum(r['critical_rank_ambiguous'] for r in round_records),
        best_validation_percent=100 * max(float(e['val_accuracy']) for e in epochs),
        first_epoch_validation65=int(first_target['epoch']), time_to_validation65_seconds=float(first_target['elapsed_training_seconds']),
        source_log=str(root / (variant + '.log')))
    return result, events, round_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path, default=Path('validation/batch600_runtime_audit_20261019'))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = resolve_result_path(args.root, repo) if args.root else find_result(repo, RUN)
    output = args.output if args.output.is_absolute() else repo / args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'source_manifest.json').read_text())
    checked = {}
    for filename in ['train_mpi.py', 'src/training/trainer_mpi.py', 'src/data/cifar10.py',
                     'src/data/augmentation.py', 'src/utils/seed.py', 'src/training/base_trainer.py',
                     'src/metrics/logger.py', 'src/training/round_diagnostic.py']:
        digest = hashlib.sha256((root / 'source' / filename).read_bytes()).hexdigest()
        assert manifest[filename] == digest, filename
        checked[filename] = digest
    a, ae, ar = audit_variant(root, 'gpu_only', ALLOCATIONS['gpu_only'])
    b, be, br = audit_variant(root, 'cluster5', ALLOCATIONS['cluster5'])
    configs = [yaml.load((root / 'ranks' / v / 'rank0/config.yaml').read_text(), Loader=SavedConfigLoader)
               for v in ALLOCATIONS]
    delta_config = differences(*configs)
    assert {r['field'] for r in delta_config} == {'experiment.name', 'training.rank_batch_sizes'}
    components = [dict(component=key, gpu_seconds=a['components_seconds'][key],
        cluster_seconds=value, delta_seconds=value - a['components_seconds'][key])
        for key, value in b['components_seconds'].items()]
    assert abs(sum(row['delta_seconds'] for row in components) - (b['total_seconds'] - a['total_seconds'])) < 1e-6
    rates = [r['steady_samples_per_second'] for r in b['ranks']]
    allocation_estimate = dict(status='ESTIMATED: constant observed throughput, unvalidated linear scaling',
        current=ALLOCATIONS['cluster5'], fractional_batch600=[600 * r / sum(rates) for r in rates],
        illustrative_gpu_even_rounding=[542, 11, 9, 9, 29],
        predicted_step_seconds=[batch / rate for batch, rate in zip([542, 11, 9, 9, 29], rates)])
    projections = []
    for h in [20, 40, 80, 160]:
        count = 40 * math.ceil(75 / h)
        removed = 160 - count
        sync_saving = removed * b['ranks'][0]['step_totals_seconds']['allreduce_ms'] / 160
        optimistic = removed * (b['ranks'][0]['step_totals_seconds']['allreduce_ms'] + b['ranks'][0]['gather_seconds']) / 160
        projections.append(dict(H=h, rounds_per_epoch=math.ceil(75 / h), rounds40e=count,
            gpu_model_sync_only_saving_seconds=removed * a['ranks'][0]['step_totals_seconds']['allreduce_ms'] / 160,
            model_sync_only_saving_seconds=sync_saving, predicted_total_sync_only_seconds=b['total_seconds'] - sync_saving,
            optimistic_gather_and_sync_saving_seconds=optimistic,
            predicted_total_optimistic_seconds=b['total_seconds'] - optimistic,
            label='ESTIMATED, unchanged compute/per-round cost; not a forecast of convergence or real waiting'))
    write_csv(output / 'runtime_components.csv', components)
    write_csv(output / 'aligned_rank_rounds.csv', ae + be)
    write_csv(output / 'sync_rounds.csv', ar + br)
    write_csv(output / 'H_projections.csv', projections)
    boundary = []
    for row in br:
        group = sorted([e for e in be if e['sync_round'] == row['sync_round']], key=lambda e: e['rank'])
        first_start = min(e['aligned_local_start_timestamp'] for e in group)
        lag = group[1]['aligned_local_start_timestamp'] - first_start
        if lag > 5:
            boundary.append(dict(epoch=row['epoch'], round=row['sync_round'],
                lab02_start_lag_seconds=lag, lab02_compute_optimizer_seconds=group[1]['compute_optimizer_ms'] / 1000,
                rank0_wait_seconds=row['rank0_wait_ms'] / 1000))
    critical_wait = {str(rank): sum(r['rank0_wait_ms'] for r in br if r['critical_rank'] == rank) / 1000
                     for rank in range(5)}
    evidence = dict(run_root=str(root), source_snapshot=str(root / 'source'), verified_source_sha256=checked,
        saved_config_differences=delta_config, gpu_only=a, cluster5=b,
        runtime_delta_seconds=b['total_seconds'] - a['total_seconds'], components=components,
        lab02_delayed_block_starts=boundary, rank0_wait_seconds_by_critical_rank=critical_wait,
        allocation_estimate=allocation_estimate, H_projections=projections,
        unknown=['pure GPU kernel time', 'pure wire/network time separate from collective residence',
                 'checkpoint-only elapsed', 'cause of delayed first-block start on lab02',
                 'payload/scalar entry and exit timestamps in these historical traces'])
    (output / 'audit.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    patch = []
    for filename in ['train_mpi.py', 'src/training/trainer_mpi.py', 'src/training/round_diagnostic.py',
                     'tools/run_batch600_comparison.py', 'tools/analyze_runtime_stage_trace.py', 'tests/test_runtime_stage_trace.py']:
        historical, current = root / 'source' / filename, repo / filename
        before = historical.read_text().splitlines(keepends=True) if historical.exists() else []
        patch.extend(difflib.unified_diff(before, current.read_text().splitlines(keepends=True),
            fromfile='a/' + filename if historical.exists() else '/dev/null', tofile='b/' + filename))
    (output / 'instrumentation.patch').write_text(''.join(patch))
    print('PASS: 2 runs, 6 rank traces, 18000 steps, 960 rank-round events; complete accounting ->', output)
    for row in components:
        print('%-48s %10.3f %10.3f %+10.3f' % (row['component'], row['gpu_seconds'], row['cluster_seconds'], row['delta_seconds']))


if __name__ == '__main__':
    main()
