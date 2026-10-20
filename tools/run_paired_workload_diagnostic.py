"""Stage, validate and run only the authorized A/B FP32 workload experiment."""
import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from paired_workload_analysis import ALLOCATIONS, analyze_run, comparison_report, read_csv
from results_layout import find_result, result_group


def execute(args, **kwargs):
    print('EXEC:', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def make_configs(repo, destination):
    base_path = repo / 'configs/mpi/local_sgd_h40_fp32_40e.yaml'
    base = yaml.safe_load(base_path.read_text())
    configs = {}
    for variant, allocation in ALLOCATIONS.items():
        config = copy.deepcopy(base)
        config['experiment']['name'] = 'paired_' + variant + '_fp32_h40_seed42'
        config['training']['rank_batch_sizes'] = allocation
        config['logging']['round_diagnostic'] = True
        t = config['training']
        assert sum(allocation) == t['global_batch_size'] == 300
        assert config['seed'] == 42 and t['epochs'] == 40 and t['local_sgd_h'] == 40
        assert t['comm_precision'] == 'fp32' and t.get('comm_collective', 'allreduce') == 'allreduce'
        assert t['sync_mode'] == 'local_sgd' and t['learning_rate'] == 0.001 and t['optimizer'] == 'adamw'
        assert not t.get('comm_error_feedback', False) and not t.get('auto_balance', False)
        assert not config.get('dynamic_scheduler', {}).get('enabled', False)
        normalized = copy.deepcopy(config)
        normalized['experiment'] = base['experiment']
        normalized['training']['rank_batch_sizes'] = base['training']['rank_batch_sizes']
        normalized['logging'].pop('round_diagnostic')
        assert normalized == base, 'Unexpected change outside allocation/naming/telemetry'
        configs[variant] = config
    a, b = [copy.deepcopy(configs[k]) for k in ALLOCATIONS]
    a['experiment'] = b['experiment']
    a['training']['rank_batch_sizes'] = b['training']['rank_batch_sizes']
    assert a == b, 'A/B differ beyond allocation and experiment name'
    check = dict(passed=True, baseline=str(base_path), seed=42, epochs=40, H=40, precision='fp32',
                 collective='unchanged baseline sample-weighted FP32 AllReduce',
                 optimizer='adamw', learning_rate=0.001, validation_target_percent=65,
                 quality_warning_margin_pp=0.5,
                 only_A_B_differences=['experiment.name (output label)', 'training.rank_batch_sizes'],
                 telemetry_identical_in_both=True, configs=configs,
                 note='No clustering, adaptive H, quantization or EF. Allocation changes dataset shards and sample weights through existing loader.')
    destination.mkdir(parents=True, exist_ok=True)
    for variant, config in configs.items():
        (destination / (variant + '.yaml')).write_text(yaml.safe_dump(config, sort_keys=False))
    (destination.parent / 'config_check.json').write_text(json.dumps(check, indent=2) + '\n')
    return configs


def stage_source(repo, root):
    source = Path(tempfile.mkdtemp(prefix='heterovit_paired_workload_'))
    for folder in ['src', 'tools', 'tests', 'configs']:
        shutil.copytree(str(repo / folder), str(source / folder),
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(str(repo / 'train_mpi.py'), str(source / 'train_mpi.py'))
    cluster = source / 'cluster'
    cluster.mkdir()
    for name in ['python_by_host.sh', 'hostfile']:
        shutil.copy2(str(repo.parent / 'mpi_cluster' / name), str(cluster / name))
    expected_runtime = []
    for rank in range(5):
        metadata = json.loads((find_result(repo, 'fp16_sweep_uydYAoMC') / 'ranks/fp32' /
                               ('rank%d' % rank) / 'metadata.json').read_text())
        expected_runtime.append(dict(rank=rank, host=metadata['hostname'],
                                     tensorflow=metadata['tensorflow_version'],
                                     python_version=metadata['python_version']))
    (cluster / 'expected_baseline_runtime.json').write_text(json.dumps(expected_runtime, indent=2) + '\n')
    for config in (root / 'configs').glob('*.yaml'):
        shutil.copy2(str(config), str(source / 'configs/mpi' / ('paired_' + config.name)))
    (root / 'source_stage.txt').write_text(str(source) + '\n')
    files = {}
    for path in sorted(source.rglob('*')):
        if path.is_file():
            files[path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / 'source_manifest.json').write_text(json.dumps(files, indent=2) + '\n')
    patch = subprocess.check_output(['git', 'diff', '--', 'src/training/trainer_mpi.py'], cwd=str(repo))
    (root / 'telemetry.patch').write_bytes(patch)
    # Keep a durable local source snapshot rather than relying only on /tmp.
    shutil.copytree(str(source), str(root / 'source'))
    return source


def collect(root, output, name, variant, workers):
    candidates = list(output.glob(name + '_*'))
    if len(candidates) != 1:
        raise ValueError('Expected exactly one rank0 output for ' + name)
    for rank in range(5):
        destination = root / 'ranks' / variant / ('rank%d' % rank)
        destination.mkdir(parents=True, exist_ok=False)
        includes = ['--include=*/', '--include=*.csv', '--include=*.json',
                    '--include=config.yaml', '--include=run.log', '--exclude=*']
        if rank == 0:
            execute(['rsync', '-a'] + includes + [str(candidates[0]) + '/', str(destination) + '/'])
        else:
            execute(['rsync', '-az', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10'] + includes +
                    ['icip@' + workers[rank - 1] + ':' + str(output / (name + '_*')) + '/', str(destination) + '/'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'smoke', 'all'], default='prepare')
    parser.add_argument('--full-run', choices=['both', 'A_control', 'B_mild_rebalance'], default='both',
                        help='Select full training runs; preflight and all smoke checks remain enabled.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.output.resolve() if args.output else Path(tempfile.mkdtemp(
        prefix='paired_workload_seed42_' + datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S') + '_',
        dir=str(result_group(repo, '03_WORKLOAD_BALANCE_BATCH300'))))
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'status.txt').exists():
        raise ValueError('Refusing to overwrite an existing experiment: ' + str(root))
    make_configs(repo, root / 'configs')
    (root / 'status.txt').write_text('prepared\n')
    (root / 'run_selection.json').write_text(json.dumps(dict(full_run=args.full_run, phase=args.phase), indent=2) + '\n')
    print('PAIR_ROOT:', root, flush=True)
    if args.phase == 'prepare':
        return
    source = stage_source(repo, root)
    workers = [line.strip() for line in (repo.parent / 'mpi_cluster/slaves.txt').read_text().splitlines() if line.strip()]
    assert len(workers) == 4
    env = os.environ.copy()
    env['PYTHONNOUSERSITE'] = '1'
    env['OUTPUT_DIR'] = str(root / 'results')
    Path(env['OUTPUT_DIR']).mkdir()
    try:
        for worker in workers:
            execute(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'icip@' + worker,
                     'mkdir -p ' + str(source)])
            execute(['rsync', '-az', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10',
                     str(source) + '/', 'icip@' + worker + ':' + str(source) + '/'])
        mpi = ['mpiexec', '-x', 'PYTHONNOUSERSITE=1', '-x', 'OUTPUT_DIR',
               '--mca', 'btl_tcp_if_include', '192.168.1.0/24', '--mca', 'oob_tcp_if_include', '192.168.1.0/24',
               '-np', '5', '--hostfile', str(source / 'cluster/hostfile'), '--map-by', 'ppr:1:node',
               '-wdir', str(source), str(source / 'cluster/python_by_host.sh')]
        with (root / 'runtime.json').open('w') as stdout, (root / 'runtime.stderr').open('w') as stderr:
            execute(mpi + ['tools/preflight_workload_diagnostic.py'], env=env, stdout=stdout, stderr=stderr)

        def run(config, name, result_root, variant, max_steps=None):
            (root / 'status.txt').write_text('running ' + name + '\n')
            command = mpi + ['train_mpi.py', '--config', str(config), '--name', name, '--epochs', '40', '--seed', '42']
            if max_steps:
                command += ['--max-steps', str(max_steps)]
            log_path = result_root / (variant + '.log')
            with log_path.open('w') as handle:
                execute(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
            collect(result_root, Path(env['OUTPUT_DIR']), name, variant, workers)

        smoke = root / 'smoke'
        smoke.mkdir()
        # Same-allocation no-timer control to check telemetry overhead.
        baseline = yaml.safe_load((source / 'configs/mpi/paired_A_control.yaml').read_text())
        baseline['logging']['round_diagnostic'] = False
        baseline_path = source / 'configs/mpi/paired_A_telemetry_disabled.yaml'
        baseline_path.write_text(yaml.safe_dump(baseline, sort_keys=False))
        for worker in workers:
            execute(['rsync', '-az', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10',
                     str(baseline_path), 'icip@' + worker + ':' + str(baseline_path)])
        run(baseline_path, 'smoke_A_no_telemetry', smoke, 'A_no_telemetry', 120)
        smoke_metrics = {}
        for variant in ALLOCATIONS:
            run(source / ('configs/mpi/paired_' + variant + '.yaml'), 'smoke_' + variant, smoke, variant, 120)
            smoke_metrics[variant] = analyze_run(smoke, variant, expected_rounds=3, expected_epochs=1)
        def steady_compute(variant, rank=0):
            filename = 'step_timing_breakdown.csv' if rank == 0 else 'step_timing_breakdown_rank%d.csv' % rank
            trace = read_csv(smoke / 'ranks' / variant / ('rank%d' % rank) / filename)
            values = [float(row['compute_ms']) for row in trace if int(row['step']) > 10 and int(row['step']) % 40]
            import statistics
            return statistics.median(values)
        old, new = steady_compute('A_no_telemetry'), steady_compute('A_control')
        overhead = 100 * (new / old - 1)
        overhead_by_rank = {}
        for rank in range(5):
            baseline_ms, timed_ms = steady_compute('A_no_telemetry', rank), steady_compute('A_control', rank)
            overhead_by_rank[str(rank)] = dict(baseline_median_ms=baseline_ms, telemetry_median_ms=timed_ms,
                                               overhead_percent=100 * (timed_ms / baseline_ms - 1))
        smoke_result = dict(passed=all(r['overhead_percent'] <= 5 for r in overhead_by_rank.values()), baseline_rank0_median_compute_ms=old,
                            telemetry_rank0_median_compute_ms=new, observed_overhead_percent=overhead,
                            overhead_by_rank=overhead_by_rank,
                            gate='Stop before full if observed median steady compute overhead >5% on any rank; single sequential smoke is a noisy estimate',
                            variants=smoke_metrics)
        (root / 'smoke_check.json').write_text(json.dumps(smoke_result, indent=2) + '\n')
        if not smoke_result['passed']:
            raise ValueError('Telemetry overhead gate failed; inspect smoke_check.json before full training')
        if args.phase == 'smoke':
            (root / 'status.txt').write_text('smoke_passed\n')
            return
        # Recheck frozen configs immediately before the two full runs.
        make_configs(repo, root / 'configs_final_check')
        for variant in ALLOCATIONS:
            expected = yaml.safe_load((root / 'configs' / (variant + '.yaml')).read_text())
            frozen = yaml.safe_load((source / ('configs/mpi/paired_' + variant + '.yaml')).read_text())
            assert frozen == expected
        selected = list(ALLOCATIONS) if args.full_run == 'both' else [args.full_run]
        for variant in selected:  # Default: A then B. Optional: one standalone repeat.
            name = 'paired_' + variant + '_fp32_h40_seed42_40e'
            run(source / ('configs/mpi/paired_' + variant + '.yaml'), name, root, variant)
            analyze_run(root, variant)
        if args.full_run == 'both':
            a = json.loads((root / 'A_control_analysis.json').read_text())
            b = json.loads((root / 'B_mild_rebalance_analysis.json').read_text())
            comparison_report(root, a, b)
        (root / 'status.txt').write_text('completed\n')
        if args.full_run == 'both':
            print('REPORT:', root / 'paired_diagnostic_report_vi.md', flush=True)
        else:
            print('STANDALONE_ANALYSIS:', root / (args.full_run + '_analysis.json'), flush=True)
    except BaseException as error:
        (root / 'status.txt').write_text('failed: ' + repr(error) + '\n')
        raise


if __name__ == '__main__':
    main()
