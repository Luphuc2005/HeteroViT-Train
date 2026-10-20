"""Compare GPU-only and five nodes at batch600/H20, defaulting to a 3-epoch pilot."""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone

import yaml
from paired_workload_analysis import align, stats
from results_layout import find_result, resolve_result_path, result_group


ALLOCATIONS = {'gpu_only': [600], 'cluster5': [532, 12, 12, 12, 32]}


def config_name(variant):
    return 'local_sgd_h20_fp32_%s_600_40e.yaml' % variant


def validate_configs(repo):
    base = yaml.safe_load((repo / 'configs/mpi/local_sgd_h40_fp32_2gpu_only_300_40e.yaml').read_text())
    configs = {}
    for variant, allocation in ALLOCATIONS.items():
        config = yaml.safe_load((repo / 'configs/mpi' / config_name(variant)).read_text())
        t = config['training']
        assert t['rank_batch_sizes'] == allocation and sum(allocation) == t['global_batch_size'] == 600
        assert t['local_sgd_h'] == 20 and t['epochs'] == 40 and config['seed'] == 42
        normalized = copy.deepcopy(config)
        normalized['experiment'] = base['experiment']
        for key in ['global_batch_size', 'rank_batch_sizes', 'local_sgd_h']:
            normalized['training'][key] = base['training'][key]
        assert normalized == base, 'Unexpected recipe change: ' + variant
        configs[variant] = config
    print('CONFIG PASS: batch600; H20; seed42; LR0.001; 75 steps/epoch; windows20/20/20/15.', flush=True)
    print('ALLOCATIONS:', json.dumps(ALLOCATIONS), flush=True)
    return configs


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def summarize(root, variant, epochs, run_dir, allocation, log_text):
    root, run_dir = resolve_result_path(root), resolve_result_path(run_dir)
    budget_steps = 75 * epochs
    train = read_csv(run_dir / 'train.csv')
    assert len(train) == epochs and [int(r['epoch']) for r in train] == list(range(1, epochs + 1))
    groups = [[] for _ in range(4 * epochs)]
    rank_stats = []
    for rank, batch in enumerate(allocation):
        folder = root / 'ranks' / variant / ('rank%d' % rank)
        rows = read_csv(folder / ('round_diagnostic_rank%d.csv' % rank))
        steps = read_csv(folder / ('step_timing_breakdown.csv' if rank == 0 else 'step_timing_breakdown_rank%d.csv' % rank))
        assert len(rows) == 4 * epochs and len(steps) == budget_steps
        clock = json.loads((folder / ('clock_calibration_rank%d.json' % rank)).read_text())
        assert len(clock['calibrations']) == epochs + 1
        for index, row in enumerate(rows):
            epoch, position = divmod(index, 4)
            expected_step = epoch * 75 + [20, 40, 60, 75][position]
            assert int(row['sync_round']) == index + 1 and int(row['rank']) == rank
            assert int(row['global_step']) == expected_step and int(row['epoch']) == epoch + 1
            assert int(row['local_batch']) == batch
            assert float(row['samples_processed']) == batch * [20, 20, 20, 15][position]
            done, _ = align(float(row['local_done_timestamp']), clock['calibrations'])
            groups[index].append((rank, done))
        rank_stats.append(dict(rank=rank, batch=batch,
            processed_samples=sum(float(r['samples_processed']) for r in rows),
            compute_optimizer_seconds=sum(float(r['compute_optimizer_ms']) for r in rows) / 1000,
            data_seconds=sum(float(r['data_ms']) for r in rows) / 1000,
            gather_seconds=sum(float(r['gather_ms']) for r in rows) / 1000,
            model_sync_seconds=sum(float(r['model_sync_ms']) for r in rows) / 1000))
    assert sum(r['processed_samples'] for r in rank_stats) == 45000 * epochs
    spread = [(max(t for _, t in group) - min(t for _, t in group)) * 1000 for group in groups]
    critical = {str(rank): sum(max(group, key=lambda item: item[1])[0] == rank for group in groups)
                for rank in range(len(allocation))}
    durations = re.findall(r'MPI Local SGD Training finished in ([\d.]+)s', log_text)
    assert len(durations) == 1
    metadata = json.loads((run_dir / 'metadata.json').read_text())
    assert metadata['gpu_info']['gpu_count'] == 2 and metadata['seed'] == 42
    if epochs == 40:
        precise = re.findall(r'Total Training Time\s*:\s*([\d.]+)s', log_text)
        assert len(precise) == 1
        duration = float(precise[0])
    else:
        duration = float(durations[0])
    tests = re.findall(r'Final Test Evaluation -> Loss:\s*([\d.]+), Accuracy:\s*([\d.]+)%', log_text)
    assert len(tests) == (1 if epochs == 40 else 0)
    elapsed = read_csv(run_dir / 'epoch_elapsed_rank0.csv')
    first = next((r for r in elapsed if float(r['val_accuracy']) >= 0.65), None)
    best = max(train, key=lambda r: float(r['val_accuracy']))
    return dict(variant=variant, epochs=epochs, batch=600, H=20, seed=42,
        training_seconds=duration, loop_samples_per_second=45000 * epochs / duration,
        optimizer_steps_per_rank=budget_steps, sync_rounds_per_rank=4 * epochs,
        processed_samples=45000 * epochs, best_validation_percent=100 * float(best['val_accuracy']),
        final_validation_percent=100 * float(train[-1]['val_accuracy']),
        test_accuracy_percent=float(tests[0][1]) if tests else None,
        first_epoch_validation_65=int(first['epoch']) if first else None,
        time_to_validation_65_seconds=float(first['elapsed_training_seconds']) if first else None,
        readiness_spread_ms=stats(spread), readiness_spread_excluding_epoch1_ms=stats(spread[4:]),
        nominal_critical_rank_counts=critical, per_rank=rank_stats, run_dir=str(run_dir),
        note='Single sequential pilot/full pair. Nominal rank order; inspect raw clocks for uncertainty. Pilot does not evaluate test.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, choices=[3, 5, 10, 40], default=3)
    parser.add_argument('--variant', choices=['both', 'gpu_only', 'cluster5'], default='both')
    parser.add_argument('--runtime-stage-trace', action='store_true', help='Buffer per-call MPI timestamps and epoch phase timings; no new collectives.')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    configs = validate_configs(repo)
    if args.runtime_stage_trace:
        for config in configs.values():
            config['logging']['runtime_stage_trace'] = True
    selected = list(ALLOCATIONS) if args.variant == 'both' else [args.variant]
    print('SELECTED:', selected, 'epochs:', args.epochs, 'runtime_stage_trace:', args.runtime_stage_trace, flush=True)
    if args.check_only:
        return
    if socket.gethostname().split('.')[0] != 'iciplab01':
        raise RuntimeError('Run on lab01')
    with open('/tmp/heterovit_paired_workload_cluster.lock', 'a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another experiment holds the cluster lock; no training started')
        stamp = datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S')
        root = Path(tempfile.mkdtemp(prefix='batch600_%de_seed42_%s_' % (args.epochs, stamp), dir=str(result_group(repo, '04_BATCH600_H20'))))
        status = root / 'status.txt'
        status.write_text('preparing\n')
        print('COMPARISON_ROOT:', root, flush=True)
        try:
            source = Path(tempfile.mkdtemp(prefix='heterovit_batch600_'))
            for folder in ['src', 'tools', 'configs']:
                shutil.copytree(str(repo / folder), str(source / folder),
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            shutil.copy2(str(repo / 'train_mpi.py'), str(source / 'train_mpi.py'))
            cluster = source / 'cluster'
            cluster.mkdir()
            for name in ['python_by_host.sh', 'hostfile']:
                shutil.copy2(str(repo.parent / 'mpi_cluster' / name), str(cluster / name))
            for variant, config in configs.items():
                (source / 'configs/mpi' / config_name(variant)).write_text(yaml.safe_dump(config, sort_keys=False))
                assert yaml.safe_load((source / 'configs/mpi' / config_name(variant)).read_text()) == config
            expected = []
            for rank in range(5):
                m = json.loads((find_result(repo, 'fp16_sweep_uydYAoMC') / 'ranks/fp32' / ('rank%d' % rank) / 'metadata.json').read_text())
                expected.append(dict(rank=rank, host=m['hostname'], tensorflow=m['tensorflow_version'], python_version=m['python_version']))
            (cluster / 'expected_baseline_runtime.json').write_text(json.dumps(expected, indent=2) + '\n')
            (root / 'source_stage.txt').write_text(str(source) + '\n')
            shutil.copytree(str(source), str(root / 'source'))
            manifest = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(source.rglob('*')) if p.is_file()}
            (root / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            (root / 'config_check.json').write_text(json.dumps(dict(passed=True, epochs=args.epochs,
                pilot=args.epochs != 40, variants=selected, runtime_stage_trace=args.runtime_stage_trace,
                full_window_samples=12000, configs=configs), indent=2) + '\n')
            workers = (repo.parent / 'mpi_cluster/slaves.txt').read_text().split()
            assert len(workers) == 4
            ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
            def execute(command, **kwargs):
                print('EXEC:', ' '.join(map(str, command)), flush=True)
                return subprocess.run(list(map(str, command)), check=True, **kwargs)
            for worker in workers:
                execute(ssh + ['icip@' + worker, 'mkdir -p ' + str(source)])
                execute(['rsync', '-az', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10', str(source) + '/', 'icip@' + worker + ':' + str(source) + '/'])
            env = os.environ.copy()
            env.update(PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='0,1', OUTPUT_DIR=str(root / 'runs'))
            (root / 'runs').mkdir()
            def mpi(n):
                hosts = ['--host', 'localhost'] if n == 1 else ['--hostfile', str(cluster / 'hostfile'), '--map-by', 'ppr:1:node']
                return ['mpiexec', '-x', 'PYTHONNOUSERSITE=1', '-x', 'CUDA_VISIBLE_DEVICES', '-x', 'OUTPUT_DIR',
                        '--mca', 'btl_tcp_if_include', '192.168.1.0/24', '--mca', 'oob_tcp_if_include', '192.168.1.0/24',
                        '-np', str(n)] + hosts + ['-wdir', str(source), str(cluster / 'python_by_host.sh')]
            status.write_text('runtime_preflight\n')
            with (root / 'runtime.json').open('w') as out, (root / 'runtime.stderr').open('w') as err:
                execute(mpi(5) + ['tools/preflight_workload_diagnostic.py', '--config', str(source / 'configs/mpi' / config_name('cluster5'))], env=env, stdout=out, stderr=err)
            summaries = []
            for variant in selected:
                allocation = ALLOCATIONS[variant]
                status.write_text('running ' + variant + '\n')
                name = 'batch600_h20_%s_seed42_%de' % (variant, args.epochs)
                command = mpi(len(allocation)) + ['train_mpi.py', '--config', str(source / 'configs/mpi' / config_name(variant)),
                           '--name', name, '--epochs', str(args.epochs)]
                if args.epochs != 40:
                    command += ['--max-steps', str(75 * args.epochs)]
                log = root / (variant + '.log')
                print('START:', variant, 'LOG:', log, flush=True)
                with log.open('w') as handle:
                    with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          universal_newlines=True, bufsize=1) as proc:
                        for line in proc.stdout:
                            handle.write(line)
                            handle.flush()
                            print(line, end='', flush=True)
                        code = proc.wait()
                        if code:
                            raise subprocess.CalledProcessError(code, command)
                matches = list((root / 'runs').glob(name + '_*'))
                assert len(matches) == 1
                for rank in range(len(allocation)):
                    dest = root / 'ranks' / variant / ('rank%d' % rank)
                    dest.mkdir(parents=True)
                    includes = ['--include=*/', '--include=*.csv', '--include=*.json', '--include=config.yaml', '--include=run.log', '--exclude=*']
                    origin = str(matches[0]) + '/' if rank == 0 else 'icip@' + workers[rank - 1] + ':' + str(root / 'runs' / (name + '_*')) + '/'
                    execute(['rsync', '-az', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10'] + includes + [origin, str(dest) + '/'])
                result = summarize(root, variant, args.epochs, matches[0], allocation, log.read_text(errors='replace'))
                summaries.append(result)
                (root / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
                fields = ['variant', 'epochs', 'batch', 'H', 'training_seconds', 'loop_samples_per_second',
                          'best_validation_percent', 'test_accuracy_percent', 'processed_samples', 'run_dir']
                with (root / 'summary.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(summaries)
                print('RESULT:', json.dumps(result), flush=True)
            status.write_text('completed\n')
            print('COMPLETED:', root / 'summary.csv', flush=True)
            if args.runtime_stage_trace:
                execute([str(cluster / 'python_by_host.sh'), str(source / 'tools/analyze_runtime_stage_trace.py'), str(root)])
            if args.epochs != 40:
                print('Pilot does not automatically start full training or establish final accuracy.', flush=True)
        except BaseException as error:
            status.write_text('failed: ' + repr(error) + '\n')
            raise


if __name__ == '__main__':
    main()
