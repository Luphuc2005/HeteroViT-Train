"""Run seed43 then seed44 on lab01's two GPUs, with the seed42 recipe."""
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


BASE = 'local_sgd_h40_fp32_2gpu_only_300_40e'
SEEDS = (43, 44)


def validate_configs(repo):
    baseline = yaml.safe_load((repo / 'configs/mpi' / (BASE + '.yaml')).read_text())
    configs = {}
    for seed in SEEDS:
        config = yaml.safe_load((repo / 'configs/mpi' / (BASE + '_seed%d.yaml' % seed)).read_text())
        assert config['seed'] == seed
        assert config['experiment']['name'] == BASE + '_seed%d' % seed
        normalized = copy.deepcopy(config)
        normalized['seed'] = baseline['seed']
        normalized['experiment'] = baseline['experiment']
        assert normalized == baseline, 'Recipe differs from seed42 beyond seed/name'
        t = config['training']
        assert t['global_batch_size'] == sum(t['rank_batch_sizes']) == 300
        assert t['rank_batch_sizes'] == [300]
        assert t['epochs'] == 40 and t['local_sgd_h'] == 40
        assert t['sync_mode'] == 'local_sgd' and t['comm_precision'] == 'fp32'
        assert t['learning_rate'] == 0.001 and t['optimizer'] == 'adamw'
        assert config['gpu']['num_gpus'] == 2 and config['logging']['round_diagnostic']
        configs[seed] = config
    return configs


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def summarize(run_dir, seed, launcher_text):
    epochs = read_csv(run_dir / 'train.csv')
    elapsed = read_csv(run_dir / 'epoch_elapsed_rank0.csv')
    rounds = read_csv(run_dir / 'round_diagnostic_rank0.csv')
    steps = read_csv(run_dir / 'step_timing_breakdown.csv')
    metadata = json.loads((run_dir / 'metadata.json').read_text())
    assert len(epochs) == 40 and [int(r['epoch']) for r in epochs] == list(range(1, 41))
    assert len(steps) == 6000 and len(rounds) == 160
    assert metadata['seed'] == seed and metadata['gpu_info']['gpu_count'] == 2
    samples = sum(float(r['samples_processed']) for r in rounds)
    assert samples == 1800000
    times = re.findall(r'Total Training Time\s*:\s*([\d.]+)s', launcher_text)
    tests = re.findall(r'Final Test Evaluation -> Loss:\s*([\d.]+), Accuracy:\s*([\d.]+)%', launcher_text)
    assert len(times) == len(tests) == 1, 'Missing or repeated final evaluation'
    first = next((r for r in elapsed if float(r['val_accuracy']) >= 0.65), None)
    best = max(epochs, key=lambda r: float(r['val_accuracy']))
    return dict(seed=seed, training_seconds=float(times[0]),
                training_minutes=float(times[0]) / 60,
                best_validation_percent=100 * float(best['val_accuracy']),
                best_validation_epoch=int(best['epoch']),
                final_validation_percent=100 * float(epochs[-1]['val_accuracy']),
                test_loss=float(tests[0][0]), test_accuracy_percent=float(tests[0][1]),
                first_epoch_validation_65=int(first['epoch']) if first else None,
                time_to_validation_65_seconds=float(first['elapsed_training_seconds']) if first else None,
                processed_samples=int(samples), optimizer_steps=len(steps), sync_rounds=len(rounds),
                run_dir=str(run_dir))


def save_summary(root, rows):
    (root / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    with (root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Validate configs without training or creating results.')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    configs = validate_configs(repo)
    print('CONFIG PASS: seeds43/44; one MPI rank; two GPUs; batch300 (150/GPU); 40e; H40; FP32; LR0.001.', flush=True)
    if args.check_only:
        return
    if socket.gethostname().split('.')[0] != 'iciplab01':
        raise RuntimeError('Run this sweep on lab01 only')
    # Same lock as the paired and GPU-only commands, held throughout both runs.
    with open('/tmp/heterovit_paired_workload_cluster.lock', 'a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another experiment holds the cluster lock; no training started')
        stamp = datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S')
        root = Path(tempfile.mkdtemp(prefix='gpu_only_seed43_44_' + stamp + '_', dir=str(repo / 'results')))
        print('SWEEP_ROOT:', root, flush=True)
        status = root / 'status.txt'
        status.write_text('prepared\n')
        try:
            source = root / 'source'
            source.mkdir()
            shutil.copytree(str(repo / 'src'), str(source / 'src'),
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            shutil.copy2(str(repo / 'train_mpi.py'), str(source / 'train_mpi.py'))
            cluster = source / 'cluster'
            cluster.mkdir()
            shutil.copy2(str(repo.parent / 'mpi_cluster/python_by_host.sh'), str(cluster / 'python_by_host.sh'))
            config_dir = source / 'configs'
            config_dir.mkdir()
            for seed, config in configs.items():
                (config_dir / ('seed%d.yaml' % seed)).write_text(yaml.safe_dump(config, sort_keys=False))
            manifest = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(source.rglob('*')) if p.is_file()}
            (root / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            (root / 'config_check.json').write_text(json.dumps(dict(passed=True, seeds=list(SEEDS),
                mpi_ranks=1, gpus=2, global_batch=300, epochs=40,
                only_differences_from_seed42=['seed', 'experiment.name'], configs=configs), indent=2) + '\n')
            env = os.environ.copy()
            env.update(PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='0,1', OUTPUT_DIR=str(root / 'runs'))
            (root / 'runs').mkdir()
            rows = []
            for seed in SEEDS:
                status.write_text('running seed%d\n' % seed)
                command = ['mpiexec', '-x', 'PYTHONNOUSERSITE=1', '-x', 'CUDA_VISIBLE_DEVICES', '-x', 'OUTPUT_DIR',
                           '-np', '1', '--host', 'localhost', str(cluster / 'python_by_host.sh'),
                           str(source / 'train_mpi.py'), '--config', str(config_dir / ('seed%d.yaml' % seed))]
                log = root / ('seed%d.log' % seed)
                print('START seed%d; LOG: %s' % (seed, log), flush=True)
                print('EXEC:', ' '.join(command), flush=True)
                with log.open('w') as handle:
                    with subprocess.Popen(command, cwd=str(source), env=env, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1) as proc:
                        for line in proc.stdout:
                            handle.write(line)
                            handle.flush()
                            print(line, end='', flush=True)
                        code = proc.wait()
                        if code:
                            raise subprocess.CalledProcessError(code, command)
                matches = list((root / 'runs').glob(configs[seed]['experiment']['name'] + '_*'))
                assert len(matches) == 1, 'Expected one fresh run directory'
                rows.append(summarize(matches[0], seed, log.read_text(errors='replace')))
                save_summary(root, rows)
                print('COMPLETED seed%d: %s' % (seed, json.dumps(rows[-1])), flush=True)
            status.write_text('completed\n')
            print('SWEEP COMPLETED:', root / 'summary.csv', flush=True)
        except BaseException as error:
            status.write_text('failed: ' + repr(error) + '\n')
            raise


if __name__ == '__main__':
    main()
