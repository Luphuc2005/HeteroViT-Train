"""Launch the same-input diagnostic or training seeds selected in the run plan."""
import argparse
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
METHODS = {'fp32', 'fp16_whole_allgather', 'fp16_delta_allgather', 'fp16_delta_allgather_ef'}


def positive(value, name, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(name + ' must be ' + ('nonnegative' if allow_zero else 'positive'))
    return str(value)


def commands(plan, phase):
    result = []
    launcher = ['bash', str(ROOT / 'scripts/sweep_fp16_comparison.sh')]
    if phase in ('diagnostic', 'all'):
        cfg = plan['diagnostic']
        result.append(launcher + ['--fp16-diagnostic', '--methods', 'fp32',
            '--seed', positive(cfg['seed'], 'diagnostic.seed', True),
            '--epochs', positive(cfg['epochs'], 'diagnostic.epochs'),
            '--max-steps', positive(cfg['max_steps'], 'diagnostic.max_steps')])
    if phase in ('smoke', 'full', 'all'):
        cfg = plan['full']
        methods = cfg['methods']
        if (not isinstance(methods, list) or not methods
                or any(m not in METHODS for m in methods) or len(set(methods)) != len(methods)):
            raise ValueError('full.methods must contain unique supported methods')
        if phase in ('smoke', 'all'):
            smoke = plan['smoke']
            result.append(launcher + ['--methods', ','.join(methods),
                '--seed', positive(smoke['seed'], 'smoke.seed', True),
                '--epochs', positive(smoke['epochs'], 'smoke.epochs'),
                '--max-steps', positive(smoke['max_steps'], 'smoke.max_steps'),
                '--verify-delta-state'])
        if phase == 'smoke':
            return result
        seeds = cfg['seeds']
        if not isinstance(seeds, list) or not seeds:
            raise ValueError('full.seeds must be a nonempty list')
        seed_args = [positive(seed, 'full.seeds', True) for seed in seeds]
        if len(set(seed_args)) != len(seed_args):
            raise ValueError('full.seeds must be unique')
        for seed in seed_args:
            result.append(launcher + ['--methods', ','.join(methods), '--seed', seed,
                          '--epochs', positive(cfg['epochs'], 'full.epochs')])
    return result


def validate_diagnostic_config():
    base = yaml.safe_load((ROOT / 'configs/mpi/local_sgd_h40_fp32_40e.yaml').read_text())
    diagnostic = yaml.safe_load((ROOT / 'configs/mpi/fp16_same_inputs_seed43_smoke.yaml').read_text())
    if diagnostic.pop('diagnostics', {}) != {'fp16_same_inputs': True}:
        raise ValueError('Diagnostic config must enable the same-input comparison')
    for cfg in (base, diagnostic):
        cfg.pop('experiment')
        cfg.pop('seed')
        cfg['training'].pop('epochs')
        cfg['training'].pop('verify_sync_every_round', None)
    if base != diagnostic:
        raise ValueError('Diagnostic settings must match the FP32 control')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('diagnostic', 'smoke', 'full', 'all'), default='diagnostic')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/mpi/fp16_followup.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    plan = yaml.safe_load(args.config.read_text())
    selected = commands(plan, args.phase)
    if args.phase in ('diagnostic', 'all'):
        validate_diagnostic_config()
    print('Plan: %s | phase=%s | jobs=%d' % (args.config, args.phase, len(selected)), flush=True)
    for command in selected:
        if args.dry_run:
            command.append('--dry-run')
        subprocess.run(command, check=True, cwd=str(ROOT.parent))


if __name__ == '__main__':
    main()
