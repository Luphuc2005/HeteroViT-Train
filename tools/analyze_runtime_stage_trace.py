"""Align buffered sync/epoch stages; separate late-entry residence without calling it wire time."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from paired_workload_analysis import align, stats
from results_layout import resolve_result_path


def split_collective_residence(enter, exit, latest_enter):
    """An interval partition, not an estimate of physical network service time."""
    if exit < enter:
        raise ValueError('Collective exit precedes entry')
    before = max(0.0, min(exit, latest_enter) - enter)
    after = exit - enter - before
    return before * 1000, after * 1000


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def analyze(root):
    root = resolve_result_path(root)
    output = root / 'runtime_stage_analysis'
    output.mkdir(exist_ok=True)
    totals = {}
    for variant in sorted((root / 'ranks').iterdir()):
        events, phases = [], []
        for folder in sorted(variant.glob('rank*')):
            rank = int(folder.name[4:])
            clock = json.loads((folder / ('clock_calibration_rank%d.json' % rank)).read_text())
            if not clock.get('runtime_stage_trace'):
                raise ValueError('Detailed instrumentation absent: ' + str(folder))
            for raw in read_csv(folder / ('sync_trace_rank%d.csv' % rank)):
                row = {key: value if key == 'host' else float(value) for key, value in raw.items()}
                for key in ['block_start', 'compute_end', 'gather_enter', 'gather_exit', 'sync_enter',
                            'collective_enter', 'collective_exit', 'sync_end']:
                    row['aligned_' + key], error = align(row[key], clock['calibrations'])
                    if key == 'collective_enter':
                        row['entry_clock_uncertainty_ms'] = error * 1000
                events.append(row)
            for raw in read_csv(folder / ('runtime_stages_rank%d.csv' % rank)):
                raw['elapsed_ms'] = float(raw['elapsed_ms'])
                phases.append(raw)
        groups = defaultdict(list)
        for event in events:
            groups[int(event['sync_round'])].append(event)
        expected_ranks = {int(e['rank']) for e in events}
        for group in groups.values():
            assert {int(e['rank']) for e in group} == expected_ranks
            assert len({e['global_step'] for e in group}) == len({e['epoch'] for e in group}) == 1
            last_done = max(e['aligned_compute_end'] for e in group)
            last_entry = max(e['aligned_collective_enter'] for e in group)
            for event in group:
                event['pre_collective_wait_ms'] = max(0, last_done - event['aligned_compute_end']) * 1000
                before, after = split_collective_residence(event['aligned_collective_enter'],
                    event['aligned_collective_exit'], last_entry)
                event['payload_residence_before_last_entry_ms'] = before
                event['payload_residence_after_last_entry_ms'] = after
        phase_totals = defaultdict(float)
        for phase in phases:
            phase_totals[(int(phase['rank']), phase['stage'])] += phase['elapsed_ms'] / 1000
        stage_table = [dict(rank=rank, stage=stage, total_seconds=seconds)
                       for (rank, stage), seconds in sorted(phase_totals.items())]
        rank_summaries = []
        for rank in sorted(expected_ranks):
            selected = [e for e in events if int(e['rank']) == rank]
            rank_summaries.append(dict(rank=rank, rounds=len(selected),
                readiness_wait_ms=stats([e['pre_collective_wait_ms'] for e in selected]),
                readiness_wait_seconds=sum(e['pre_collective_wait_ms'] for e in selected) / 1000,
                payload_pre_last_entry_seconds=sum(e['payload_residence_before_last_entry_ms'] for e in selected) / 1000,
                payload_post_last_entry_seconds=sum(e['payload_residence_after_last_entry_ms'] for e in selected) / 1000))
        write_csv(output / (variant.name + '_sync_trace.csv'), events)
        write_csv(output / (variant.name + '_epoch_stages.csv'), stage_table)
        totals[variant.name] = dict(ranks=rank_summaries, epoch_stages=stage_table)
    result = dict(variants=totals, semantics={
        'pre_collective_wait_ms': 'DERIVED max aligned compute_end minus own compute_end; logical readiness gap, may overlap worker MPI residence; never add to gather.',
        'payload_residence_before_last_entry': 'DERIVED part of recorded blocking payload call before last rank enters; includes MPI progress/overlap, not necessarily idle CPU.',
        'payload_residence_after_last_entry': 'DERIVED remaining elapsed payload call; not pure wire time, may include progress, latency and return scheduling.',
        'epoch_stages': 'MEASURED local host durations; ranks overlap, do not sum across ranks.',
        'clock_uncertainty': 'Min-RTT calibration plus interpolation; inspect raw metadata before interpreting sub-millisecond differences.'})
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print('RUNTIME TRACE ANALYSIS:', output / 'summary.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
