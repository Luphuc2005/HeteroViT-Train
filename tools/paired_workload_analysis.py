"""Align round clocks and analyze the fixed A/B workload diagnostic."""
import argparse
import bisect
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from results_layout import resolve_result_path


ALLOCATIONS = {'A_control': [266, 6, 6, 6, 16], 'B_mild_rebalance': [266, 6, 5, 5, 18]}
TIMES = ['local_start_timestamp', 'local_done_timestamp', 'gather_enter_timestamp',
         'gather_exit_timestamp', 'sync_enter_timestamp', 'sync_exit_timestamp']


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def stats(values):
    values = sorted(values)
    if not values:
        return None
    point = (len(values) - 1) * 0.95
    left = int(point)
    p95 = values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (point - left)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=p95, minimum=values[0], maximum=values[-1])


def align(timestamp, markers):
    """Piecewise affine correction; return aligned seconds and offset uncertainty."""
    xs = [p['local_timestamp'] for p in markers]
    right = bisect.bisect_right(xs, timestamp)
    if right == 0 or right == len(xs):
        # Startup/end edge of a calibration exchange may straddle the event.
        point = markers[0 if right == 0 else -1]
        return timestamp + point['root_minus_local_seconds'], point['uncertainty_seconds']
    a, b = markers[right - 1:right + 1]
    weight = (timestamp - a['local_timestamp']) / (b['local_timestamp'] - a['local_timestamp'])
    correction = (1 - weight) * a['root_minus_local_seconds'] + weight * b['root_minus_local_seconds']
    uncertainty = (1 - weight) * a['uncertainty_seconds'] + weight * b['uncertainty_seconds']
    return timestamp + correction, uncertainty


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_run(root, variant, expected_rounds=160, expected_epochs=40):
    all_rows, clock = [], []
    allocation = ALLOCATIONS[variant]
    for rank in range(5):
        rank_dir = root / 'ranks' / variant / ('rank%d' % rank)
        raw = read_csv(rank_dir / ('round_diagnostic_rank%d.csv' % rank))
        metadata = json.loads((rank_dir / ('clock_calibration_rank%d.json' % rank)).read_text())
        markers = metadata['calibrations']
        if len(markers) != expected_epochs + 1:
            raise ValueError('Calibration markers missing: ' + str(rank_dir))
        if len(raw) != expected_rounds:
            raise ValueError('Round count mismatch: ' + str(rank_dir))
        clock.append(dict(rank=rank, calibration_elapsed_seconds=metadata['calibration_elapsed_seconds'],
                          buffered_write_seconds=metadata['write_elapsed_seconds_before_this_flush'],
                          max_offset_uncertainty_ms=1000 * max(p['uncertainty_seconds'] for p in markers),
                          offset_change_ms=1000 * (markers[-1]['root_minus_local_seconds'] - markers[0]['root_minus_local_seconds'])))
        for index, record in enumerate(raw, 1):
            row = {k: float(v) for k, v in record.items()}
            if not all(math.isfinite(v) for v in row.values()):
                raise ValueError('Non-finite telemetry')
            if row['sync_round'] != index or row['rank'] != rank or row['local_batch'] != allocation[rank]:
                raise ValueError('Unexpected round/rank/allocation')
            if row['samples_processed'] != row['local_steps'] * allocation[rank]:
                raise ValueError('Sample count mismatch')
            if row['optimizer_ms'] < 0 or row['compute_ms'] < 0:
                raise ValueError('Invalid optimizer/compute envelope')
            if abs(row['compute_ms'] + row['optimizer_ms'] - row['compute_optimizer_ms']) > 1e-5:
                raise ValueError('Compute/optimizer timer mismatch')
            for key in TIMES:
                row['aligned_' + key], error = align(row[key], markers)
                if key == 'local_done_timestamp':
                    row['clock_uncertainty_ms'] = error * 1000
            row['local_elapsed_ms'] = (row['local_done_timestamp'] - row['local_start_timestamp']) * 1000
            row['local_other_ms'] = row['local_elapsed_ms'] - row['compute_optimizer_ms'] - row['data_ms']
            all_rows.append(row)
    windows = []
    for round_id in range(1, expected_rounds + 1):
        group = sorted([r for r in all_rows if r['sync_round'] == round_id], key=lambda r: r['rank'])
        if len({r['global_step'] for r in group}) != 1 or len({r['epoch'] for r in group}) != 1:
            raise ValueError('Ranks disagree on round identity')
        latest = max(group, key=lambda r: r['aligned_local_done_timestamp'])
        earliest = min(group, key=lambda r: r['aligned_local_done_timestamp'])
        latest_lower = max(r['aligned_local_done_timestamp'] - r['clock_uncertainty_ms'] / 1000 for r in group)
        candidates = [int(r['rank']) for r in group if r['aligned_local_done_timestamp'] + r['clock_uncertainty_ms'] / 1000 >= latest_lower]
        first_done = earliest['aligned_local_done_timestamp']
        spread_lower = max(0.0, 1000 * (latest_lower - min(
            r['aligned_local_done_timestamp'] + r['clock_uncertainty_ms'] / 1000 for r in group)))
        spread_upper = 1000 * (max(r['aligned_local_done_timestamp'] + r['clock_uncertainty_ms'] / 1000 for r in group)
                              - min(r['aligned_local_done_timestamp'] - r['clock_uncertainty_ms'] / 1000 for r in group))
        for row in group:
            row['readiness_after_first_ms'] = (row['aligned_local_done_timestamp'] - first_done) * 1000
        windows.append(dict(epoch=int(latest['epoch']), global_step=int(latest['global_step']), sync_round=round_id,
                            critical_rank=int(latest['rank']), critical_rank_unique_within_clock_uncertainty=len(candidates) == 1,
                            possible_critical_ranks=';'.join(map(str, candidates)),
                            readiness_spread_ms=1000 * (latest['aligned_local_done_timestamp'] - first_done),
                            readiness_spread_uncertainty_ms=max(
                                1000 * (latest['aligned_local_done_timestamp'] - first_done) - spread_lower,
                                spread_upper - 1000 * (latest['aligned_local_done_timestamp'] - first_done)),
                            readiness_spread_lower_ms=spread_lower, readiness_spread_upper_ms=spread_upper,
                            local_start_spread_ms=1000 * (max(r['aligned_local_start_timestamp'] for r in group) - min(r['aligned_local_start_timestamp'] for r in group)),
                            round_envelope_ms=1000 * (max(r['aligned_sync_exit_timestamp'] for r in group) - min(r['aligned_local_start_timestamp'] for r in group)),
                            rank0_round_elapsed_ms=group[0]['round_elapsed_ms'],
                            rank0_gather_ms=group[0]['gather_ms'], rank0_model_sync_ms=group[0]['model_sync_ms'],
                            rank0_wait_until_last_done_ms=max(0.0, 1000 * (latest['aligned_local_done_timestamp'] - group[0]['aligned_local_done_timestamp'])),
                            processed_samples=sum(r['samples_processed'] for r in group)))
    log = (root / (variant + '.log')).read_text(errors='replace')
    duration = re.findall(r'MPI Local SGD Training finished in ([0-9.]+)s', log)
    if len(duration) != 1:
        raise ValueError('Missing completion log')
    tests = re.findall(r'Final Test Evaluation -> Loss: ([0-9.]+), Accuracy: ([0-9.]+)%', log)
    epochs = read_csv(root / 'ranks' / variant / 'rank0/epoch_elapsed_rank0.csv')
    train = read_csv(root / 'ranks' / variant / 'rank0/train.csv')
    if len(epochs) != expected_epochs or len(train) != expected_epochs:
        raise ValueError('Epoch count mismatch')
    target = next((r for r in epochs if float(r['val_accuracy']) >= 0.65), None)
    result = dict(variant=variant, allocation=allocation, total_training_seconds=float(duration[0]),
                  best_validation_accuracy_percent=100 * max(float(r['val_accuracy']) for r in train),
                  final_validation_accuracy_percent=100 * float(train[-1]['val_accuracy']),
                  test_loss=float(tests[0][0]) if tests else None,
                  test_accuracy_percent=float(tests[0][1]) if tests else None,
                  first_epoch_validation_65=int(target['epoch']) if target else None,
                  time_to_validation_65_seconds=float(target['elapsed_training_seconds']) if target else None,
                  processed_samples=sum(r['processed_samples'] for r in windows), clock=clock)
    for scope, chosen in [('all_rounds', windows), ('excluding_epoch1', [r for r in windows if r['epoch'] > 1])]:
        if not chosen:
            continue
        counts = Counter(r['critical_rank'] for r in chosen)
        unique = Counter(r['critical_rank'] for r in chosen if r['critical_rank_unique_within_clock_uncertainty'])
        result[scope] = dict(
            readiness_spread_ms=stats([r['readiness_spread_ms'] for r in chosen]),
            round_envelope_ms=stats([r['round_envelope_ms'] for r in chosen]),
            rank0_round_elapsed_ms=stats([r['rank0_round_elapsed_ms'] for r in chosen]),
            rank0_gather_ms=stats([r['rank0_gather_ms'] for r in chosen]),
            rank0_model_sync_ms=stats([r['rank0_model_sync_ms'] for r in chosen]),
            critical_rank_counts={str(rank): counts[rank] for rank in range(5)},
            unambiguous_critical_rank_counts={str(rank): unique[rank] for rank in range(5)},
            ambiguous_rounds=sum(not r['critical_rank_unique_within_clock_uncertainty'] for r in chosen))
    result['per_rank'] = []
    for rank in range(5):
        selected = [r for r in all_rows if r['rank'] == rank]
        result['per_rank'].append(dict(rank=rank, local_batch=allocation[rank],
            compute_seconds=sum(r['compute_ms'] for r in selected) / 1000,
            optimizer_seconds=sum(r['optimizer_ms'] for r in selected) / 1000,
            compute_optimizer_seconds=sum(r['compute_optimizer_ms'] for r in selected) / 1000,
            gather_seconds=sum(r['gather_ms'] for r in selected) / 1000,
            model_sync_seconds=sum(r['model_sync_ms'] for r in selected) / 1000,
            local_other_seconds=sum(r['local_other_ms'] for r in selected) / 1000,
            local_elapsed_ms=stats([r['local_elapsed_ms'] for r in selected]),
            readiness_after_first_ms=stats([r['readiness_after_first_ms'] for r in selected])))
    if result['processed_samples'] != 300 * windows[-1]['global_step']:
        raise ValueError('Global sample budget does not match optimizer steps')
    if expected_epochs == 40:
        if windows[-1]['global_step'] != 6000 or result['processed_samples'] != 1800000:
            raise ValueError('Incomplete full-training budget')
        expected_steps = [(epoch - 1) * 150 + step for epoch in range(1, 41) for step in [40, 80, 120, 150]]
        if [r['global_step'] for r in windows] != expected_steps:
            raise ValueError('Unexpected full-training synchronization positions')
    write_csv(root / (variant + '_aligned_ranks.csv'), sorted(all_rows, key=lambda r: (r['sync_round'], r['rank'])))
    write_csv(root / (variant + '_rounds.csv'), windows)
    (root / (variant + '_analysis.json')).write_text(json.dumps(result, indent=2) + '\n')
    return result


def comparison_report(root, a, b):
    metrics = [
        ('Training loop 40e (s)', a['total_training_seconds'], b['total_training_seconds']),
        ('Best validation (%)', a['best_validation_accuracy_percent'], b['best_validation_accuracy_percent']),
        ('Validation E40 (%)', a['final_validation_accuracy_percent'], b['final_validation_accuracy_percent']),
        ('Test best-val checkpoint (%)', a['test_accuracy_percent'], b['test_accuracy_percent']),
        ('Epoch đầu val ≥65%', a['first_epoch_validation_65'], b['first_epoch_validation_65']),
        ('Time-to-val65 từ loop start (s)', a['time_to_validation_65_seconds'], b['time_to_validation_65_seconds']),
        ('Rank0 gather tổng (s)', a['per_rank'][0]['gather_seconds'], b['per_rank'][0]['gather_seconds']),
        ('Rank0 model sync tổng (s)', a['per_rank'][0]['model_sync_seconds'], b['per_rank'][0]['model_sync_seconds']),
    ]
    for field in ['readiness_spread_ms', 'round_envelope_ms', 'rank0_gather_ms', 'rank0_model_sync_ms']:
        for statistic in ['mean', 'median', 'p95']:
            metrics.append((field + ' / ' + statistic, a['all_rounds'][field][statistic], b['all_rounds'][field][statistic]))
    for field in ['readiness_spread_ms', 'round_envelope_ms']:
        for statistic in ['mean', 'median', 'p95']:
            metrics.append(('Bỏ epoch1: ' + field + ' / ' + statistic,
                            a['excluding_epoch1'][field][statistic], b['excluding_epoch1'][field][statistic]))
    def fmt(value):
        return 'Chưa đạt/không có' if value is None else ('%.3f' % value)
    table = '\n'.join('| %s | %s | %s | %s |' % (label, fmt(av), fmt(bv), fmt(bv-av) if av is not None and bv is not None else '—') for label, av, bv in metrics)
    rank_rows, frequency_rows = [], []
    for ra, rb in zip(a['per_rank'], b['per_rank']):
        rank = ra['rank']
        rank_rows.append('| %d | %d → %d | %.2f / %.2f | %.2f / %.2f | %.2f / %.2f | %.2f / %.2f | %.2f / %.2f |' % (
            rank, ra['local_batch'], rb['local_batch'], ra['compute_seconds'], rb['compute_seconds'],
            ra['optimizer_seconds'], rb['optimizer_seconds'], ra['compute_optimizer_seconds'], rb['compute_optimizer_seconds'],
            ra['readiness_after_first_ms']['mean'], rb['readiness_after_first_ms']['mean'], ra['model_sync_seconds'], rb['model_sync_seconds']))
        ca = a['all_rounds']['critical_rank_counts'][str(rank)]
        cb = b['all_rounds']['critical_rank_counts'][str(rank)]
        n = a['all_rounds']['readiness_spread_ms']['n']
        frequency_rows.append('| %d | %d / %d (%.2f%%) | %d / %d (%.2f%%) |' % (rank, ca, n, 100*ca/n, cb, n, 100*cb/n))
    system = dict(
        readiness_mean_decreased=b['all_rounds']['readiness_spread_ms']['mean'] < a['all_rounds']['readiness_spread_ms']['mean'],
        rank0_gather_total_decreased=b['per_rank'][0]['gather_seconds'] < a['per_rank'][0]['gather_seconds'],
        round_envelope_mean_decreased=b['all_rounds']['round_envelope_ms']['mean'] < a['all_rounds']['round_envelope_ms']['mean'],
        training_loop_decreased=b['total_training_seconds'] < a['total_training_seconds'])
    system['time_to_65_not_slower_when_control_reaches'] = (
        a['time_to_validation_65_seconds'] is None
        or (b['time_to_validation_65_seconds'] is not None
            and b['time_to_validation_65_seconds'] <= a['time_to_validation_65_seconds']))
    # Predeclared diagnostic flag, not a statistical non-inferiority claim.
    quality_loss = (b['best_validation_accuracy_percent'] < a['best_validation_accuracy_percent'] - 0.5
                    or b['test_accuracy_percent'] < a['test_accuracy_percent'] - 0.5
                    or (a['first_epoch_validation_65'] is not None and b['first_epoch_validation_65'] is None))
    positive = all(system.values()) and not quality_loss
    verdict = ('Có tín hiệu tích cực trong paired seed42: cả readiness, gather wait, round time và training time giảm; không kích hoạt cờ suy giảm quality đã đặt trước.'
               if positive else 'Chưa đủ điều kiện kết luận rebalance có tín hiệu tích cực đồng thời về critical path và quality theo tiêu chí đã đặt trước.')
    text = '''# Paired workload diagnostic — FP32 Local SGD H40, seed42, 40 epochs

Control A = [266,6,6,6,16]; mild rebalance B = [266,6,5,5,18]. Global batch đều 300.
Hai run dùng cùng snapshot source, optimizer, LR 0,001, augmentation, evaluation và FP32 sample-weighted AllReduce. Không adaptive H, EF, FP16, INT8 hoặc clustering.

## Kết quả A/B

Δ = B − A. Timestamp đã align về monotonic clock rank0 bằng hiệu chỉnh nội suy giữa các markers ở ranh giới epoch. Target validation 65% được đặt trước khi chạy, elapsed đo trực tiếp sau validation và trước checkpoint saving.

| Metric | A | B | Δ |
|---|---:|---:|---:|
{table}

## Per-rank compute và readiness

Mỗi ô hai số là A / B. Compute riêng = train-step wall time trừ optimizer envelope; combined là metric tương thích compute+optimizer của baseline cũ. Optimizer envelope dùng host monotonic callbacks và các dependency phục vụ đo thời gian, gồm aggregation nội bộ MirroredStrategy tại rank0; không phải tổng kernel time GPU thuần. Deterministic ops giữ nguyên. Các duration là tổng theo rank, không cộng xuyên ranks thành wall time.

| Rank | Batch A→B | Compute (s) A/B | Optimizer (s) A/B | Combined (s) A/B | Mean arrival sau rank đầu (ms) A/B | Model sync tổng (s) A/B |
|---|---:|---:|---:|---:|---:|---:|
{ranks}

## Critical-rank frequency

Critical rank danh nghĩa = argmax aligned local_done. Những round có khoảng bất định clock chồng lấn được ghi riêng trong JSON và không coi thứ hạng đó chắc chắn.

| Rank | A | B |
|---|---:|---:|
{frequencies}

Ambiguous rounds do clock uncertainty: A={amb_a}, B={amb_b}. Clock RTT/uncertainty, offset drift, thời gian calibration và flush được lưu trong analysis JSON. Có thêm thống kê bỏ epoch1 trong JSON để tách khởi động/JIT; bảng chính giữ đủ 40 epochs.

## Đánh giá tiêu chí đồng thời

```json
{criteria}
```

{verdict}

Quality flag đặt trước: best validation hoặc final test giảm trên 0,5 điểm phần trăm, hoặc B không tới val65 khi A tới. Đây là ngưỡng chẩn đoán thực dụng, không phải chứng minh non-inferiority. Time-to-val65, final val và đường convergence cần được đọc cùng bảng; không lấy riêng compute rank2/3 giảm để nhận B tốt hơn.

## Phạm vi kết luận

Đây là một paired pilot seed42 chạy A rồi B, không randomize thứ tự. Round samples trong cùng run không phải independent training seeds. Nhiễu hệ thống theo thời gian và thay đổi phân phối samples/moments local có thể ảnh hưởng kết quả. Loader hiện tại repartition shards theo allocation và averaging vẫn sample-weighted; thay batch không giữ nguyên training trajectory. Không suy rộng kết quả này thành bảo đảm mọi workload/cụm đều được cải thiện.

Readiness spread = max(done) − min(done). `round_envelope` = max(sync_exit) − min(local_start); các envelopes có thể overlap giữa rounds nên không cộng chúng thành training time. Gather/model-sync là elapsed, có thể gồm chờ, không phải wire time thuần. Calibration dùng half network RTT làm ước lượng bất định offset; nội suy không bảo đảm giới hạn tuyệt đối cho clock-rate biến động giữa markers. Total loop gồm calibration/flush tại ranh giới epoch; startup calibration nằm ngoài loop. Không thêm collective/barrier vào từng round ngoài FP32 baseline.

## Nguồn

- `config_check.json`: kiểm tra A/B và baseline trước training.
- `source_stage.txt`, `source_manifest.json`, `telemetry.patch`: snapshot và thay đổi telemetry.
- `A_control.log`, `B_mild_rebalance.log`: full launcher logs.
- `ranks/<variant>/rankN/round_diagnostic_rankN.csv`, `clock_calibration_rankN.json`, `epoch_elapsed_rankN.csv`: raw telemetry.
- `<variant>_aligned_ranks.csv`, `<variant>_rounds.csv`, `<variant>_analysis.json`: kết quả tính từng rank/round.
'''.format(table=table, ranks='\n'.join(rank_rows), frequencies='\n'.join(frequency_rows),
           amb_a=a['all_rounds']['ambiguous_rounds'], amb_b=b['all_rounds']['ambiguous_rounds'],
           criteria=json.dumps(dict(**system, quality_loss_flag=quality_loss, positive_diagnostic_signal=positive), indent=2), verdict=verdict)
    (root / 'paired_diagnostic_report_vi.md').write_text(text)
    (root / 'comparison.json').write_text(json.dumps(dict(A=a, B=b, criteria=system,
                                                       quality_loss_flag=quality_loss,
                                                       positive_diagnostic_signal=positive), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    args.root = resolve_result_path(args.root)
    a = analyze_run(args.root, 'A_control')
    b = analyze_run(args.root, 'B_mild_rebalance')
    comparison_report(args.root, a, b)
    print('PASS: report saved to', args.root / 'paired_diagnostic_report_vi.md')
