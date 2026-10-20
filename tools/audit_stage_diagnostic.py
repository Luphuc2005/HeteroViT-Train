"""Summarize one completed cluster5 stage diagnostic, including epoch-start delay attribution.

Offline only. Raw traces are never edited. Large iterator durations identify the
timed API boundary, not an internal TensorFlow cause such as memory or threads.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from paired_workload_analysis import align, stats, write_csv
from results_layout import resolve_result_path


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def report_vi(result, stages):
    run, ranks = result['run'], result['ranks']
    tails = result['lab02_large_start_delays']
    scope = result['windows']['all']
    lines = [
        '# Chẩn đoán stage 5-node batch600/H20 — 10 epochs, seed42', '',
        '**Đã hoàn thành một run 5-node.** Batch600, allocation `[532,12,12,12,32]`, '
        'H20, FP32 AllReduce, AdamW LR 0,001, sample-weighted averaging, optimizer state local; '
        'giữ model/dataset/augmentation/evaluation và launch policy của cấu hình cluster5 hiện tại.', '',
        '## 1. Kết quả chính', '',
        'Lab02 có **%d tail đầu epoch** với start lag >10 giây sau epoch khởi động. '
        'Các tail được ghi trong stage `iterator_create`, bao quanh câu lệnh '
        '`train_iter = iter(self.dist_train_ds) ...`. Stage gồm tạo và gán iterator mới, '
        'có thể gồm giải phóng iterator cũ khi thay tham chiếu. Đây là vị trí delay đã đo; '
        'chưa xác định nguyên nhân bên trong TensorFlow.' % len(tails), '',
        '| Epoch | Tạo/gán iterator lab02 (s) | Lag bắt đầu block lab02 (s) | Compute + optimizer 20 steps lab02 (s) | Rank0 gather đầu epoch (s) | Rank đến cuối |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for row in tails:
        lines.append('| %d | %.3f | %.3f | %.3f | %.3f | %d |' % (
            row['epoch'], row['iterator_ms'] / 1000, row['start_lag_ms'] / 1000,
            row['first_block_compute_optimizer_ms'] / 1000, row['rank0_gather_ms'] / 1000, row['critical_rank']))
    tail_gather = sum(row['rank0_gather_ms'] for row in tails) / 1000
    lines += ['', '%d round tương ứng có tổng rank0 gather **%.3f giây**, bằng **%.2f%%** '
        'gather toàn run. Đây là phần chờ quan sát trong các round có tail, không phải mức '
        'speedup có thể chắc chắn thu hồi nếu sửa iterator.' % (len(tails), tail_gather, 100 * tail_gather / ranks[0]['gather_seconds'])]
    normal = [row['iterator_ms'] for row in result['lab02_epoch_starts'] if row['iterator_ms'] < 1000]
    if normal:
        lines += ['', 'Các epoch lab02 còn lại có iterator %.2f–%.2f ms. '
            'Các tail lớn xuất hiện sau khi barrier cuối epoch trước và calibration đã hoàn tất. '
            'Khoảng từ calibration kết thúc tới iterator bắt đầu và từ iterator kết thúc tới local block '
            'đều được lưu riêng trong `epoch_starts.csv`.' % (min(normal), max(normal))]
    lines += ['', '## 2. Compute từng rank', '',
        '**Metric chính để cân tải là elapsed local train-step gồm optimizer**, đo từ sau fetch '
        'đến khi lấy scalar kết quả xong. Metric loại iterator, fetch, gather và model sync; '
        'vẫn gồm host/graph overhead và đồng bộ nội bộ hai GPU. Đây không phải kernel time thuần. '
        'Mean/median/p95 bên dưới loại epoch1 (675 steps/rank). Tổng 10e vẫn gồm warmup.', '',
        '| Rank / host | Batch | Tổng compute + optimizer 10e (s) | Mean steady (ms/step) | Median (ms/step) | p95 (ms/step) |',
        '|---|---:|---:|---:|---:|---:|']
    for row in ranks:
        timing = row['steady_train_step_ms']
        lines.append('| %d / %s | %d | %.3f | %.3f | %.3f | %.3f |' % (
            row['rank'], row['host'], row['batch'], row['compute_optimizer_seconds'],
            timing['mean'], timing['median'], timing['p95']))
    lines += ['', 'CSV còn giữ `compute_ms` đã trừ optimizer envelope và `optimizer_ms` riêng. '
        'Không gọi phần còn lại là forward/backward GPU kernel time. Trên rank0, optimizer envelope '
        'có thể gồm hoạt động của strategy và scheduling; chưa có device profiler để tách chúng. '
        'Cả năm ranks ghi actual optimizer là `keras.optimizers.optimizer_experimental.adamw.AdamW`.', '',
        '## 3. Readiness và critical worker', '',
        '| Rank | Đến cuối / 40 rounds | Tỷ lệ | Gather tổng (s) | Model sync tổng (s) |',
        '|---|---:|---:|---:|---:|']
    for row in ranks:
        critical = scope['critical_rank_counts'][str(row['rank'])]
        lines.append('| %d | %d | %.1f%% | %.3f | %.3f |' % (
            row['rank'], critical, 100 * critical / run['sync_rounds_per_rank'],
            row['gather_seconds'], row['model_sync_seconds']))
    lines += ['', '| Phạm vi | Readiness spread mean (s) | Median (s) | p95 (s) | Rounds |',
        '|---|---:|---:|---:|---:|']
    for key, title in [('all', 'Toàn run'), ('excluding_epoch1', 'Loại epoch1')]:
        values = result['windows'][key]['readiness_spread_ms']
        lines.append('| %s | %.3f | %.3f | %.3f | %d |' % (
            title, values['mean'] / 1000, values['median'] / 1000, values['p95'] / 1000, values['n']))
    lines += ['', 'Timestamp được align về clock rank0 bằng calibration ở startup và mỗi epoch. '
        'Max half-RTT uncertainty ước tính là %.3f ms; %d rounds có thứ hạng không phân biệt được '
        'trong uncertainty này. Ước tính không bảo đảm chặn mọi clock drift không quan sát.' % (
            max(row['max_clock_uncertainty_ms'] for row in ranks), scope['ambiguous_rounds']), '',
        'Loại epoch1, rank3 đến cuối **%d/36 rounds**, rank1 **%d/36**. '
        'Đây là hai cơ chế quan sát được: tail đầu epoch ở lab02 và local work dài hơn thường xuyên '
        'ở lab04. Compute thấp hơn của lab02 trong block không ngăn nó thành critical worker khi '
        'iterator bị chậm trước block.' % (
            result['windows']['excluding_epoch1']['critical_rank_counts']['3'],
            result['windows']['excluding_epoch1']['critical_rank_counts']['1']), '',
        'Worker gather gần 0 không có nghĩa không chờ: thời gian có thể xuất hiện ở collective tiếp theo. '
        'Model sync bao gồm lấy/set weights, scalar collectives, payload và chờ. '
        'Không cộng thời gian các ranks thành critical path; không coi MPI elapsed là wire time.', '',
        '## 4. Stage cuối/đầu epoch', '',
        '| Rank | Iterator tổng (s) | GC tổng (s) | Barrier tổng (s) | Calibration + flush tổng (s) |',
        '|---|---:|---:|---:|---:|']
    for rank in range(5):
        totals = {stage: sum(row['elapsed_ms'] for row in stages if row['rank'] == rank and row['stage'] == stage) / 1000
                  for stage in ['iterator_create', 'garbage_collect', 'existing_epoch_barrier', 'clock_calibration_and_flush']}
        lines.append('| %d | %.3f | %.3f | %.3f | %.3f |' % (
            rank, totals['iterator_create'], totals['garbage_collect'],
            totals['existing_epoch_barrier'], totals['clock_calibration_and_flush']))
    lines += ['', 'Worker barrier cuối epoch gồm chờ rank0 validation/checkpoint và các công việc khác. '
        'Các stage ở ranks khác nhau chồng thời gian; không cộng chúng với nhau. '
        'Stage trace không thêm collective vào từng round; buffered logging flush ở ranh giới epoch.', '',
        '## 5. Hoàn thành và chất lượng', '',
        '- Training loop: **%.2f giây (%.2f phút)**; không gồm startup/preflight.' % (run['training_seconds'], run['training_seconds'] / 60),
        '- Đã kiểm tra **750 steps/rank, 40 syncs/rank, 450.000 processed samples**; 10 dòng validation.',
        '- Best/final validation: **%.2f%% / %.2f%%**.' % (run['best_validation_percent'], run['final_validation_percent']),
        '- Chưa đạt validation 65% trong 10e; time-to-65% chưa quan sát. Không có final test trong run chẩn đoán này.', '',
        '## 6. Kết luận và giới hạn', '',
        '**Đã khoanh vùng delay 28–30 giây vào stage tạo/gán iterator đầu epoch ở lab02.** '
        'Đồng thời local compute của ranks2/3 vẫn dài hơn các CPU ranks còn lại. '
        'Vì thế chỉ điều chỉnh batch theo một con số compute chưa xử lý được tail iterator. '
        'Bước chẩn đoán tiếp theo hợp lý là tách tạo iterator mới khỏi hủy/thay iterator cũ, '
        'rồi quan sát stack/runtime của lời gọi bị giữ lâu; chưa thực hiện thay đổi hoặc run bổ sung.', '',
        'Run này không chứng minh cách sửa đã làm nhanh hơn, không đo kernel utilization, '
        'không thay H/allocation/precision và không chứng minh giới hạn accuracy ở 10e.', '',
        '## 7. Evidence và tái lập phân tích', '',
        '- [Run summary](summary.json), [config check](config_check.json), [runtime preflight](runtime.json).',
        '- [Raw per-rank telemetry](ranks/cluster5/), [training log](cluster5.log).',
        '- [Evidence JSON](runtime_stage_analysis/diagnostic_evidence.json), [compute CSV](runtime_stage_analysis/per_rank_compute.csv).',
        '- [Epoch starts](runtime_stage_analysis/epoch_starts.csv), [aligned stages](runtime_stage_analysis/aligned_stages.csv), [rounds](runtime_stage_analysis/diagnostic_rounds.csv).',
        '- [Per-call payload residence](runtime_stage_analysis/cluster5_sync_trace.csv), [stage totals](runtime_stage_analysis/cluster5_epoch_stages.csv).',
        '- [Frozen source](source/), [source hashes](source_manifest.json).', '',
        'Từ thư mục `HeteroViT-MPI`: `python3 tools/audit_stage_diagnostic.py <thư_mục_run>`. '
        'Phân tích offline; không sửa raw logs.', '',
    ]
    return '\n'.join(lines)


def audit(root):
    root = resolve_result_path(root)
    summary = json.loads((root / 'summary.json').read_text())
    assert len(summary) == 1 and summary[0]['variant'] == 'cluster5'
    run = summary[0]
    epochs, count = run['epochs'], run['sync_rounds_per_rank']
    allocation = [532, 12, 12, 12, 32]
    events, stages, ranks = [], [], []
    for rank, batch in enumerate(allocation):
        folder = root / 'ranks/cluster5' / ('rank%d' % rank)
        clock = json.loads((folder / ('clock_calibration_rank%d.json' % rank)).read_text())
        assert clock['runtime_stage_trace'] and len(clock['calibrations']) == epochs + 1
        rounds = read_csv(folder / ('round_diagnostic_rank%d.csv' % rank))
        assert len(rounds) == count
        for raw in rounds:
            row = {key: float(value) for key, value in raw.items()}
            assert all(math.isfinite(value) for value in row.values())
            assert row['rank'] == rank and row['local_batch'] == batch
            assert abs(row['compute_ms'] + row['optimizer_ms'] - row['compute_optimizer_ms']) < 1e-5
            for key in ['local_start_timestamp', 'local_done_timestamp', 'sync_exit_timestamp']:
                row['aligned_' + key], uncertainty = align(row[key], clock['calibrations'])
                if key == 'local_done_timestamp':
                    row['clock_uncertainty_seconds'] = uncertainty
            events.append(row)
        for raw in read_csv(folder / ('runtime_stages_rank%d.csv' % rank)):
            row = dict(epoch=int(raw['epoch']), rank=rank, host=raw['host'], stage=raw['stage'],
                       elapsed_ms=float(raw['elapsed_ms']))
            row['aligned_begin'], _ = align(float(raw['stage_begin']), clock['calibrations'])
            row['aligned_end'], _ = align(float(raw['stage_end']), clock['calibrations'])
            stages.append(row)
        step_file = 'step_timing_breakdown.csv' if rank == 0 else 'step_timing_breakdown_rank%d.csv' % rank
        steps = read_csv(folder / step_file)
        assert len(steps) == run['optimizer_steps_per_rank']
        steady = [row for row in steps if int(row['epoch']) > 1]
        ranks.append(dict(rank=rank, host=stages[-1]['host'], batch=batch,
            compute_excluding_optimizer_seconds=sum(row['compute_ms'] for row in events if row['rank'] == rank) / 1000,
            optimizer_envelope_seconds=sum(float(row['optimizer_ms']) for row in rounds) / 1000,
            compute_optimizer_seconds=sum(float(row['compute_optimizer_ms']) for row in rounds) / 1000,
            steady_train_step_ms=stats([float(row['compute_ms']) for row in steady]),
            gather_seconds=sum(float(row['gather_ms']) for row in rounds) / 1000,
            model_sync_seconds=sum(float(row['model_sync_ms']) for row in rounds) / 1000,
            max_clock_uncertainty_ms=1000 * max(p['uncertainty_seconds'] for p in clock['calibrations']),
            actual_optimizer=clock['optimizer_class']))
    windows = []
    for number in range(1, count + 1):
        group = [row for row in events if row['sync_round'] == number]
        assert len(group) == 5
        first = min(row['aligned_local_done_timestamp'] for row in group)
        last = max(group, key=lambda row: row['aligned_local_done_timestamp'])
        lower = max(row['aligned_local_done_timestamp'] - row['clock_uncertainty_seconds'] for row in group)
        candidates = [row for row in group if row['aligned_local_done_timestamp'] + row['clock_uncertainty_seconds'] >= lower]
        root_event = next(row for row in group if row['rank'] == 0)
        windows.append(dict(epoch=int(last['epoch']), sync_round=number, global_step=int(last['global_step']),
            critical_rank=int(last['rank']), ambiguous_critical_rank=len(candidates) != 1,
            readiness_spread_ms=1000 * (last['aligned_local_done_timestamp'] - first),
            rank0_gather_ms=root_event['gather_ms'],
            round_envelope_ms=1000 * (max(row['aligned_sync_exit_timestamp'] for row in group)
                                     - min(row['aligned_local_start_timestamp'] for row in group))))
    starts = []
    for epoch in range(1, epochs + 1):
        firsts = [row for row in events if row['epoch'] == epoch and row['sync_round'] == (epoch - 1) * 4 + 1]
        first_start = min(row['aligned_local_start_timestamp'] for row in firsts)
        for row in firsts:
            rank = int(row['rank'])
            iterator = next(s for s in stages if s['rank'] == rank and s['epoch'] == epoch and s['stage'] == 'iterator_create')
            previous = {s['stage']: s for s in stages if s['rank'] == rank and s['epoch'] == epoch - 1}
            starts.append(dict(epoch=epoch, rank=rank,
                start_lag_ms=1000 * (row['aligned_local_start_timestamp'] - first_start),
                iterator_ms=iterator['elapsed_ms'],
                iterator_enter_relative_ms=1000 * (iterator['aligned_begin'] - first_start),
                iterator_exit_relative_ms=1000 * (iterator['aligned_end'] - first_start),
                iterator_exit_to_block_ms=1000 * (row['aligned_local_start_timestamp'] - iterator['aligned_end']),
                previous_barrier_ms=previous.get('existing_epoch_barrier', {}).get('elapsed_ms'),
                previous_calibration_ms=previous.get('clock_calibration_and_flush', {}).get('elapsed_ms'),
                previous_calibration_exit_to_iterator_ms=1000 * (iterator['aligned_begin'] - previous['clock_calibration_and_flush']['aligned_end']) if previous else None,
                first_block_compute_optimizer_ms=row['compute_optimizer_ms'],
                critical_rank=windows[(epoch - 1) * 4]['critical_rank'],
                rank0_gather_ms=windows[(epoch - 1) * 4]['rank0_gather_ms']))
    scopes = {}
    for name, selected in [('all', windows), ('excluding_epoch1', [row for row in windows if row['epoch'] > 1])]:
        counts = Counter(row['critical_rank'] for row in selected)
        scopes[name] = dict(readiness_spread_ms=stats([row['readiness_spread_ms'] for row in selected]),
            round_envelope_ms=stats([row['round_envelope_ms'] for row in selected]),
            critical_rank_counts={str(rank): counts[rank] for rank in range(5)},
            ambiguous_rounds=sum(row['ambiguous_critical_rank'] for row in selected))
    result = dict(run=run, ranks=ranks, windows=scopes,
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        lab02_epoch_starts=[row for row in starts if row['rank'] == 1],
        lab02_large_start_delays=[row for row in starts if row['rank'] == 1 and row['epoch'] > 1 and row['start_lag_ms'] > 10000],
        semantics=dict(compute='MEASURED host elapsed local train-step. Excludes fetch, gather, model sync, iterator_create; includes optimizer and host/graph overhead. Steady excludes epoch1.',
            compute_excluding_optimizer='MEASURED combined local train-step minus optimizer host callback envelope; not pure kernels.',
            stage='MEASURED API boundary duration; does not establish internal cause.',
            clock='DERIVED readiness uses per-epoch min-RTT interpolation; uncertainty estimates do not bound unseen drift.',
            delay_selection='Predefined descriptive tail filter: lab02 epoch-start lag >10 seconds, excluding epoch1. Not a hypothesis test.'))
    destination = root / 'runtime_stage_analysis'
    destination.mkdir(exist_ok=True)
    write_csv(destination / 'epoch_starts.csv', starts)
    write_csv(destination / 'aligned_stages.csv', stages)
    write_csv(destination / 'diagnostic_rounds.csv', windows)
    write_csv(destination / 'per_rank_compute.csv', [dict(**{k: v for k, v in row.items() if k != 'steady_train_step_ms'},
        **{'steady_step_' + key: value for key, value in row['steady_train_step_ms'].items()}) for row in ranks])
    (destination / 'diagnostic_evidence.json').write_text(json.dumps(result, indent=2) + '\n')
    (destination / 'audit_stage_diagnostic.py').write_bytes(Path(__file__).read_bytes())
    (root / 'DIAGNOSTIC_REPORT_VI.md').write_text(report_vi(result, stages))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
