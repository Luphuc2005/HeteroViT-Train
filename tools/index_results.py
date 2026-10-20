"""Organize experiment directories into groups and rebuild the results catalog."""
import argparse
import fcntl
import csv
import json
import os
from pathlib import Path
import re
from datetime import datetime, timezone

import yaml
from results_layout import result_roots


class SavedConfigLoader(yaml.SafeLoader):
    """Read historical ConfigDict data without instantiating Python objects."""


def config_dict(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    return mapping.get('dictitems', mapping)


SavedConfigLoader.add_constructor('tag:yaml.org,2002:python/object/new:src.utils.config.ConfigDict', config_dict)


NAV = '00_BY_EXPERIMENT'
GROUPS = {
    '01_GPU_ONLY_BATCH300': ('GPU-only batch300 — seeds42/43/44', 'Baseline một MPI rank, hai GPU lab01; global batch300, H40.'),
    '02_FP16_BATCH300_PRIMARY': ('FP16 batch300 — bộ ba seeds chính', 'Một sweep đủ năm methods cho mỗi seed42/43/44, tổng 15 runs chính.'),
    '03_WORKLOAD_BALANCE_BATCH300': ('Workload A/B và B chạy lại', 'Paired seed42 A=[266,6,6,6,16], B=[266,6,5,5,18]; rerun cùng seed không phải seed độc lập.'),
    '04_BATCH600_H20': ('Batch600/H20 — GPU-only so với 5 máy', 'Allocation [600] và [532,12,12,12,32]. Phân biệt pilot/diagnostic 3/5/10e và full 40e qua tên/status/config.'),
    '05_LOCAL_SGD_H_SWEEPS': ('Baseline AllReduce và các giá trị H', 'Các experiments lịch sử; kiểm tra epochs, seed và evaluation trước khi so với bảng mới.'),
    '06_ADAPTIVE_EXPLORATORY': ('Adaptive H / consensus / thử nghiệm', 'Exploratory, không tự gộp vào thống kê FP16/GPU-only ba seeds.'),
    '07_EF_AND_DIAGNOSTICS': ('EF, same-input diagnostic và reruns FP16', 'Giữ paired EF riêng; diagnostic/smoke không phải bằng chứng final accuracy của full training.'),
    '08_SMOKE_FAILED_PREPARED': ('Smoke, setup, failed, prepared', 'Giữ nguyên artifacts để truy lỗi; không coi là full experiment hoàn chỉnh.'),
    '09_EARLY_FP16_SUPERSEDED': ('FP16 lịch sử / evaluation cũ', 'Không dùng thay sweep seed42 canonical uydYAoMC trong mean ba seeds.'),
    '98_OTHER_REVIEW': ('Chưa phân loại — cần xem lại', 'Các thư mục chưa khớp quy tắc; kiểm tra config trước khi dùng số liệu.'),
    '99_ARCHIVE': ('Archive lịch sử', 'Các nhóm lịch sử đã có sẵn; không gộp thành thêm seeds.'),
}
PRIMARY = {'fp16_sweep_uydYAoMC': 42, 'fp16_sweep_seed43_RIHAAkW2': 43,
           'fp16_sweep_seed44_swTaOOce': 44}
ALIASES = {
    'fp16_sweep_uydYAoMC': 'seed42_canonical',
    'fp16_sweep_seed43_RIHAAkW2': 'seed43_canonical',
    'fp16_sweep_seed44_swTaOOce': 'seed44_canonical',
    'local_sgd_h40_fp32_2gpu_only_300_40e_20261019_131721': 'seed42_gpu_only',
    'paired_workload_seed42_20261019_102016_6vd7mn35': 'seed42_A_vs_B',
    'paired_workload_seed42_20261019_122142_ws_di4yu': 'seed42_B_repeat',
    'fp16_sweep_seed43_2tyCNQ2H': 'seed43_EF_paired',
    'fp16_sweep_seed43_2yEHiooV': 'seed43_FP32_Delta_rerun',
    'fp16_sweep_seed43_E2dtk5hD': 'seed43_same_input_360steps',
    'fp16_sweep_83LF5CdI': 'seed42_obsolete_evaluation',
}
REPORTS = {
    '01_GPU_ONLY_BATCH300': ['gpu_only_vs_cluster_20261019_vi.md'],
    '02_FP16_BATCH300_PRIMARY': ['fp16_results_by_seed_20261015_vi.md', 'bottleneck_story_20261018_vi.md'],
    '03_WORKLOAD_BALANCE_BATCH300': ['run_paired_workload_diagnostic_vi.md'],
    '04_BATCH600_H20': ['batch600_stage_diagnostic_20261019_vi.md', 'batch600_runtime_audit_20261019_vi.md', 'batch600_results_20261019_vi.md', 'run_batch600_comparison_vi.md'],
    '05_LOCAL_SGD_H_SWEEPS': ['fp16_results_by_seed_20261015_vi.md'],
    '07_EF_AND_DIAGNOSTICS': ['fp16_results_by_seed_20261015_vi.md'],
}


def classify(path, status):
    name = path.name
    if name == 'archive':
        return '99_ARCHIVE'
    if status.startswith(('failed', 'prepared', 'stopped')) or 'smoke' in name:
        return '08_SMOKE_FAILED_PREPARED'
    if name.startswith('batch600_'):
        return '04_BATCH600_H20'
    if name.startswith('gpu_only_seed') or '2gpu_only' in name:
        return '01_GPU_ONLY_BATCH300'
    if name in PRIMARY:
        return '02_FP16_BATCH300_PRIMARY'
    if name.startswith('paired_workload_'):
        return '03_WORKLOAD_BALANCE_BATCH300'
    if 'adaptive' in name:
        return '06_ADAPTIVE_EXPLORATORY'
    if name == 'fp16_sweep_83LF5CdI':
        return '09_EARLY_FP16_SUPERSEDED'
    if name.startswith('fp16_sweep_'):
        return '07_EF_AND_DIAGNOSTICS'
    if '_fp16' in name:
        return '09_EARLY_FP16_SUPERSEDED'
    if name.startswith(('local_sgd_h', 'allreduce_baseline_')):
        return '05_LOCAL_SGD_H_SWEEPS'
    return '98_OTHER_REVIEW'


def relative(target, parent):
    return Path(os.path.relpath(str(target), str(parent))).as_posix()


def link_to(target, destination):
    target = target.absolute()
    assert target.exists(), 'Missing link target: ' + str(target)
    if destination.is_symlink():
        if destination.resolve() != target.resolve():
            raise ValueError('Refusing to replace a different link: ' + str(destination))
    elif destination.exists():
        raise ValueError('Refusing to replace a real file/directory: ' + str(destination))
    else:
        destination.symlink_to(relative(target, destination.parent), target_is_directory=target.is_dir())


def inspect(path):
    status_file = path / 'status.txt'
    status = status_file.read_text().strip().splitlines()[0] if status_file.exists() else ''
    config_file = path / 'config.yaml'
    config = yaml.load(config_file.read_text(), Loader=SavedConfigLoader) if config_file.exists() else {}
    config = config or {}
    training = config.get('training', {})
    seed = config.get('seed', PRIMARY.get(path.name, ''))
    if seed == '':
        match = re.search(r'seed[_-]?(\d+)', path.name)
        if match:
            seed = match.group(1)
    if path.name.startswith('gpu_only_seed43_44_'):
        seed = '43,44'
    if path.name.startswith('batch600_'):
        batch, h = 600, 20
    elif path.name in PRIMARY or path.name.startswith(('paired_workload_', 'gpu_only_seed43_44_')):
        batch, h = 300, 40
    else:
        batch, h = training.get('global_batch_size', ''), training.get('local_sgd_h', '')
    observed = ''
    if (path / 'train.csv').exists():
        with (path / 'train.csv').open(newline='') as stream:
            observed = len(list(csv.DictReader(stream)))
    log_path = path / 'run.log'
    text = log_path.read_text(errors='replace') if log_path.exists() else ''
    tests = re.findall(r'Final Test Evaluation -> Loss:\s*[\d.]+, Accuracy:\s*([\d.]+)%', text)
    duration = re.findall(r'Total Training Time\s*:\s*([\d.]+)s', text)
    if not status:
        status = 'completed_with_final_test' if tests else 'historical_unverified'
    if path.name == 'archive':
        status = 'historical_archive'
    group = classify(path, status)
    return dict(group=group, name=path.name, seed=seed, global_batch=batch, H=h,
                epochs_config=training.get('epochs', ''), observed_epoch_rows=observed,
                status=status, test_accuracy_percent=tests[-1] if tests else '',
                training_seconds=duration[-1] if duration else '',
                original_path=str(path),
                browse_path='results/' + NAV + '/' + group + '/' + path.name,
                note=GROUPS[group][1])


def organize(repo):
    root = repo / 'results'
    nav = root / NAV
    originals = result_roots(repo)
    plan = []
    for path in originals:
        row = inspect(path)
        target = nav / row['group'] / path.name
        if path != target:
            if target.exists() and not target.is_symlink():
                raise ValueError('Destination already exists: ' + str(target))
            plan.append((path, target, path.stat().st_ino))
    nav.mkdir(exist_ok=True)
    # Remove generated shortcuts only. Experiment trees contain no shortcuts.
    for shortcut in list(nav.rglob('*')):
        if shortcut.is_symlink():
            shortcut.unlink()
    record = nav / 'relocations.json'
    previous = json.loads(record.read_text()) if record.exists() else []
    entries = [dict(old=str(a), new=str(b), inode=inode) for a, b, inode in plan]
    record.write_text(json.dumps(previous + entries, indent=2) + '\n')
    for source, target, inode in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        assert target.stat().st_ino == inode
        print('MOVED:', source.name, '->', target.relative_to(root))
    return plan


def build(repo):
    root = repo / 'results'
    nav = root / NAV
    originals = result_roots(repo)
    nav.mkdir(exist_ok=True)
    rows = []
    for original in originals:
        row = inspect(original)
        row['original_path'] = original.relative_to(repo).as_posix()
        row['browse_path'] = row['original_path']
        rows.append(row)
    legacy = nav / 'README_before_index.md'
    if (root / 'README.md').exists() and not legacy.exists():
        legacy.write_text((root / 'README.md').read_text())
    for group, (title, note) in GROUPS.items():
        folder = nav / group
        folder.mkdir(exist_ok=True)
        entries = [row for row in rows if row['group'] == group]
        lines = ['# ' + title, '', note, '', '[← Mục lục](../README.md)', '',
                 '| Run / sweep | Seed | Batch | H | Trạng thái |',
                 '|---|---|---|---|---|']
        for row in entries:
            target = repo / row['original_path']
            state = row['status'].replace('|', '/').replace('\n', ' ')[:100]
            lines.append('| [%s](%s/) | %s | %s | %s | %s |' %
                         (row['name'], relative(target, folder), row['seed'], row['global_batch'], row['H'], state))
            if group == '01_GPU_ONLY_BATCH300':
                for candidate in sorted((target / 'runs').glob('local_sgd_h40_fp32_2gpu_only_300_40e_seed*')):
                    lines += ['', '- [Mở trực tiếp %s](%s/).' % (candidate.name, relative(candidate, folder))]
            if group == '99_ARCHIVE':
                lines += ['', '## Các nhóm archive', '']
                for archived in sorted(target.iterdir()):
                    if archived.is_dir():
                        lines.append('- [%s](%s/)' % (archived.name, relative(archived, folder)))
            if group == '03_WORKLOAD_BALANCE_BATCH300':
                report = target / 'paired_diagnostic_report_vi.md'
                if report.exists():
                    lines += ['', '- [Báo cáo A/B seed42](%s)' % relative(report, folder)]
        for filename in REPORTS.get(group, []):
            target = repo / 'docs' / filename
            if target.exists():
                lines += ['', '- [Báo cáo: %s](%s)' % (filename, relative(target, folder))]
        lines += ['', 'Các thư mục run trong nhóm là dữ liệu thật. `summary.json/csv` là bảng tổng hợp;',
                  '`runs/` hoặc `results/` là output training; `ranks/` là telemetry đã collect.',
                  'Không cộng bản collect và output gốc thành hai runs độc lập.', '']
        (folder / 'README.md').write_text('\n'.join(lines))
    with (nav / 'catalog.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    now = datetime.now(timezone.utc).isoformat()
    (nav / 'catalog.json').write_text(json.dumps(dict(generated_utc=now, original_directories=len(rows),
        scope='Experiment bundles and standalone runs; archive is one historical container.', rows=rows),
        indent=2, ensure_ascii=False) + '\n')
    index = ['# Mục lục kết quả HeteroViT', '',
             'Dữ liệu thật đã được gom vào các nhóm dưới đây. Nhóm 01–04 là các phép so hiện tại.', '',
             '| Nhóm | Số thư mục experiment | Nội dung |', '|---|---:|---|']
    for group, (title, _) in GROUPS.items():
        count = sum(row['group'] == group for row in rows)
        if count:
            index.append('| [%s](%s/README.md) | %d | %s |' % (group, group, count, title))
    index += ['', '**Catalog %d thư mục:** [CSV](catalog.csv) · [JSON](catalog.json).' % len(rows), '',
              'Một sweep chứa nhiều runs: mở summary để lấy từng method/seed. Reruns, worker CSV và',
              'smoke không tăng số seeds. `historical_unverified` chưa xác minh completion, không phải failed.', '',
              '## File cần tìm trong mỗi run', '',
              '- `config.yaml` / `configs/`: recipe; `config_check.json`: cấu hình trước chạy.',
              '- `summary.csv/json`, `*_analysis.json`: kết quả và timing/readiness.',
              '- `train.csv`, `run.log`, `*.log`: metric từng epoch và log training.',
              '- `ranks/`: telemetry/clock từng rank; `checkpoints/`: weights; `source/`: snapshot.', '',
              '## Cập nhật sau khi có run mới', '', 'Từ `HeteroViT-MPI`:', '', '```bash',
              '../mpi_cluster/python_by_host.sh tools/index_results.py --organize', '```', '',
              '`relocations.json` lưu đường dẫn cũ → mới. Nội dung raw logs, config, checkpoint và source',
              'snapshots giữ nguyên; các đường dẫn bên trong artifacts lịch sử là provenance thời điểm chạy.',
              'Các công cụ hiện tại hỗ trợ tìm run theo cả layout cũ và mới.', '',
              'Generated UTC: ' + now, '']
    (nav / 'README.md').write_text('\n'.join(index))
    (root / 'README.md').write_text('\n'.join([
        '# Kết quả thực nghiệm HeteroViT', '',
        '**[MỞ CÁC NHÓM KẾT QUẢ →](%s/README.md)**' % NAV, '',
        '- [01 — GPU-only batch300](%s/01_GPU_ONLY_BATCH300/README.md)' % NAV,
        '- [02 — FP16 ba seeds chính](%s/02_FP16_BATCH300_PRIMARY/README.md)' % NAV,
        '- [03 — Workload A/B](%s/03_WORKLOAD_BALANCE_BATCH300/README.md)' % NAV,
        '- [04 — Batch600 GPU-only/5 máy](%s/04_BATCH600_H20/README.md)' % NAV, '',
        '[Catalog CSV](%s/catalog.csv) · [Catalog JSON](%s/catalog.json).' % (NAV, NAV), '',
        'Dữ liệu đã được gom thật vào nhóm; không có bản sao checkpoint/log.', '',
        'Cập nhật: `../mpi_cluster/python_by_host.sh tools/index_results.py --organize`.', '']))
    for row in rows:
        assert (repo / row['browse_path']).is_dir()
    print('INDEX:', nav / 'README.md')
    print('CATALOG:', nav / 'catalog.csv')
    print('Indexed %d experiment directories.' % len(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--organize', action='store_true', help='Move loose experiment directories into their groups.')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    with open('/tmp/heterovit_paired_workload_cluster.lock', 'a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('An experiment is active; wait until it finishes before organizing results.')
        if args.organize:
            organize(repo)
        build(repo)
