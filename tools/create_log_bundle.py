"""Create a self-contained ZIP of local research logs, catalogs and source.

Excludes datasets, model checkpoints, VCS/auth directories and runtime caches.
Includes EF communication-state NPZ files and available historical source stages.
No upload or network operations are performed by this script.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import stat
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


LOG_EXTENSIONS = {'.log', '.out', '.stderr', '.csv', '.json'}
WEIGHTS = {'.h5', '.keras', '.ckpt', '.pt', '.pth', '.safetensors', '.tflite'}
SKIP_DIRS = {'.git', '.aws', '.codex', '.agents', '__pycache__', '.pytest_cache', '.cache'}
MAIN = {'fp16_sweep_uydYAoMC': 42, 'fp16_sweep_seed43_RIHAAkW2': 43,
        'fp16_sweep_seed44_swTaOOce': 44}
PAIR = 'fp16_sweep_seed43_2tyCNQ2H'
CLASSIFICATION = {
    'fp16_sweep_83LF5CdI': ('obsolete_evaluation', 'Sweep seed42 cũ trước cập nhật evaluation; không dùng bảng chính.'),
    'fp16_sweep_seed43_2yEHiooV': ('seed43_rerun', 'Đối chứng chạy lại FP32/Delta AG seed43; không phải seed độc lập.'),
    'fp16_sweep_seed43_5jdeOPhh': ('smoke_120_steps', 'Smoke EF/control 120 steps; không dùng làm kết quả full training.'),
    'fp16_sweep_seed43_E2dtk5hD': ('diagnostic_360_steps', 'Shadow same-input diagnostic, trajectory FP32; không công bố speedup/accuracy của shadow.'),
    'fp16_sweep_seed43_AMbc0guE': ('failed_launcher', 'Launcher thất bại, status failed exit=255; không có train.csv.'),
}


def csv_data(records, fields=None):
    fields = fields or list(records[0])
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    return out.getvalue().encode('utf-8-sig')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def classify(relative):
    parts = Path(relative).parts
    if len(parts) > 2 and parts[:2] == ('HeteroViT-MPI', 'results'):
        name = parts[2]
        if name in MAIN:
            return 'primary_three_seed'
        if name == PAIR:
            return 'paired_ef_seed43'
        if name in CLASSIFICATION:
            return CLASSIFICATION[name][0]
        if name == 'archive':
            return 'archive_' + (parts[3] if len(parts) > 3 else 'other')
        return 'legacy_or_exploratory_results'
    if len(parts) > 1 and parts[0] == 'HeteroViT-MPI':
        return 'repo_' + parts[1] if len(parts) > 2 else 'repo_root'
    return parts[0] if len(parts) > 1 else 'workspace_root'


def inventory(workspace):
    included, excluded = [], []
    for current, dirs, files in os.walk(str(workspace), followlinks=False):
        current = Path(current)
        kept_dirs = []
        for name in sorted(dirs):
            path = current / name
            relative = path.relative_to(workspace).as_posix()
            reason = None
            if name in SKIP_DIRS:
                reason = 'VCS/auth/session/cache directory; tree excluded'
            elif relative == 'deliverables':
                reason = 'Bundle output directory; avoid recursive inclusion'
            elif relative == 'HeteroViT-MPI/data':
                reason = 'Training dataset; not experiment logs'
            elif path.is_symlink():
                reason = 'Directory symlink; not traversed'
            if reason:
                excluded.append(dict(source=relative, item_type='directory_tree', bytes='', reason=reason))
            else:
                kept_dirs.append(name)
        dirs[:] = kept_dirs
        for name in sorted(files):
            path = current / name
            relative = path.relative_to(workspace).as_posix()
            reason = None
            if path.is_symlink():
                try:
                    resolved = path.resolve(strict=True)
                    resolved.relative_to(workspace)
                except (ValueError, FileNotFoundError, RuntimeError):
                    reason = 'External or broken symlink; runtime dependency, not log'
            if path.suffix in WEIGHTS:
                reason = 'Model checkpoint/weights; log bundle excludes weights'
            elif path.suffix in {'.pyc', '.pyo', '.so', '.bc'}:
                reason = 'Compiled cache/runtime binary; not log'
            if reason:
                excluded.append(dict(source=relative, item_type='file', bytes=path.lstat().st_size, reason=reason))
            elif path.is_file():
                included.append((path, 'workspace/' + relative, classify(relative)))
    return sorted(included, key=lambda x: x[1]), excluded


def source_snapshots(workspace):
    entries, index = [], []
    repo = workspace / 'HeteroViT-MPI'
    for marker in sorted((repo / 'results').glob('fp16_sweep*/source_stage.txt')):
        source = Path(marker.read_text().strip())
        if not source.is_absolute() or source.parent != Path('/tmp'):
            raise ValueError('Unexpected source stage: ' + str(source))
        label = source.name
        count = 0
        if source.is_dir():
            for path in sorted(source.rglob('*')):
                relative = path.relative_to(source)
                if (not path.is_file() or path.is_symlink()
                        or any(p in SKIP_DIRS for p in relative.parts)
                        or path.suffix in {'.pyc', '.pyo', '.so', '.bc'}):
                    continue
                if relative.parts[0] not in {'src', 'tools', 'configs', 'scripts', 'tests'} and len(relative.parts) != 1:
                    continue
                if len(relative.parts) == 1 and path.suffix not in {'.py', '.sh', '.md', '.txt', '.yaml', '.json'}:
                    continue
                entries.append((path, 'source_snapshots/' + label + '/' + relative.as_posix(), 'historical_source_snapshots'))
                count += 1
        index.append(dict(sweep=marker.parent.name, original_source_stage=str(source),
                          available=source.is_dir(), included_files=count,
                          bundle_directory='source_snapshots/' + label))
    return entries, index


def run_catalog(workspace):
    root = workspace / 'HeteroViT-MPI/results'
    locations, catalog = [], []
    for path in sorted(root.rglob('train.csv')):
        relative = path.relative_to(workspace).as_posix()
        parts = path.relative_to(root).parts
        role, method = 'standalone_run_directory', path.parent.name
        if parts[0].startswith('fp16_sweep'):
            if 'ranks' in parts:
                i = parts.index('ranks')
                method, rank = parts[i + 1:i + 3]
                role = 'canonical_rank0' if rank == 'rank0' else 'worker_telemetry'
            else:
                role = 'original_output_copy'
        with path.open(newline='') as handle:
            values = list(csv.DictReader(handle))
        epoch_numbers = []
        for value in values:
            try:
                epoch_numbers.append(int(float(value.get('epoch', ''))))
            except (ValueError, TypeError):
                pass
        metadata_path = path.parent / 'metadata.json'
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        config_path = path.parent / 'config.yaml'
        config = config_path.read_text() if config_path.exists() else ''
        seed = metadata.get('seed', '')
        if seed is None or seed == '':
            match = re.search(r'^seed:\s*(\d+)\s*$', config, re.M)
            seed = match.group(1) if match else ''
        log_path = path.parent / 'run.log'
        log = log_path.read_text(errors='replace') if log_path.exists() else ''
        summary = {}
        summary_path = root / parts[0] / 'summary.csv'
        if role == 'canonical_rank0' and summary_path.exists():
            with summary_path.open(newline='') as handle:
                summary = next((r for r in csv.DictReader(handle) if r['method'] == method), {})
        if parts[0] in MAIN:
            note = 'Bộ bảng chính: một sweep đủ năm methods/seed.'
        elif parts[0] == PAIR:
            note = 'Paired EF seed43 riêng; không thêm vào mean ba seeds hoặc coi là seed mới.'
        elif parts[0] in CLASSIFICATION:
            note = CLASSIFICATION[parts[0]][1]
        elif 'archive' in parts:
            note = 'Run lịch sử; recipe/evaluation có thể khác; chưa audit như bộ 17 runs chính.'
        else:
            note = 'Run khác/đối chứng cũ; chưa audit như bộ 17 runs chính; không tự gộp vào FP16.'
        entry = dict(train_csv='workspace/' + relative, storage_role=role,
                     category=classify(relative), seed=seed, observed_epoch_rows=len(values),
                     largest_epoch_in_csv=max(epoch_numbers) if epoch_numbers else '',
                     method_or_directory=method, summary_syncs=summary.get('synchronization_rounds', ''),
                     summary_test_accuracy_percent=summary.get('test_accuracy_percent', ''),
                     summary_training_seconds=summary.get('training_time_seconds', ''),
                     completion_marker_present=bool(re.search(r'Training finished|Training completed|training completed', log)),
                     traceback_present='Traceback (most recent call last)' in log,
                     note=note)
        locations.append(entry)
        if role not in {'worker_telemetry', 'original_output_copy'}:
            catalog.append(entry.copy())
    # Failed launchers without a train.csv remain visible in a separate list.
    failed = []
    for path in sorted(root.glob('fp16_sweep*/status.txt')):
        status = path.read_text().strip()
        if status.startswith('failed'):
            failed.append(dict(directory='workspace/' + path.parent.relative_to(workspace).as_posix(), status=status))
    return catalog, locations, failed


def build(workspace, output_dir):
    ict = timezone(timedelta(hours=7))
    now = datetime.now(timezone.utc)
    name = 'HeteroViT_all_logs_' + now.astimezone(ict).strftime('%Y%m%d_%H%M%S_ICT')
    bundle_dir = output_dir / name
    bundle_dir.mkdir(parents=True, exist_ok=False)
    archive = output_dir / (name + '.zip')
    entries, excluded = inventory(workspace)
    stages, stage_index = source_snapshots(workspace)
    entries.extend(stages)
    catalog, locations, failed = run_catalog(workspace)
    evidence_path = workspace / 'HeteroViT-MPI/docs/bottleneck_audit_20261018.json'
    evidence = json.loads(evidence_path.read_text())
    primary = [dict(seed=r['seed'], role='paired_ef' if r['paired'] else 'primary_three_seed',
                    method=r['method'], test_accuracy_percent=r['test_accuracy_percent'],
                    training_loop_seconds=r['training_loop_seconds'],
                    log='workspace/HeteroViT-MPI/' + r['log'],
                    train_csv='workspace/HeteroViT-MPI/' + r['train_csv']) for r in evidence['runs']]
    manifest, groups, hashes = [], defaultdict(lambda: dict(files=0, bytes=0, log_metric_files=0)), defaultdict(list)
    total = len(entries)
    with zipfile.ZipFile(str(archive), 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
        for i, (path, arcname, group) in enumerate(sorted(entries, key=lambda e: e[1])):
            before = path.stat()
            checksum = hashlib.sha256()
            info = zipfile.ZipInfo(name + '/' + arcname, now.astimezone(ict).timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            with path.open('rb') as source, zf.open(info, 'w', force_zip64=True) as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    checksum.update(chunk)
                    target.write(chunk)
            after = path.stat()
            if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                raise RuntimeError('Source changed while archiving: ' + str(path))
            result = dict(archive_path=arcname, source_path=str(path.relative_to(workspace))
                          if arcname.startswith('workspace/') else str(path),
                          category=group, bytes=before.st_size, sha256=checksum.hexdigest(),
                          source_mtime_utc=datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat(),
                          symlink_target=os.readlink(str(path)) if path.is_symlink() else '')
            manifest.append(result)
            groups[group]['files'] += 1
            groups[group]['bytes'] += before.st_size
            groups[group]['log_metric_files'] += path.suffix in LOG_EXTENSIONS
            hashes[(before.st_size, result['sha256'])].append(arcname)
            if (i + 1) % 300 == 0:
                print('Archived %d/%d source files' % (i + 1, total), flush=True)
        included_names = {r['archive_path'] for r in manifest}
        expected_log_paths = {arc for p, arc, _ in entries if arc.startswith('workspace/') and p.suffix in LOG_EXTENSIONS}
        if not expected_log_paths.issubset(included_names):
            raise RuntimeError('Incomplete log coverage')
        roles = Counter(r['storage_role'] for r in locations)
        suffixes = Counter(Path(r['archive_path']).suffix for r in manifest if r['archive_path'].startswith('workspace/'))
        summary = dict(bundle=name, snapshot_time_utc=now.isoformat(), snapshot_time_vietnam=now.astimezone(ict).isoformat(),
                       source_workspace=str(workspace), source_files=len(manifest),
                       source_bytes=sum(r['bytes'] for r in manifest),
                       workspace_source_files=sum(r['archive_path'].startswith('workspace/') for r in manifest),
                       historical_source_files=sum(r['archive_path'].startswith('source_snapshots/') for r in manifest),
                       workspace_log_metric_files=len(expected_log_paths), workspace_extensions=dict(suffixes),
                       train_csv_locations=len(locations), train_csv_storage_roles=dict(roles),
                       catalog_run_directories=len(catalog),
                       catalog_note='Storage directories, not independent trials; old/incomplete/rerun entries retained.',
                       primary_full_runs=len(primary), primary_three_seed_runs=15, paired_ef_runs=2,
                       failed_sweeps=failed, excluded_file_entries=sum(r['item_type']=='file' for r in excluded),
                       excluded_directory_trees=[r['source'] for r in excluded if r['item_type']=='directory_tree'],
                       scope='All eligible files currently in local workspace plus available referenced source snapshots; no SSH collection.',
                       source_files_unchanged_during_copy=True, log_coverage_check_passed=True,
                       drive_folder_id='1eJvGHuR0ZNwNp10m5WI74spHelMeO_X5', upload_status='Prepared locally; upload tracked outside immutable archive')
        group_rows = [dict(category=k, **v) for k, v in sorted(groups.items())]
        duplicate_rows = []
        for (size, sha), paths in sorted(hashes.items()):
            if len(paths) > 1:
                for path in paths:
                    duplicate_rows.append(dict(sha256=sha, bytes=size, same_content_copies=len(paths), archive_path=path))
        files = {
            '01_INVENTORY.json': json.dumps(summary, ensure_ascii=False, indent=2).encode('utf-8'),
            '01_INVENTORY_BY_GROUP.csv': csv_data(group_rows),
            '03_RUN_CATALOG.csv': csv_data(catalog),
            '04_PRIMARY_17_RUNS.csv': csv_data(primary),
            '05_TRAIN_CSV_LOCATIONS.csv': csv_data(locations),
            '06_FILE_MANIFEST_SHA256.csv': csv_data(manifest),
            '07_EXCLUDED_FILES.csv': csv_data(excluded),
            '08_DUPLICATE_CONTENT.csv': csv_data(duplicate_rows),
            '09_SOURCE_SNAPSHOTS.csv': csv_data(stage_index),
        }
        paper = workspace / 'HeteroViT-MPI/docs/paper_story_bundle_vi.md'
        files['02_PAPER_STORY_VI.md'] = paper.read_bytes()
        files['00_README_VI.md'] = readme(summary, group_rows).encode('utf-8')
        files['VERIFY_BUNDLE.py'] = VERIFY.encode('utf-8')
        for filename, contents in files.items():
            (bundle_dir / filename).write_bytes(contents)
            zf.writestr(name + '/' + filename, contents)
    print('Checking ZIP CRC for all entries...', flush=True)
    with zipfile.ZipFile(str(archive)) as zf:
        error = zf.testzip()
        if error:
            raise RuntimeError('ZIP CRC failure: ' + error)
        names = zf.namelist()
        if len(names) != len(set(names)) or len(names) != len(manifest) + len(files):
            raise RuntimeError('ZIP entry coverage or duplicate-path failure')
    sha = digest(archive)
    check = dict(zip_file=str(archive), bytes=archive.stat().st_size, sha256=sha,
                 entries=len(names), source_entries=len(manifest), generated_metadata_entries=len(files),
                 crc_check='PASS', source_manifest_coverage='PASS', missing_source_entries=0,
                 upload_status='not_uploaded', target_drive_folder_id=summary['drive_folder_id'])
    receipt = archive.with_suffix('.verification.json')
    receipt.write_text(json.dumps(check, ensure_ascii=False, indent=2) + '\n')
    archive.with_suffix('.zip.sha256').write_text(sha + '  ' + archive.name + '\n')
    print(json.dumps(check, ensure_ascii=False, indent=2), flush=True)
    return archive


def readme(s, groups):
    table = '\n'.join('| %s | %d | %d | %.2f |' % (g['category'], g['files'], g['log_metric_files'], g['bytes']/1000000)
                      for g in groups)
    return '''# Gói log HeteroViT — đọc file này trước

Thời điểm snapshot (Việt Nam): **{time}**. Giữ nguyên đường dẫn log trong `workspace/`.

## Số lượng chính xác

- **{files} file nguồn**, tổng {mb:.2f} MB trước nén: {workspace} từ workspace và {stages} từ source snapshots.
- **{logs} file log/metric/evidence** trong workspace có đuôi `.log`, `.out`, `.stderr`, `.csv`, `.json`; YAML, README, ảnh, source và EF NPZ được tính riêng trong inventory.
- **{trains} vị trí train.csv**, gồm {rank0} canonical rank0 của sweep, {workers} worker telemetry, {copies} bản output gốc, {other} thư mục run cũ/khác.
- **{catalog} thư mục run trong catalog**, không có nghĩa là {catalog} thí nghiệm độc lập: có run cũ, chưa xong, lặp seed và smoke.
- Bộ bằng chứng chính đã audit: **15 full runs ba seeds + 2 full runs paired EF = 17**; 85 rank traces, mỗi trace 6.000 steps/160 syncs.
- Giữ cả launcher thất bại trong raw logs. Danh sách failed sweeps nằm trong `01_INVENTORY.json`.

## Đọc theo thứ tự

1. `02_PAPER_STORY_VI.md`: câu chuyện paper, số liệu có thể claim và giới hạn.
2. `04_PRIMARY_17_RUNS.csv`: chỉ các runs đã audit dùng cho kết quả hiện tại.
3. `workspace/HeteroViT-MPI/docs/bottleneck_story_20261018_vi.md`: phân tích đầy đủ từng rank và đường đo thời gian.
4. `03_RUN_CATALOG.csv`: mỗi canonical rank0/standalone run directory một dòng; đọc category/note trước khi dùng.
5. `05_TRAIN_CSV_LOCATIONS.csv`: toàn bộ vị trí train.csv, kể cả worker và bản sao.
6. `06_FILE_MANIFEST_SHA256.csv`: từng file nguồn, bytes, SHA256, category, đường dẫn.
7. `08_DUPLICATE_CONTENT.csv`: file trùng nội dung, vẫn giữ tất cả đường dẫn. Trùng file không tự chứng minh hai run giống nhau.
8. `07_EXCLUDED_FILES.csv`: file/tree ngoài phạm vi log, có lý do; checkpoint/dataset không nằm trong ZIP.
9. `09_SOURCE_SNAPSHOTS.csv`: ánh xạ source_stage.txt tuyệt đối sang source snapshot đã đóng gói.

## Quy tắc để không nhầm

- Bảng ba seeds chỉ dùng seed42 `uydYAoMC`, seed43 `RIHAAkW2`, seed44 `swTaOOce`, mỗi sweep năm methods.
- `83LF5CdI` là sweep cũ trước cập nhật evaluation; giữ lịch sử nhưng không gộp vào bảng chính.
- `2tyCNQ2H` là paired seed43 có/không EF. Dùng control cùng đợt khi so thời gian EF; không coi là seed mới.
- `2yEHiooV` là lần chạy lại seed43 FP32/Delta AG; không coi là seed độc lập.
- `5jdeOPhh` là smoke 120 steps, `E2dtk5hD` là diagnostic 360 steps/shadow FP32. Chúng không phải full training accuracy benchmark.
- `AMbc0guE` có status failed exit=255, không có train.csv; không được tính là run thành công.
- `ranks/method/rank0` là bản canonical dùng audit; `results/method_timestamp` trong cùng sweep thường là output gốc đã được collect lại. Giữ cả hai, không cộng số lần chạy.
- CSV Whole FP16 cũ có 320 dòng cho 160 syncs, header thiếu total_sync_ms. Dùng `communication_rows` trong tools/summarize_fp16_sweep.py để tránh đếm đôi.
- Train TB trên slide là thời gian chạy đủ 40 epochs, gồm validation/overhead trong loop. Không đồng nhất với time-to-target hay thời gian launcher.
- Payload/rank/sync là một buffer; không phải tổng traffic MPI. Không cộng maxima của các ranks khác nhau.
- Tài liệu lịch sử có thể có claim/đơn vị cũ (ví dụ 1Gbps); dùng báo cáo audit hiện tại và raw profiles để xác nhận số liệu.

## Phạm vi và tính đầy đủ

Gói lấy toàn bộ file đủ điều kiện hiện có trong workspace này, kể cả logs/, HeteroViT-MPI/logs/, results/archive/, validation/, profiles/, docs/, reports/, plots/, source/config/scripts và mpi_cluster/. Thêm source snapshots /tmp được tham chiếu bởi chín sweep khi còn tồn tại. Không SSH lấy thêm log ở máy khác và không có nghĩa toàn bộ lịch sử cụm đều đã được collect về máy này.

Dataset, model weights/checkpoints, Git/auth/session directories, caches và runtime binaries không phải log nên loại theo danh mục. **EF NPZ communication states được giữ**; chúng không đủ để resume optimizer/RNG/dataloader. Các symlink config trỏ bên trong workspace được đóng thành file thường để dễ giải nén; external CUDA symlink bị loại.

12 file hướng dẫn/catalog/manifest/verifier ở gốc ZIP là metadata sinh thêm, không cộng vào số file nguồn. File SHA256 và biên nhận kiểm tra ZIP nằm cạnh ZIP. Sau giải nén, chạy `python VERIFY_BUNDLE.py` tại thư mục gốc gói để kiểm tra SHA256 toàn bộ file nguồn. Bản source hiện tại và snapshot cũ là bằng chứng code, không phải môi trường CUDA/MPI hoàn chỉnh để tái chạy mọi run lịch sử.

## Tổng hợp theo nhóm

| Nhóm | Files | Log/metric files | MB trước nén |
|---|---:|---:|---:|
{table}
'''.format(time=s['snapshot_time_vietnam'], files=s['source_files'], mb=s['source_bytes']/1000000,
           workspace=s['workspace_source_files'], stages=s['historical_source_files'], logs=s['workspace_log_metric_files'],
           trains=s['train_csv_locations'], rank0=s['train_csv_storage_roles'].get('canonical_rank0',0),
           workers=s['train_csv_storage_roles'].get('worker_telemetry',0), copies=s['train_csv_storage_roles'].get('original_output_copy',0),
           other=s['train_csv_storage_roles'].get('standalone_run_directory',0), catalog=s['catalog_run_directories'], table=table)


VERIFY = '''"""Verify extracted source files against 06_FILE_MANIFEST_SHA256.csv."""
import csv
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
count = 0
with (root / '06_FILE_MANIFEST_SHA256.csv').open(encoding='utf-8-sig', newline='') as handle:
    for row in csv.DictReader(handle):
        relative = Path(row['archive_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe path in manifest')
        path = root / relative
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        if path.stat().st_size != int(row['bytes']) or digest.hexdigest() != row['sha256']:
            raise ValueError('Checksum mismatch: ' + str(relative))
        count += 1
print('PASS: verified %d source files' % count)
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output_dir or workspace / 'deliverables'
    output.mkdir(parents=True, exist_ok=True)
    build(workspace, output)
