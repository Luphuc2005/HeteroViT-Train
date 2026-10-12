"""Verify the saved five-node delta smoke evidence and regenerate summary.json."""
import csv
import json
import math
from pathlib import Path

base = Path(__file__).resolve().parent
runtime = json.loads((base / 'runtime.json').read_text())
assert runtime['passed']
transport = (base / 'transport.log').read_text()
assert transport.count('Ran 12 tests') == 5 and transport.splitlines().count('OK') == 5
summary = dict(passed=True, transport_tests_per_rank=12, nodes=5, modes={})
fields = ('encode_ms', 'communication_ms', 'decode_ms', 'sync_total_ms',
          'relative_l2_quantization_error', 'delta_l2_norm', 'relative_l2_error')
for mode in ('allgather', 'allreduce'):
    state = json.loads((base / (mode + '_state.json')).read_text())
    assert state['passed'] and len(state['ranks']) == 5
    assert all(r['valid'] and r['optimizer_iterations'] == 120 for r in state['ranks'])
    assert len({r['model_sha256'] for r in state['ranks']}) == 1
    rank_metrics = []
    for rank in range(5):
        path = base / mode / ('rank%d' % rank) / ('weight_communication_rank%d.csv' % rank)
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 3
        for row in rows:
            assert row['method'] == 'fp16_delta_' + mode and row['collective'] == mode
            assert int(row['payload_bytes']) == 5387156
            assert all(math.isfinite(float(row[k])) and float(row[k]) >= 0 for k in fields)
            assert float(row['relative_l2_error']) < 0.01
        rank_metrics.append(dict(rank=rank, rounds=len(rows),
            max_relative_l2_quantization_error=max(float(r['relative_l2_quantization_error']) for r in rows),
            max_relative_l2_reference_error=max(float(r['relative_l2_error']) for r in rows)))
    timeline = list(csv.DictReader((base / mode / 'rank0/local_sgd_timeline.csv').open()))
    assert [int(r['step']) for r in timeline] == [40, 80, 120]
    assert all(float(r['max_weight_diff']) == 0 for r in timeline)
    for row in timeline:
        assert [int(row['rank%d_samples' % rank]) for rank in range(5)] == [10640, 240, 240, 240, 640]
        assert int(row['total_samples_since_sync']) == 12000
        assert math.isfinite(float(row['step_loss']))
    summary['modes'][mode] = dict(steps=120, synchronization_rounds=3, payload_bytes_per_rank=5387156,
        optimizer_iterations_on_every_rank=120, model_and_optimizer_finite_fp32=True,
        identical_global_model=True, max_weight_diff=0,
        max_relative_l2_quantization_error=max(r['max_relative_l2_quantization_error'] for r in rank_metrics),
        max_relative_l2_reference_error=max(r['max_relative_l2_reference_error'] for r in rank_metrics),
        rank_metrics=rank_metrics)
(base / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
