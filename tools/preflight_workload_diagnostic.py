"""Read-only five-node runtime/source/dataset preflight, outside timed training."""
import argparse
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' if rank == 0 else ''
import tensorflow as tf
import yaml

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', type=Path, default=root / 'configs/mpi/paired_A_control.yaml')
args = parser.parse_args()
config = yaml.safe_load(args.config.read_text())
data = Path(config['dataset']['data_dir'])
source_files = ['train_mpi.py', 'src/training/trainer_mpi.py', 'src/training/base_trainer.py',
                'src/data/cifar10.py', 'src/data/augmentation.py', 'src/training/round_diagnostic.py']
source_hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in source_files}
gpu_count = len(tf.config.list_physical_devices('GPU'))
record = dict(rank=rank, host=socket.gethostname(), python=sys.version, tensorflow=tf.__version__,
              gpu_count=gpu_count, data_exists=data.is_dir(), source_sha256=source_hashes,
              mpi_library=MPI.Get_library_version())
records = comm.gather(record, root=0)
passed = None
if rank == 0:
    expected = json.loads((root / 'cluster/expected_baseline_runtime.json').read_text())
    passed = (comm.Get_size() == 5 and records[0]['gpu_count'] == 2
              and all(r['gpu_count'] == 0 for r in records[1:])
              and all(r['data_exists'] for r in records)
              and all(r['source_sha256'] == source_hashes for r in records)
              and all(r['tensorflow'] == e['tensorflow'] and r['host'] == e['host']
                      and r['python'].startswith(e['python_version'])
                      for r, e in zip(records, expected)))
    print(json.dumps(dict(passed=passed, ranks=records, expected_per_rank_baseline=expected), indent=2), flush=True)
passed = comm.bcast(passed, root=0)
if not passed:
    raise RuntimeError('Paired workload runtime/source preflight failed')
