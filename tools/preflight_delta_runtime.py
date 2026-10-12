"""Read-only cluster runtime check; constructs an FP32 model on CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import socket
import json
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpi4py import MPI
import mpi4py

comm = MPI.COMM_WORLD
report = dict(rank=comm.rank, host=socket.gethostname(), python=sys.version.split()[0],
              executable=sys.executable, mpi4py=mpi4py.__version__,
              mpi_library=MPI.Get_library_version().strip())
try:
    import numpy as np
    import tensorflow as tf
    import yaml
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from src.models.vit import build_vit_from_config
    from src.training.base_trainer import get_optimizer
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    model = build_vit_from_config(config)
    model(tf.zeros([1, 32, 32, 3], dtype=tf.float32), training=False)
    optimizer = get_optimizer(config['training'])
    # Initialize optimizer slots without running training or altering files.
    optimizer.apply_gradients([(tf.zeros_like(v), v) for v in model.trainable_variables])
    layout = [(w.name, w.shape.as_list(), w.dtype.name) for w in model.weights]
    optimizer_variables = optimizer.variables()
    report.update(tensorflow=tf.__version__, numpy=np.__version__,
                  model_parameters=model.count_params(), model_layout=layout,
                  compute_dtype=model.dtype_policy.compute_dtype,
                  optimizer=type(optimizer).__module__ + '.' + type(optimizer).__name__,
                  optimizer_dtypes=sorted(set(v.dtype.name for v in optimizer_variables)),
                  dataset_files=sorted(os.listdir(config['dataset']['data_dir'])))
    report['ok'] = (all(w.dtype == tf.float32 for w in model.weights)
                    and model.dtype_policy.compute_dtype == 'float32'
                    and all(not v.dtype.is_floating or v.dtype == tf.float32 for v in optimizer_variables))
except Exception:
    report.update(ok=False, error=traceback.format_exc())
reports = comm.allgather(report)
layouts = [r.get('model_layout') for r in reports]
passed = all(r['ok'] for r in reports) and all(x == layouts[0] for x in layouts)
if comm.rank == 0:
    print(json.dumps(dict(passed=passed, ranks=reports), indent=2), flush=True)
if not passed:
    sys.exit(1)
