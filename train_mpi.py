"""Main MPI training entry point for Heterogeneous Vision Transformer on 5 nodes.

Topology:
  - Rank 0: iciplab01 (2x Titan Z GPUs via MirroredStrategy in single MPI rank)
  - Rank 1: iciplab02 (CPU)
  - Rank 2: iciplab03 (CPU)
  - Rank 3: iciplab04 (CPU)
  - Rank 4: iciplab05 (CPU)
"""
import os
import sys
import socket
import argparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -----------------------------------------------------------------------------
# 1. MPI Initialization and Early Device Environment Setup
# CRITICAL: CUDA_VISIBLE_DEVICES must be configured before importing TensorFlow
# or any module that imports TensorFlow!
# -----------------------------------------------------------------------------

#-> phân bổ thiết bị
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
hostname = socket.gethostname()

# Configure GPU visibility based on MPI rank
if rank == 0:
    # lab01 uses 2 Titan Z GPUs in rank 0
    if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
else:
    # ranks 1-4 are CPU-only
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Correction 4: Runtime hostname-rank validation
EXPECTED_RANKS = {
    0: ["iciplab01", "192.168.1.131", "localhost"],
    1: ["iciplab02", "192.168.1.132"],
    2: ["iciplab03", "192.168.1.133"],
    3: ["iciplab04", "192.168.1.134"],
    4: ["iciplab05", "192.168.1.135"],
}
expected_hosts = EXPECTED_RANKS.get(rank, [])
if expected_hosts and not any(h in hostname.lower() for h in expected_hosts) and world_size == 5:
    print(
        f"[WARNING] Rank {rank} running on '{hostname}', expected {expected_hosts}. "
        f"Verify mpi_cluster/hostfile order!",
        flush=True,
    )

# -----------------------------------------------------------------------------
# 2. Argument Parsing and Configuration Loading
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="HeteroViT Synchronous Distributed Training (5 Nodes MPI Baseline)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., configs/mpi/baseline_5nodes.yaml)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps before stopping (useful for smoke tests)",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume training from checkpoint. Pass path to weights file, or just --resume to auto-detect latest run.",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="Starting epoch number when resuming. Auto-detected from train.csv if omitted.",
    )
    return parser.parse_args()


args = parse_args()

from src.utils.config import load_config
from src.utils.seed import set_seed

config = load_config(args.config)
seed = int(config.get("seed", 42))
set_seed(seed)

# Resolve resume settings on rank 0 and broadcast to all ranks
resume_info = {
    "resume_dir": None,
    "resume_weights": None,
    "start_epoch": 1,
    "best_val_accuracy": 0.0,
}

if rank == 0 and args.resume:
    base_output_dir = os.environ.get("OUTPUT_DIR", config.get("logging", {}).get("output_dir", "./results"))
    target_weights = None
    target_dir = None

    if args.resume == "auto":
        if os.path.exists(base_output_dir):
            exp_prefix = config.get("experiment", {}).get("name", "mpi_baseline")
            candidate_dirs = [
                os.path.join(base_output_dir, d)
                for d in os.listdir(base_output_dir)
                if os.path.isdir(os.path.join(base_output_dir, d)) and d.startswith(exp_prefix)
            ]
            if candidate_dirs:
                candidate_dirs.sort(key=os.path.getmtime, reverse=True)
                for cdir in candidate_dirs:
                    wpath = os.path.join(cdir, "checkpoints", "last.weights.h5")
                    if os.path.exists(wpath):
                        target_dir = cdir
                        target_weights = wpath
                        break
    else:
        p = os.path.abspath(args.resume)
        if os.path.isdir(p):
            target_dir = p
            target_weights = os.path.join(p, "checkpoints", "last.weights.h5")
        else:
            target_weights = p
            target_dir = os.path.dirname(os.path.dirname(p))

    if target_weights and os.path.exists(target_weights):
        start_ep = 1
        best_acc = 0.0
        csv_path = os.path.join(target_dir, "train.csv") if target_dir else None
        if csv_path and os.path.exists(csv_path):
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    import csv
                    reader = list(csv.DictReader(f))
                    if reader:
                        epochs = [int(row["epoch"]) for row in reader if "epoch" in row and row["epoch"].isdigit()]
                        if epochs:
                            start_ep = max(epochs) + 1
                        val_accs = [float(row["val_accuracy"]) for row in reader if "val_accuracy" in row]
                        if val_accs:
                            best_acc = max(val_accs)
            except Exception as e:
                print(f"[Warning] Failed to parse existing train.csv: {e}")

        if args.start_epoch is not None:
            start_ep = int(args.start_epoch)

        resume_info["resume_dir"] = target_dir
        resume_info["resume_weights"] = target_weights
        resume_info["start_epoch"] = start_ep
        resume_info["best_val_accuracy"] = best_acc

resume_info = comm.bcast(resume_info, root=0)

# Configure CPU threading if in CPU mode or for ranks 1-4
if rank != 0 or "cpu" in config or "TF_NUM_INTRAOP_THREADS" in os.environ:
    from src.training.trainer_cpu import configure_cpu_runtime
    configure_cpu_runtime(config.get("cpu", {}))

# -----------------------------------------------------------------------------
# 3. TensorFlow and Project Imports (Delayed after CUDA_VISIBLE_DEVICES)
# -----------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
from src.data.cifar10 import build_cifar10_datasets
from src.models.vit import build_vit_from_config
from src.metrics.logger import ExperimentLogger
from src.training.trainer_mpi import MPITrainer

# Check physical devices
physical_gpus = tf.config.list_physical_devices("GPU")
num_gpus = len(physical_gpus)

strategy = None
if rank == 0:
    if num_gpus > 0:
        for gpu in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"[Warning] Failed setting memory growth on rank 0: {e}")
        if num_gpus > 1:
            strategy = tf.distribute.MirroredStrategy()

# Initialize Logger (rank 0 logs to file & console; worker ranks log warnings/errors to console)
logger = ExperimentLogger(config, resume_dir=resume_info["resume_dir"])
if rank != 0:
    import logging
    for h in logger.logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(logging.WARNING)

training_cfg = config.get("training", {})
rank_batch_sizes = training_cfg.get("rank_batch_sizes", None)
if rank_batch_sizes is not None:
    rank_batch_sizes = [int(b) for b in rank_batch_sizes]
    local_batch_size = rank_batch_sizes[rank]
    global_batch = sum(rank_batch_sizes)
else:
    local_batch_size = int(training_cfg.get("batch_size", 128))
    global_batch = local_batch_size * world_size

dataset_cfg = config.get("dataset", {})
data_dir = dataset_cfg.get("data_dir", "./data/cifar10")
model_cfg = config.get("model", {})
image_size = int(model_cfg.get("image_size", 32))
drop_remainder = bool(training_cfg.get("drop_remainder", True))
sync_tolerance = float(training_cfg.get("sync_tolerance", 1e-5))

# -----------------------------------------------------------------------------
# 4. Dataset Loading with MPI Sharding
# -----------------------------------------------------------------------------
try:
    train_ds, val_ds, test_ds, steps_per_epoch, val_steps = build_cifar10_datasets(
        data_dir=data_dir,
        batch_size=local_batch_size,
        val_split=0.1,
        image_size=image_size,
        seed=seed,
        cache=True,
        rank=rank,
        num_ranks=world_size,
        drop_remainder=drop_remainder,
        rank_batch_sizes=rank_batch_sizes,
    )
except FileNotFoundError as err:
    print(f"[rank {rank}] Dataset initialization failed: {err}", flush=True)
    sys.exit(1)

# -----------------------------------------------------------------------------
# 5. Cluster Topology Reporting
# -----------------------------------------------------------------------------
device_str = f"{num_gpus}xGPU" if num_gpus > 0 else "CPU"
shard_samples = steps_per_epoch * local_batch_size

grad_accum_cfg = training_cfg.get("grad_accum", {})
accum_info = ""
if rank == 0 and grad_accum_cfg.get("enabled", False):
    mb = int(grad_accum_cfg.get("micro_batch_size", 128))
    if local_batch_size > mb:
        steps = local_batch_size // mb
        accum_info = f" (grad_accum={steps}x{mb})"

node_info = {
    "rank": rank,
    "host": hostname,
    "device": device_str,
    "gpus": num_gpus,
    "batch": local_batch_size,
    "samples": shard_samples,
    "accum": accum_info,
}
all_node_info = comm.gather(node_info, root=0)

if rank == 0:
    print("=" * 80, flush=True)
    print(" HETEROVIT-MPI: 5-NODE SYNCHRONOUS DISTRIBUTED TRAINING", flush=True)
    print("=" * 80, flush=True)
    print("[CLUSTER TOPOLOGY]", flush=True)
    for info in sorted(all_node_info, key=lambda x: x["rank"]):
        r = info["rank"]
        h = info["host"]
        dev = info["device"]
        g = info["gpus"]
        b = info["batch"]
        s = info["samples"]
        acc = info.get("accum", "")
        print(
            f"  [rank {r}/{world_size}][{h}] device={dev:<5} "
            f"gpus_visible={g} local_batch={b:<3} shard_samples={s:,}{acc}",
            flush=True,
        )
    print("-" * 80, flush=True)
    batch_detail = f"sum({rank_batch_sizes})" if rank_batch_sizes else f"local_batch={local_batch_size} * {world_size} ranks"
    print(f"  Global Batch Size : {global_batch} ({batch_detail})", flush=True)
    print(f"  Steps per Epoch   : {steps_per_epoch} (drop_remainder={drop_remainder})", flush=True)
    print(f"  Sync Check Policy : Every {training_cfg.get('sync_interval', 20)} steps (tolerance: {sync_tolerance:.1e})", flush=True)
    print("=" * 80, flush=True)
    logger.info(f"Loaded config from: {args.config}")
    logger.info(f"Execution: MPI Distributed Baseline ({world_size} ranks) | Seed: {seed}")

comm.Barrier()

# -----------------------------------------------------------------------------
# 6. Model Instantiation & Initial Weight Broadcast
# -----------------------------------------------------------------------------
if rank == 0 and strategy is not None:
    logger.info(f"Rank 0 constructing ViT model across {strategy.num_replicas_in_sync} GPUs (MirroredStrategy)...")
    with strategy.scope():
        model = build_vit_from_config(config)
        dummy_input = tf.zeros([1, image_size, image_size, 3], dtype=tf.float32)
        _ = model(dummy_input, training=False)
        if resume_info["resume_weights"] and os.path.exists(resume_info["resume_weights"]):
            logger.info(f"Loading weights from checkpoint: {resume_info['resume_weights']}")
            model.load_weights(resume_info["resume_weights"])
            logger.info(f"Successfully loaded checkpoint! Resuming training from Epoch {resume_info['start_epoch']}...")
else:
    model = build_vit_from_config(config)
    dummy_input = tf.zeros([1, image_size, image_size, 3], dtype=tf.float32)
    _ = model(dummy_input, training=False)

if rank == 0:
    logger.info(f"Model '{model.name}' constructed. Total parameters: {model.count_params():,}")

# Broadcast complete model weights (model.get_weights())
if rank == 0:
    initial_weights = model.get_weights()
else:
    initial_weights = None

initial_weights = comm.bcast(initial_weights, root=0)

if rank != 0:
    model.set_weights(initial_weights)

# Synchronization verification at step 0 (initial weights)
flat_weights = np.concatenate([w.ravel() for w in model.get_weights()]).astype(np.float32)
max_buf = np.empty_like(flat_weights)
min_buf = np.empty_like(flat_weights)

comm.Barrier()
comm.Allreduce(flat_weights, max_buf, op=MPI.MAX)
comm.Allreduce(flat_weights, min_buf, op=MPI.MIN)
comm.Barrier()

initial_diff = float(np.max(np.abs(max_buf - min_buf)))
if rank == 0:
    print(f"[SYNC] step=0 weight_max_diff={initial_diff:.8e} [INITIAL WEIGHTS SYNCHRONIZED]", flush=True)

if initial_diff > sync_tolerance:
    msg = f"[SYNC FAILED] Initial weight broadcast discrepancy: {initial_diff:.8e} > {sync_tolerance}"
    if rank == 0:
        print(f"FAILED: {msg}", flush=True)
    raise RuntimeError(msg)

# -----------------------------------------------------------------------------
# 6. Execute MPI Distributed Training
# -----------------------------------------------------------------------------
trainer = MPITrainer(
    comm=comm,
    rank=rank,
    world_size=world_size,
    strategy=strategy,
    model=model,
    config=config,
    train_ds=train_ds,
    val_ds=val_ds,
    test_ds=test_ds,
    steps_per_epoch=steps_per_epoch,
    val_steps=val_steps,
    logger=logger,
    max_steps=args.max_steps,
    start_epoch=resume_info["start_epoch"],
    best_val_accuracy=resume_info["best_val_accuracy"],
)

trainer.train()

# Final evaluation (only if not a short smoke test)
if args.max_steps is None:
    if rank == 0:
        logger.info("Executing final evaluation on test set...")
    test_loss, test_acc = trainer.evaluate(test_ds, steps=int(10000 / local_batch_size) + 1)
    if rank == 0:
        logger.info(f"Final Test Evaluation -> Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%")

