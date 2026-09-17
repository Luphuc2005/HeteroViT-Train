"""Main training entry point for Heterogeneous Vision Transformer benchmarks."""
import os
import sys
import argparse
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Heterogeneous ViT Training Benchmark (CPU-only and 1 V100 GPU)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (e.g., configs/cpu/cpu_12cores.yaml)",
    )
    return parser.parse_args()


def setup_xla_environment():
    """Locates and configures CUDA libdevice.10.bc for XLA JIT compilation if needed."""
    if "XLA_FLAGS" in os.environ and "--xla_gpu_cuda_data_dir" in os.environ["XLA_FLAGS"]:
        return

    import glob
    candidates = [
        "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
        "/usr/local/cuda-*/nvvm/libdevice/libdevice.10.bc",
        "/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc",
    ]
    for sp in sys.path:
        if "site-packages" in sp:
            candidates.append(os.path.join(sp, "nvidia", "**", "libdevice.10.bc"))

    for pattern in candidates:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            libdevice_path = matches[0]
            cuda_dir = os.path.dirname(os.path.dirname(os.path.dirname(libdevice_path)))
            if not os.path.exists(os.path.join(cuda_dir, "nvvm", "libdevice", "libdevice.10.bc")):
                cuda_dir = os.path.dirname(os.path.dirname(libdevice_path))
            existing_xla = os.environ.get("XLA_FLAGS", "")
            os.environ["XLA_FLAGS"] = f"{existing_xla} --xla_gpu_cuda_data_dir={cuda_dir}".strip()
            try:
                if not os.path.exists("./libdevice.10.bc"):
                    os.symlink(libdevice_path, "./libdevice.10.bc")
            except Exception:
                pass
            print(f"[Info] Configured XLA CUDA data dir: {cuda_dir}")
            break


def main():
    args = parse_args()
    config = load_config(args.config)

    mode = config.get("mode", "cpu").lower()

    # If CPU mode, ensure GPU visibility is disabled in environment if not already
    if mode == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif mode == "gpu":
        setup_xla_environment()

    # Initialize TensorFlow CPU threading if in CPU mode, if cpu config is present, or if env vars are set
    if mode == "cpu" or "cpu" in config or "TF_NUM_INTRAOP_THREADS" in os.environ:
        from src.training.trainer_cpu import configure_cpu_runtime
        configure_cpu_runtime(config.get("cpu", {}))

    # Set random seed early for reproducibility across all libraries
    seed = int(config.get("seed", 42))
    set_seed(seed)

    # Delay TensorFlow import after CUDA_VISIBLE_DEVICES and seed setup
    import tensorflow as tf
    from src.data.cifar10 import build_cifar10_datasets
    from src.models.vit import build_vit_from_config
    from src.metrics.logger import ExperimentLogger
    from src.training.trainer_cpu import CPUTrainer
    from src.training.trainer_gpu import GPUTrainer, MultiGPUTrainer, configure_gpu_runtime

    strategy = None
    if mode == "gpu":
        configure_gpu_runtime()
        gpu_cfg = config.get("gpu", {})
        num_gpus_cfg = int(gpu_cfg.get("num_gpus", 1))
        strategy_name = str(gpu_cfg.get("strategy", "")).lower()
        physical_gpus = tf.config.list_physical_devices("GPU")

        if (num_gpus_cfg > 1 or strategy_name == "mirrored") and len(physical_gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()

    # Initialize logger and experiment output directory
    logger = ExperimentLogger(config)
    logger.info(f"Loaded config from: {args.config}")
    logger.info(f"Execution Mode: {mode.upper()} | Seed: {seed}")

    # Load dataset
    dataset_cfg = config.get("dataset", {})
    data_dir = dataset_cfg.get("data_dir", "./data/cifar10")
    training_cfg = config.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 128))
    model_cfg = config.get("model", {})
    image_size = int(model_cfg.get("image_size", 32))

    logger.info(f"Loading CIFAR-10 dataset from: {data_dir} ...")
    try:
        train_ds, val_ds, test_ds, steps_per_epoch, val_steps = build_cifar10_datasets(
            data_dir=data_dir,
            batch_size=batch_size,
            val_split=0.1,
            image_size=image_size,
            seed=seed,
            cache=True,
        )
    except FileNotFoundError as err:
        logger.error(f"Dataset initialization failed: {err}")
        sys.exit(1)

    # Build model
    logger.info("Instantiating Vision Transformer model...")
    if strategy is not None:
        logger.info(f"Using tf.distribute.MirroredStrategy across {strategy.num_replicas_in_sync} GPUs.")
        with strategy.scope():
            model = build_vit_from_config(config)
            dummy_input = tf.zeros([1, image_size, image_size, 3], dtype=tf.float32)
            _ = model(dummy_input, training=False)
    else:
        model = build_vit_from_config(config)
        dummy_input = tf.zeros([1, image_size, image_size, 3], dtype=tf.float32)
        _ = model(dummy_input, training=False)

    num_params = model.count_params()
    logger.info(f"ViT Model '{model.name}' constructed successfully. Total trainable parameters: {num_params:,}")

    # Dispatch to appropriate trainer
    if mode == "cpu":
        trainer = CPUTrainer(
            model=model,
            config=config,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            steps_per_epoch=steps_per_epoch,
            val_steps=val_steps,
            logger=logger,
        )
    elif mode == "gpu":
        if strategy is not None:
            trainer = MultiGPUTrainer(
                strategy=strategy,
                model=model,
                config=config,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                steps_per_epoch=steps_per_epoch,
                val_steps=val_steps,
                logger=logger,
            )
        else:
            trainer = GPUTrainer(
                model=model,
                config=config,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                steps_per_epoch=steps_per_epoch,
                val_steps=val_steps,
                logger=logger,
            )
    else:
        raise ValueError(
            f"Unsupported mode: '{mode}'. Must be either 'cpu' or 'gpu'. "
            "(Hybrid CPU+GPU will be supported in future releases)"
        )

    # Execute training loop
    trainer.train()

    # Final evaluation on test dataset
    logger.info("Executing final evaluation on test set...")
    test_loss, test_acc = trainer.evaluate(test_ds, steps=int(10000 / batch_size) + 1)
    logger.info(f"Final Test Evaluation -> Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
