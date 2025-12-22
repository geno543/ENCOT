"""
Training entry point for ColiFormer.

This script wraps finetune.py and loads configuration from YAML files.

Usage:
    python scripts/train.py --config configs/train_ecoli_alm.yaml
    python scripts/train.py --config configs/train_ecoli_quick.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import finetune
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with configuration values
    """
    # Lazy import so `python scripts/train.py --help` works without dependencies installed.
    import yaml

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def config_to_args(config: dict) -> argparse.Namespace:
    """
    Convert config dictionary to argparse.Namespace compatible with finetune.py.

    Args:
        config: Configuration dictionary from YAML

    Returns:
        argparse.Namespace with all required arguments
    """
    # Extract nested config values
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    checkpoint_config = config.get('checkpoint', {})
    alm_config = config.get('alm', {})
    gc_penalty_config = config.get('gc_penalty', {})

    # Build args namespace
    args = argparse.Namespace()

    # Data paths
    args.dataset_dir = data_config.get('dataset_dir', 'data')

    # Checkpoint paths
    args.checkpoint_dir = checkpoint_config.get('checkpoint_dir', 'models/checkpoints')
    args.checkpoint_filename = checkpoint_config.get('checkpoint_filename', 'finetune.ckpt')

    # Training parameters
    args.batch_size = training_config.get('batch_size', 6)
    args.max_epochs = training_config.get('max_epochs', 15)
    args.num_workers = training_config.get('num_workers', 5)
    args.accumulate_grad_batches = training_config.get('accumulate_grad_batches', 1)
    args.num_gpus = training_config.get('num_gpus', 4)
    args.learning_rate = training_config.get('learning_rate', 5e-5)
    args.warmup_fraction = training_config.get('warmup_fraction', 0.1)
    args.save_every_n_steps = training_config.get('save_every_n_steps', 512)
    args.seed = training_config.get('seed', 123)
    args.log_every_n_steps = training_config.get('log_every_n_steps', 20)
    args.debug = training_config.get('debug', False)

    # GC penalty (legacy)
    args.gc_penalty_weight = gc_penalty_config.get('weight', 0.0)

    # ALM parameters
    args.use_lagrangian = alm_config.get('enabled', False)
    args.gc_target = alm_config.get('gc_target', 0.52)
    args.curriculum_epochs = alm_config.get('curriculum_epochs', 3)
    args.lagrangian_rho = alm_config.get('initial_penalty_factor', 20.0)  # Use initial_penalty_factor as rho
    args.alm_tolerance = alm_config.get('tolerance', 1e-5)
    args.alm_dual_tolerance = alm_config.get('dual_tolerance', 1e-5)
    args.alm_penalty_update_factor = alm_config.get('penalty_update_factor', 10.0)
    args.alm_initial_penalty_factor = alm_config.get('initial_penalty_factor', 20.0)
    args.alm_tolerance_update_factor = alm_config.get('tolerance_update_factor', 0.1)
    args.alm_rel_penalty_increase_threshold = alm_config.get('rel_penalty_increase_threshold', 0.1)
    args.alm_max_penalty = alm_config.get('max_penalty', 1e6)
    args.alm_min_penalty = alm_config.get('min_penalty', 1e-6)

    return args


def validate_config(config: dict):
    """
    Validate configuration before training.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    data_config = config.get('data', {})
    dataset_dir = data_config.get('dataset_dir', 'data')

    # Check dataset directory exists
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    # Check for expected data files
    finetune_set = os.path.join(dataset_dir, 'finetune_set.json')
    if not os.path.exists(finetune_set):
        raise ValueError(
            f"Training data not found: {finetune_set}\n"
            "Please run data preprocessing first:\n"
            "  python scripts/preprocess_data.py"
        )

    # Validate checkpoint directory can be created
    checkpoint_config = config.get('checkpoint', {})
    checkpoint_dir = checkpoint_config.get('checkpoint_dir', 'models/checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train ColiFormer model with configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with main ALM configuration
    python scripts/train.py --config configs/train_ecoli_alm.yaml

    # Quick test training (CPU, 1 epoch)
    python scripts/train.py --config configs/train_ecoli_quick.yaml

    # Override config values from command line
    python scripts/train.py --config configs/train_ecoli_alm.yaml --num_gpus 2 --batch_size 4
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Override number of GPUs from config"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Override max epochs from config"
    )

    args = parser.parse_args()

    try:
        # Lazy import so `--help` works even if training deps are missing.
        from finetune import main as finetune_main

        # Load configuration
        print(f"Loading configuration from {args.config}...")
        config = load_config(args.config)

        # Override with command-line arguments if provided
        if args.num_gpus is not None:
            config.setdefault('training', {})['num_gpus'] = args.num_gpus
        if args.batch_size is not None:
            config.setdefault('training', {})['batch_size'] = args.batch_size
        if args.max_epochs is not None:
            config.setdefault('training', {})['max_epochs'] = args.max_epochs

        # Validate configuration
        print("Validating configuration...")
        validate_config(config)

        # Convert config to args namespace
        train_args = config_to_args(config)

        # Print training summary
        print("\n" + "="*60)
        print("Training Configuration Summary")
        print("="*60)
        print(f"Dataset directory: {train_args.dataset_dir}")
        print(f"Checkpoint directory: {train_args.checkpoint_dir}")
        print(f"Checkpoint filename: {train_args.checkpoint_filename}")
        print(f"Batch size: {train_args.batch_size}")
        print(f"Max epochs: {train_args.max_epochs}")
        print(f"Learning rate: {train_args.learning_rate}")
        print(f"Number of GPUs: {train_args.num_gpus}")
        print(f"ALM enabled: {train_args.use_lagrangian}")
        if train_args.use_lagrangian:
            print(f"GC target: {train_args.gc_target}")
            print(f"Curriculum epochs: {train_args.curriculum_epochs}")
        print("="*60 + "\n")

        # Run training
        finetune_main(train_args)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

