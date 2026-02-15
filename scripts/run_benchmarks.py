"""
Run benchmark evaluation for ColiFormer.

This script wraps benchmark_evaluation.py and evaluate_optimizer.py to provide
a unified interface for running comprehensive evaluations.

Usage:
    python scripts/run_benchmarks.py --config configs/benchmark.yaml
    python scripts/run_benchmarks.py --excel_path Benchmark_80_sequences.xlsx --checkpoint_path models/my_model.ckpt
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import benchmark scripts
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with configuration values
    """
    # Lazy import so `python scripts/run_benchmarks.py --help` works without dependencies installed.
    import yaml

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def config_to_args(config: dict) -> argparse.Namespace:
    """
    Convert config dictionary to argparse.Namespace compatible with benchmark_evaluation.py.

    Args:
        config: Configuration dictionary from YAML

    Returns:
        argparse.Namespace with all required arguments
    """
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    output_config = config.get('output', {})
    eval_config = config.get('evaluation', {})

    args = argparse.Namespace()

    # Model paths
    args.checkpoint_path = model_config.get('checkpoint_path', 'models/alm-enhanced-training/balanced_alm_finetune.ckpt')

    # Data paths
    args.excel_path = data_config.get('excel_path', 'Benchmark 80 sequences.xlsx')
    args.natural_sequences_path = data_config.get('natural_sequences_path', 'data/ecoli_processed_genes.csv')
    args.name_col = data_config.get('name_col')
    args.seq_col = data_config.get('seq_col')
    args.sheet_name = data_config.get('sheet_name')

    # Output paths
    args.output_dir = output_config.get('output_dir', 'benchmark_results')

    # Evaluation parameters
    args.use_gpu = eval_config.get('use_gpu', True)
    args.compare_with_base = eval_config.get('compare_with_base', False)
    args.max_test_proteins = eval_config.get('max_test_proteins', 0)

    return args


def validate_config(config: dict):
    """
    Validate configuration before running benchmarks.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    data_config = config.get('data', {})
    excel_path = data_config.get('excel_path', 'Benchmark 80 sequences.xlsx')

    if not os.path.exists(excel_path):
        raise ValueError(
            f"Benchmark Excel file not found: {excel_path}\n"
            "Please provide a valid path to your benchmark sequences file."
        )

    model_config = config.get('model', {})
    checkpoint_path = model_config.get('checkpoint_path')

    # Check if checkpoint exists locally, or will be downloaded from HF
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Using local checkpoint: {checkpoint_path}")
    else:
        print(f"Checkpoint not found locally: {checkpoint_path}")
        print("Will attempt to download from Hugging Face (saketh11/ColiFormer) if needed")


def main():
    """Main entry point for benchmark evaluation."""
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation for ENCOT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with configuration file
    python scripts/run_benchmarks.py --config configs/benchmark.yaml

    # Run with command-line arguments
    python scripts/run_benchmarks.py --excel_path Benchmark_80_sequences.xlsx --checkpoint_path models/my_model.ckpt

    # Override config values
    python scripts/run_benchmarks.py --config configs/benchmark.yaml --use_gpu --max_test_proteins 50
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--excel_path",
        type=str,
        default=None,
        help="Path to benchmark Excel file (overrides config)"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to model checkpoint (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (overrides config)"
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU if available (overrides config)"
    )
    parser.add_argument(
        "--max_test_proteins",
        type=int,
        default=None,
        help="Maximum number of proteins to test (overrides config)"
    )

    args = parser.parse_args()

    try:
        # Lazy import so `--help` works even if plotting/ML deps are missing.
        from benchmark_evaluation import main as benchmark_main

        if args.config:
            # Load configuration from file
            print(f"Loading configuration from {args.config}...")
            config = load_config(args.config)

            # Override with command-line arguments if provided
            if args.excel_path:
                config.setdefault('data', {})['excel_path'] = args.excel_path
            if args.checkpoint_path:
                config.setdefault('model', {})['checkpoint_path'] = args.checkpoint_path
            if args.output_dir:
                config.setdefault('output', {})['output_dir'] = args.output_dir
            if args.use_gpu:
                config.setdefault('evaluation', {})['use_gpu'] = True
            if args.max_test_proteins is not None:
                config.setdefault('evaluation', {})['max_test_proteins'] = args.max_test_proteins

            # Validate configuration
            validate_config(config)

            # Convert config to args namespace
            benchmark_args = config_to_args(config)
        else:
            # Use command-line arguments directly
            if not args.excel_path:
                parser.error("Either --config or --excel_path must be provided")

            benchmark_args = argparse.Namespace()
            benchmark_args.excel_path = args.excel_path
            benchmark_args.checkpoint_path = args.checkpoint_path or 'models/alm-enhanced-training/balanced_alm_finetune.ckpt'
            benchmark_args.natural_sequences_path = 'data/ecoli_processed_genes.csv'
            benchmark_args.output_dir = args.output_dir or 'benchmark_results'
            benchmark_args.use_gpu = args.use_gpu
            benchmark_args.max_test_proteins = args.max_test_proteins or 0
            benchmark_args.name_col = None
            benchmark_args.seq_col = None
            benchmark_args.sheet_name = None

            # Validate
            if not os.path.exists(benchmark_args.excel_path):
                raise ValueError(f"Benchmark Excel file not found: {benchmark_args.excel_path}")

        # Print configuration summary
        print("\n" + "="*60)
        print("Benchmark Configuration Summary")
        print("="*60)
        print(f"Excel file: {benchmark_args.excel_path}")
        print(f"Checkpoint: {benchmark_args.checkpoint_path}")
        print(f"Output directory: {benchmark_args.output_dir}")
        print(f"Use GPU: {benchmark_args.use_gpu}")
        print(f"Max test proteins: {benchmark_args.max_test_proteins if benchmark_args.max_test_proteins > 0 else 'All'}")
        print("="*60 + "\n")

        # Run benchmark
        benchmark_main(benchmark_args)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

