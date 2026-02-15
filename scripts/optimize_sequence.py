"""
Optimize protein sequences using ColiFormer.

This script provides a user-friendly interface for codon optimization,
supporting both single sequences and batch processing via FASTA files.

Usage:
    # Single sequence
    python scripts/optimize_sequence.py --input "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGG" --output optimized.fasta

    # Batch processing from FASTA file
    python scripts/optimize_sequence.py --input sequences.fasta --output optimized.fasta --batch

    # With GC content constraints
    python scripts/optimize_sequence.py --input protein.fasta --output optimized.fasta --gc-min 0.45 --gc-max 0.55
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Tuple

# Add parent directory to path to import CodonTransformer
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    """
    Parse FASTA file into list of (name, sequence) tuples.

    Args:
        fasta_path: Path to FASTA file

    Returns:
        List of (name, sequence) tuples
    """
    sequences = []
    current_name = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name is not None:
                    sequences.append((current_name, ''.join(current_seq)))
                current_name = line[1:] if len(line) > 1 else f"sequence_{len(sequences)+1}"
                current_seq = []
            else:
                current_seq.append(line.upper())

        if current_name is not None:
            sequences.append((current_name, ''.join(current_seq)))

    return sequences


def write_fasta(output_path: str, sequences: List[Tuple[str, str]]):
    """
    Write sequences to FASTA file.

    Args:
        output_path: Output FASTA file path
        sequences: List of (name, sequence) tuples
    """
    with open(output_path, 'w') as f:
        for name, seq in sequences:
            f.write(f">{name}\n")
            # Write sequence in 60-character lines
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")


def optimize_single_sequence(
    protein: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    organism: str = "Escherichia coli general",
    gc_min: float = None,
    gc_max: float = None,
    cai_weights: dict = None,
    tai_weights: dict = None
) -> dict:
    """
    Optimize a single protein sequence.

    Args:
        protein: Protein sequence string
        model: Loaded ColiFormer model
        tokenizer: Tokenizer
        device: PyTorch device
        organism: Target organism name
        gc_min: Minimum GC content (0-1)
        gc_max: Maximum GC content (0-1)
        cai_weights: CAI weights dictionary
        tai_weights: tAI weights dictionary

    Returns:
        Dictionary with optimization results
    """
    # Lazy imports so `python scripts/optimize_sequence.py --help` works without ML deps installed.
    from CodonTransformer.CodonPrediction import predict_dna_sequence
    from CodonTransformer.CodonEvaluation import get_GC_content, calculate_tAI
    from CAI import CAI

    # Determine GC bounds if specified
    gc_bounds = None
    use_constrained = False
    if gc_min is not None and gc_max is not None:
        gc_bounds = (gc_min, gc_max)
        use_constrained = True

    # Run optimization
    output = predict_dna_sequence(
        protein=protein,
        organism=organism,
        device=device,
        model=model,
        tokenizer=tokenizer,
        deterministic=True,
        match_protein=True,
        use_constrained_search=use_constrained,
        gc_bounds=gc_bounds,
        beam_size=20 if use_constrained else 5,
    )

    if isinstance(output, list):
        output = output[0]

    optimized_dna = output.predicted_dna

    # Calculate metrics
    gc_content = get_GC_content(optimized_dna) / 100.0  # Convert to fraction

    metrics = {
        'protein': protein,
        'optimized_dna': optimized_dna,
        'gc_content': gc_content,
        'length': len(optimized_dna),
    }

    if cai_weights:
        try:
            metrics['cai'] = CAI(optimized_dna, weights=cai_weights)
        except:
            metrics['cai'] = None
    else:
        metrics['cai'] = None

    if tai_weights:
        try:
            metrics['tai'] = calculate_tAI(optimized_dna, tai_weights)
        except:
            metrics['tai'] = None
    else:
        metrics['tai'] = None

    return metrics


def load_reference_data(ref_sequences_path: str = None):
    """
    Load reference sequences and calculate CAI weights.

    Args:
        ref_sequences_path: Path to CSV with reference sequences

    Returns:
        Tuple of (cai_weights, tai_weights)
    """
    # Lazy imports so `--help` works without ML deps installed.
    import pandas as pd
    from CAI import relative_adaptiveness
    from CodonTransformer.CodonEvaluation import get_ecoli_tai_weights

    cai_weights = None
    tai_weights = None

    # Try to load reference sequences for CAI
    if ref_sequences_path and os.path.exists(ref_sequences_path):
        try:
            df = pd.read_csv(ref_sequences_path)
            if 'dna_sequence' in df.columns:
                ref_sequences = df['dna_sequence'].tolist()
                cai_weights = relative_adaptiveness(sequences=ref_sequences)
                print(f"Loaded CAI weights from {len(ref_sequences)} reference sequences")
        except Exception as e:
            print(f"Warning: Could not load CAI weights: {e}")

    # Load tAI weights
    try:
        tai_weights = get_ecoli_tai_weights()
        print("Loaded E. coli tAI weights")
    except Exception as e:
        print(f"Warning: Could not load tAI weights: {e}")

    return cai_weights, tai_weights


def main():
    """Main entry point for sequence optimization."""
    parser = argparse.ArgumentParser(
        description="Optimize protein sequences using ENCOT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single sequence
    python scripts/optimize_sequence.py --input "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGG" --output optimized.fasta

    # Batch processing from FASTA file
    python scripts/optimize_sequence.py --input sequences.fasta --output optimized.fasta --batch

    # With GC content constraints
    python scripts/optimize_sequence.py --input protein.fasta --output optimized.fasta --gc-min 0.45 --gc-max 0.55

    # Use custom checkpoint
    python scripts/optimize_sequence.py --input protein.fasta --output optimized.fasta --checkpoint models/my_model.ckpt
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input protein sequence (string) or FASTA file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output FASTA file path"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: auto-download from Hugging Face)"
    )
    parser.add_argument(
        "--organism",
        type=str,
        default="Escherichia coli general",
        help="Target organism (default: Escherichia coli general)"
    )
    parser.add_argument(
        "--gc-min",
        type=float,
        default=None,
        help="Minimum GC content (0-1, e.g., 0.45 for 45%%)"
    )
    parser.add_argument(
        "--gc-max",
        type=float,
        default=None,
        help="Maximum GC content (0-1, e.g., 0.55 for 55%%)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process input as FASTA file with multiple sequences"
    )
    parser.add_argument(
        "--ref-sequences",
        type=str,
        default="data/ecoli_processed_genes.csv",
        help="Path to reference sequences CSV for CAI calculation"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU if available"
    )

    args = parser.parse_args()

    try:
        # Lazy imports so `--help` works without ML deps installed.
        import torch
        from transformers import AutoTokenizer
        from CodonTransformer.CodonPrediction import load_model
        import pandas as pd

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
        print(f"Using device: {device}")

        # Load model
        print("Loading ColiFormer model...")
        if args.checkpoint:
            model = load_model(model_path=args.checkpoint, device=device)
            print(f"Loaded model from {args.checkpoint}")
        else:
            # Try to load from Hugging Face
            try:
                from huggingface_hub import hf_hub_download
                checkpoint_path = hf_hub_download(
                    repo_id="saketh11/ColiFormer",
                    filename="balanced_alm_finetune.ckpt",
                    cache_dir="./hf_cache"
                )
                model = load_model(model_path=checkpoint_path, device=device)
                print("Loaded model from Hugging Face (saketh11/ColiFormer)")
            except Exception as e:
                print(f"Warning: Could not load from Hugging Face: {e}")
                print("Falling back to base CodonTransformer model...")
                from transformers import BigBirdForMaskedLM
                model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer").to(device)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")

        # Load reference data for metrics
        cai_weights, tai_weights = load_reference_data(args.ref_sequences)

        # Parse input
        if args.batch or os.path.exists(args.input):
            # FASTA file
            print(f"Reading sequences from {args.input}...")
            sequences = parse_fasta(args.input)
            print(f"Found {len(sequences)} sequences")
        else:
            # Single sequence string
            sequences = [("sequence_1", args.input.upper())]

        # Optimize sequences
        optimized_sequences = []
        results = []

        for i, (name, protein_seq) in enumerate(sequences, 1):
            print(f"\nOptimizing sequence {i}/{len(sequences)}: {name}")

            metrics = optimize_single_sequence(
                protein=protein_seq,
                model=model,
                tokenizer=tokenizer,
                device=device,
                organism=args.organism,
                gc_min=args.gc_min,
                gc_max=args.gc_max,
                cai_weights=cai_weights,
                tai_weights=tai_weights
            )

            optimized_sequences.append((name, metrics['optimized_dna']))
            results.append({
                'name': name,
                'protein_length': len(protein_seq),
                'dna_length': metrics['length'],
                'gc_content': f"{metrics['gc_content']*100:.2f}%",
                'cai': metrics['cai'],
                'tai': metrics['tai'],
            })

            print(f"  GC content: {metrics['gc_content']*100:.2f}%")
            if metrics['cai']:
                print(f"  CAI: {metrics['cai']:.3f}")
            if metrics['tai']:
                print(f"  tAI: {metrics['tai']:.3f}")

        # Write output
        write_fasta(args.output, optimized_sequences)
        print(f"\nOptimized sequences saved to {args.output}")

        # Print summary
        if len(results) > 1:
            print("\n" + "="*60)
            print("Summary Statistics")
            print("="*60)
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
            print("="*60)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

