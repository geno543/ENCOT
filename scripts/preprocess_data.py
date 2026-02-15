"""
Preprocess E. coli gene data for ColiFormer training.

This script combines the functionality of prepare_ecoli_data.py and
create_model_datasets.py to prepare training and test datasets from raw CSV files.

Usage:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --cai_csv data/CAI.csv --high_cai_csv data/Database_3_4300_gene.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path to import CodonTransformer
sys.path.insert(0, str(Path(__file__).parent.parent))


def is_valid_sequence(dna_seq: str) -> bool:
    """
    Validate a DNA sequence for training suitability.

    Args:
        dna_seq: DNA sequence string

    Returns:
        True if sequence is valid (divisible by 3, proper start/stop codons, no internal stops)
    """
    if len(dna_seq) % 3 != 0:
        return False
    if not dna_seq.upper().startswith(('ATG', 'TTG', 'CTG', 'GTG')):
        return False
    if not dna_seq.upper().endswith(('TAA', 'TAG', 'TGA')):
        return False

    codons = [dna_seq[i:i+3].upper() for i in range(0, len(dna_seq) - 3, 3)]
    if any(codon in ['TAA', 'TAG', 'TGA'] for codon in codons):
        return False

    if not all(c in 'ATGC' for c in dna_seq.upper()):
        return False

    return True


def process_ecoli_data(cai_csv: str, high_cai_csv: str, output_dir: str = "data"):
    """
    Process raw E. coli gene data from CSV files.

    Args:
        cai_csv: Path to CAI.csv file with gene data
        high_cai_csv: Path to Database 3_4300 gene.csv with high-CAI sequences
        output_dir: Output directory for processed files

    Returns:
        Path to processed CSV file
    """
    # Lazy imports so `python scripts/preprocess_data.py --help` works without heavy deps installed.
    import pandas as pd
    from Bio.Seq import Seq

    # Validate input files exist
    if not os.path.exists(cai_csv):
        raise FileNotFoundError(f"CAI CSV file not found: {cai_csv}")
    if not os.path.exists(high_cai_csv):
        raise FileNotFoundError(f"High-CAI CSV file not found: {high_cai_csv}")

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data from CSV files...")
    df_all = pd.read_csv(
        cai_csv,
        header=0,
        names=['gene_id', 'cai_score', 'drop1', 'drop2', 'dna_sequence', 'drop3']
    )
    df_high_cai = pd.read_csv(
        high_cai_csv,
        header=0,
        names=['dna_sequence']
    )

    high_cai_sequences = set(df_high_cai['dna_sequence'])

    validated_genes = []
    for index, row in df_all.iterrows():
        gene_id = row['gene_id']
        dna_sequence = str(row['dna_sequence'])

        if is_valid_sequence(dna_sequence):
            protein_sequence = str(Seq(dna_sequence).translate())
            is_high_cai = dna_sequence in high_cai_sequences

            validated_genes.append({
                'gene_id': gene_id,
                'dna_sequence': dna_sequence,
                'protein_sequence': protein_sequence,
                'cai_score': row.get('cai_score', None),
                'is_high_cai': is_high_cai
            })

    df_processed = pd.DataFrame(validated_genes)

    output_path = os.path.join(output_dir, 'ecoli_processed_genes.csv')
    df_processed.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    print(f"Total validated genes: {len(df_processed)}")

    return output_path


def create_train_test_splits(processed_csv: str, output_dir: str = "data", test_size: int = 100):
    """
    Create training and test splits from processed data.

    Args:
        processed_csv: Path to processed ecoli_processed_genes.csv
        output_dir: Output directory for JSON files
        test_size: Number of sequences for test set

    Returns:
        Tuple of (finetune_json_path, test_json_path)
    """
    # Lazy imports so `--help` works without heavy deps installed.
    import pandas as pd
    from CodonTransformer.CodonData import prepare_training_data

    if not os.path.exists(processed_csv):
        raise FileNotFoundError(f"Processed data file not found: {processed_csv}")

    os.makedirs(output_dir, exist_ok=True)

    df_processed = pd.read_csv(processed_csv)

    # Create fine-tuning set (high-CAI sequences)
    df_finetune = df_processed[df_processed['is_high_cai'] == True].copy()
    df_finetune.drop_duplicates(subset=['dna_sequence'], inplace=True)
    df_finetune.rename(columns={'dna_sequence': 'dna', 'protein_sequence': 'protein'}, inplace=True)
    df_finetune['organism'] = "Escherichia coli general"

    finetune_output_path = os.path.join(output_dir, 'finetune_set.json')
    prepare_training_data(df_finetune, finetune_output_path, shuffle=True)
    print(f"Fine-tuning set saved to {finetune_output_path} with {len(df_finetune)} records.")

    # Create test set (non-high-CAI sequences)
    df_test_pool = df_processed[df_processed['is_high_cai'] == False].copy()
    df_test = df_test_pool.sample(n=test_size, random_state=42)  # for reproducibility
    df_test['organism'] = 51  # E. coli general organism ID
    df_test.rename(columns={'dna_sequence': 'codons'}, inplace=True)
    test_records = df_test[['codons', 'organism']].to_dict(orient='records')

    test_output_path = os.path.join(output_dir, 'test_set.json')
    with open(test_output_path, 'w') as f:
        json.dump(test_records, f, indent=4)
    print(f"Test set saved to {test_output_path} with {len(df_test)} records.")

    return finetune_output_path, test_output_path


def main():
    """Main entry point for data preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess E. coli gene data for ENCOT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default paths
    python scripts/preprocess_data.py

    # Specify custom input files
    python scripts/preprocess_data.py --cai_csv data/CAI.csv --high_cai_csv data/Database_3_4300_gene.csv

    # Custom output directory and test size
    python scripts/preprocess_data.py --output_dir my_data --test_size 200
        """
    )
    parser.add_argument(
        "--cai_csv",
        type=str,
        default="data/CAI.csv",
        help="Path to CAI.csv file with gene data (default: data/CAI.csv)"
    )
    parser.add_argument(
        "--high_cai_csv",
        type=str,
        default="data/Database 3_4300 gene.csv",
        help="Path to Database 3_4300 gene.csv file (default: data/Database 3_4300 gene.csv)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for processed files (default: data)"
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=100,
        help="Number of sequences for test set (default: 100)"
    )
    parser.add_argument(
        "--skip_processing",
        action="store_true",
        help="Skip data processing step (assume ecoli_processed_genes.csv exists)"
    )

    args = parser.parse_args()

    try:
        # Step 1: Process raw data
        if not args.skip_processing:
            processed_csv = process_ecoli_data(
                args.cai_csv,
                args.high_cai_csv,
                args.output_dir
            )
        else:
            processed_csv = os.path.join(args.output_dir, 'ecoli_processed_genes.csv')
            if not os.path.exists(processed_csv):
                raise FileNotFoundError(
                    f"Processed data not found at {processed_csv}. "
                    "Remove --skip_processing flag to process raw data first."
                )
            print(f"Using existing processed data: {processed_csv}")

        # Step 2: Create train/test splits
        finetune_path, test_path = create_train_test_splits(
            processed_csv,
            args.output_dir,
            args.test_size
        )

        print("\n" + "="*60)
        print("Data preprocessing complete!")
        print("="*60)
        print(f"Training set: {finetune_path}")
        print(f"Test set: {test_path}")
        print("\nYou can now run training with:")
        print(f"  python scripts/train.py --config configs/train_ecoli_alm.yaml")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

