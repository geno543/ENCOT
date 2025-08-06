import sys
import os
import argparse
import pandas as pd
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Any

from CAI import CAI, relative_adaptiveness
from CodonTransformer.CodonData import (
    download_codon_frequencies_from_kazusa,
    get_codon_frequencies,
)
from CodonTransformer.CodonPrediction import (
    load_model,
    predict_dna_sequence,
)
from CodonTransformer.CodonEvaluation import (
    get_GC_content,
    get_ecoli_tai_weights,
    get_min_max_profile,
    calculate_tAI,
    count_negative_cis_elements,
)
from transformers import AutoTokenizer

# Import translate function from evaluate_optimizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_optimizer import translate_dna_to_protein


def find_longest_orf(dna_sequence: str) -> str:
    """Find the longest open reading frame in a DNA sequence."""
    dna_sequence = dna_sequence.upper()
    start_codons = ['ATG']
    stop_codons = ['TAA', 'TAG', 'TGA']
    
    longest_orf = ""
    
    # Check all 3 reading frames
    for frame in range(3):
        current_orf = ""
        in_orf = False
        
        for i in range(frame, len(dna_sequence) - 2, 3):
            codon = dna_sequence[i:i+3]
            if len(codon) != 3:
                break
                
            if codon in start_codons and not in_orf:
                # Start of ORF
                in_orf = True
                current_orf = codon
            elif in_orf:
                current_orf += codon
                if codon in stop_codons:
                    # End of ORF
                    if len(current_orf) > len(longest_orf):
                        longest_orf = current_orf
                    in_orf = False
                    current_orf = ""
        
        # Handle ORF that goes to end of sequence
        if in_orf and len(current_orf) > len(longest_orf):
            longest_orf = current_orf
    
    return longest_orf


def parse_excel_sequences(excel_path: str) -> List[Dict[str, str]]:
    """Parse sequences from the benchmark Excel file."""
    df = pd.read_excel(excel_path)
    sequences = []
    
    for idx, row in df.iterrows():
        # Extract sequence from the Sequence column
        sequence = str(row['Sequence']).strip()
        name = str(row['Name']).strip()
        
        # Remove any '>' characters from name
        if name.startswith('>'):
            name = name[1:].strip()
        
        # Clean sequence - remove any non-alphabetic characters
        sequence = ''.join(filter(str.isalpha, sequence))
        
        # Check if it's a DNA or protein sequence
        # More robust detection: if >95% of characters are ATCGN, it's likely DNA
        dna_chars = sum(1 for c in sequence.upper() if c in 'ATCGN')
        is_dna = (dna_chars / len(sequence)) > 0.95 if len(sequence) > 0 else False
        
        if is_dna:
            # For DNA sequences, try to find the longest ORF
            longest_orf = find_longest_orf(sequence)
            
            if longest_orf and len(longest_orf) >= 30:  # At least 10 codons
                # Use the ORF
                original_dna = longest_orf
                protein_seq = translate_dna_to_protein(longest_orf)
            else:
                # No good ORF found, truncate to nearest multiple of 3
                truncated_len = (len(sequence) // 3) * 3
                if truncated_len >= 30:  # At least 10 codons
                    original_dna = sequence[:truncated_len]
                    protein_seq = translate_dna_to_protein(original_dna)
                else:
                    # Skip sequences that are too short
                    continue
            
            # Remove sequences with early stop codons in protein
            if '*' in protein_seq:
                # Truncate at first stop codon
                stop_pos = protein_seq.find('*')
                if stop_pos >= 10:  # At least 10 amino acids
                    protein_seq = protein_seq[:stop_pos]
                    original_dna = original_dna[:stop_pos*3]
                else:
                    continue  # Skip if too short
            
        else:
            # It's already a protein sequence
            protein_seq = sequence.upper()
            # Remove any stop codons
            protein_seq = protein_seq.replace('*', '')
            original_dna = None
        
        # Skip very short sequences
        if len(protein_seq) < 10:
            continue
        
        sequences.append({
            'id': idx,
            'name': name,
            'protein_sequence': protein_seq,
            'original_sequence': original_dna,
            'is_dna': is_dna
        })
    
    return sequences


def calculate_cfd(dna_sequence: str, codon_frequencies: Dict) -> float:
    """Calculate Codon Frequency Distribution similarity."""
    if not dna_sequence:
        return 0.0
    
    # Calculate codon frequency distribution for the sequence
    codon_count = {}
    total_codons = 0
    
    for i in range(0, len(dna_sequence) - 2, 3):
        codon = dna_sequence[i:i+3].upper()
        if len(codon) == 3:
            codon_count[codon] = codon_count.get(codon, 0) + 1
            total_codons += 1
    
    # Convert to frequencies
    seq_freq = {}
    if total_codons > 0:
        for codon, count in codon_count.items():
            seq_freq[codon] = count / total_codons
    
    # For codon_frequencies dict that has amino2codon structure, we need to extract codon frequencies
    # First, let's flatten the codon frequencies if it's in amino2codon format
    flat_codon_freq = {}
    if isinstance(codon_frequencies, dict):
        # Check if it's amino2codon format
        first_key = next(iter(codon_frequencies.keys()))
        if isinstance(codon_frequencies[first_key], tuple) and len(codon_frequencies[first_key]) == 2:
            # It's amino2codon format - extract codon frequencies
            for amino, (codons, freqs) in codon_frequencies.items():
                for codon, freq in zip(codons, freqs):
                    flat_codon_freq[codon] = freq
        else:
            # Already flat format
            flat_codon_freq = codon_frequencies
    
    # Calculate similarity to E. coli reference
    similarity = 0.0
    count = 0
    
    for codon in set(list(seq_freq.keys()) + list(flat_codon_freq.keys())):
        seq_f = seq_freq.get(codon, 0.0)
        ref_f = flat_codon_freq.get(codon, 0.0)
        # Use 1 - absolute difference as similarity measure
        similarity += 1 - abs(seq_f - ref_f)
        count += 1
    
    return similarity / count if count > 0 else 0.0


def run_model_on_sequences(
    sequences: List[Dict],
    model,
    tokenizer,
    device,
    cai_weights: Dict,
    tai_weights: Dict,
    codon_frequencies: Dict,
    reference_profile: List[float],
    output_dir: str
) -> pd.DataFrame:
    """Run ColiFormer model on sequences and calculate metrics."""
    
    results = []
    
    print(f"Processing {len(sequences)} sequences...")
    
    for seq_data in tqdm(sequences, desc="Optimizing sequences"):
        protein_seq = seq_data['protein_sequence']
        
        # Skip very short sequences (let model handle length limits)
        if len(protein_seq) < 10:
            continue
        
        try:
            # Start timing
            start_time = time.time()
            
            # Run model prediction
            output = predict_dna_sequence(
                protein=protein_seq,
                organism="Escherichia coli general",
                device=device,
                model=model,
                deterministic=True,
                match_protein=True,
            )
            
            runtime = time.time() - start_time
            
            if isinstance(output, list):
                optimized_dna = output[0].predicted_dna
            else:
                optimized_dna = output.predicted_dna
            
            # Calculate metrics for original sequence (if DNA)
            original_metrics = {}
            if seq_data['is_dna'] and seq_data['original_sequence']:
                original_dna = seq_data['original_sequence'].upper()
                original_metrics = {
                    'original_cai': CAI(original_dna, weights=cai_weights),
                    'original_gc': get_GC_content(original_dna),
                    'original_tai': calculate_tAI(original_dna, tai_weights),
                    'original_cfd': calculate_cfd(original_dna, codon_frequencies),
                    'original_neg_cis': count_negative_cis_elements(original_dna),
                }
            
            # Calculate metrics for optimized sequence
            optimized_metrics = {
                'optimized_cai': CAI(optimized_dna, weights=cai_weights),
                'optimized_gc': get_GC_content(optimized_dna),
                'optimized_tai': calculate_tAI(optimized_dna, tai_weights),
                'optimized_cfd': calculate_cfd(optimized_dna, codon_frequencies),
                'optimized_neg_cis': count_negative_cis_elements(optimized_dna),
                'runtime': runtime,
            }
            
            # Combine results
            result = {
                'id': seq_data['id'],
                'name': seq_data['name'],
                'protein_sequence': protein_seq,
                'protein_length': len(protein_seq),
                'optimized_dna': optimized_dna,
                **original_metrics,
                **optimized_metrics,
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing sequence {seq_data['id']}: {str(e)}")
            continue
    
    return pd.DataFrame(results)


def generate_visualizations(results_df: pd.DataFrame, output_dir: str):
    """Generate all required visualizations."""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create output directory for figures
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Before/After CAI Graph
    if 'original_cai' in results_df.columns:
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        before_cai = results_df['original_cai'].dropna()
        after_cai = results_df.loc[before_cai.index, 'optimized_cai']
        
        # Create bar plot
        x = np.arange(len(before_cai))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        bars1 = ax.bar(x - width/2, before_cai, width, label='Before Optimization', alpha=0.8)
        bars2 = ax.bar(x + width/2, after_cai, width, label='After Optimization', alpha=0.8)
        
        ax.set_xlabel('Sequence Index', fontsize=12)
        ax.set_ylabel('CAI Score', fontsize=12)
        ax.set_title('ColiFormer: CAI Before and After Optimization', fontsize=14, fontweight='bold')
        ax.set_xticks(x[::5])  # Show every 5th label
        ax.set_xticklabels(x[::5])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add improvement percentage
        avg_before = before_cai.mean()
        avg_after = after_cai.mean()
        improvement = ((avg_after - avg_before) / avg_before) * 100
        
        ax.text(0.02, 0.98, f'Average CAI Before: {avg_before:.3f}\nAverage CAI After: {avg_after:.3f}\nImprovement: {improvement:.1f}%',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'cai_before_after.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"CAI Before/After graph saved to {os.path.join(fig_dir, 'cai_before_after.png')}")
        
        # 1b. Median CAI Before/After Graph
        plt.figure(figsize=(8, 6))
        
        median_before = before_cai.median()
        median_after = after_cai.median()
        
        categories = ['Before Optimization', 'After Optimization']
        medians = [median_before, median_after]
        colors = ['#ff7f0e', '#2ca02c']
        
        bars = plt.bar(categories, medians, color=colors, alpha=0.8, width=0.6)
        plt.ylabel('Median CAI Score', fontsize=12)
        plt.title('ColiFormer: Median CAI Before and After Optimization', fontsize=14, fontweight='bold')
        plt.ylim(0, max(medians) * 1.2)
        
        # Add value labels on bars
        for bar, median in zip(bars, medians):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{median:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add improvement percentage
        improvement_pct = ((median_after - median_before) / median_before) * 100
        plt.text(0.5, max(medians) * 0.95, f'Improvement: {improvement_pct:.1f}%', 
                ha='center', transform=plt.gca().transData, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'median_cai_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Median CAI comparison graph saved to {os.path.join(fig_dir, 'median_cai_comparison.png')}")
    
    # 2. Summary metrics table
    metrics_summary = {}
    
    # Calculate averages for all metrics
    if 'original_cai' in results_df.columns:
        metrics_summary['CAI'] = {
            'Before': results_df['original_cai'].mean(),
            'After': results_df['optimized_cai'].mean(),
            'Improvement': ((results_df['optimized_cai'].mean() - results_df['original_cai'].mean()) / results_df['original_cai'].mean()) * 100
        }
        metrics_summary['GC Content (%)'] = {
            'Before': results_df['original_gc'].mean(),
            'After': results_df['optimized_gc'].mean(),
            'Difference': results_df['optimized_gc'].mean() - results_df['original_gc'].mean()
        }
        metrics_summary['tAI'] = {
            'Before': results_df['original_tai'].mean(),
            'After': results_df['optimized_tai'].mean(),
            'Improvement': ((results_df['optimized_tai'].mean() - results_df['original_tai'].mean()) / results_df['original_tai'].mean()) * 100
        }
        metrics_summary['CFD'] = {
            'Before': results_df['original_cfd'].mean(),
            'After': results_df['optimized_cfd'].mean(),
            'Improvement': ((results_df['optimized_cfd'].mean() - results_df['original_cfd'].mean()) / results_df['original_cfd'].mean()) * 100
        }
        metrics_summary['Negative Cis Elements'] = {
            'Before': results_df['original_neg_cis'].mean(),
            'After': results_df['optimized_neg_cis'].mean(),
            'Reduction': results_df['original_neg_cis'].mean() - results_df['optimized_neg_cis'].mean()
        }
    else:
        # Only optimized metrics available
        metrics_summary['CAI'] = {
            'Optimized': results_df['optimized_cai'].mean(),
            'Std Dev': results_df['optimized_cai'].std()
        }
        metrics_summary['GC Content (%)'] = {
            'Optimized': results_df['optimized_gc'].mean(),
            'Std Dev': results_df['optimized_gc'].std()
        }
        metrics_summary['tAI'] = {
            'Optimized': results_df['optimized_tai'].mean(),
            'Std Dev': results_df['optimized_tai'].std()
        }
        metrics_summary['CFD'] = {
            'Optimized': results_df['optimized_cfd'].mean(),
            'Std Dev': results_df['optimized_cfd'].std()
        }
        metrics_summary['Negative Cis Elements'] = {
            'Optimized': results_df['optimized_neg_cis'].mean(),
            'Std Dev': results_df['optimized_neg_cis'].std()
        }
    
    # Add runtime statistics
    metrics_summary['Runtime (seconds)'] = {
        'Mean': results_df['runtime'].mean(),
        'Median': results_df['runtime'].median(),
        'Total': results_df['runtime'].sum()
    }
    
    # Convert to DataFrame for nice display
    summary_df = pd.DataFrame(metrics_summary).T
    summary_df = summary_df.round(4)
    
    # Save summary table
    summary_df.to_csv(os.path.join(output_dir, 'metrics_summary.csv'))
    print(f"\nMetrics Summary saved to {os.path.join(output_dir, 'metrics_summary.csv')}")
    print("\n" + "="*60)
    print("METRICS SUMMARY:")
    print("="*60)
    print(summary_df.to_string())
    
    # 3. Distribution plots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics_to_plot = [
        ('optimized_cai', 'CAI Distribution'),
        ('optimized_gc', 'GC Content Distribution (%)'),
        ('optimized_tai', 'tAI Distribution'),
        ('optimized_cfd', 'CFD Distribution'),
        ('optimized_neg_cis', 'Negative Cis Elements'),
        ('runtime', 'Runtime Distribution (seconds)')
    ]
    
    for idx, (col, title) in enumerate(metrics_to_plot):
        if col in results_df.columns:
            axes[idx].hist(results_df[col].dropna(), bins=20, edgecolor='black', alpha=0.7)
            axes[idx].set_title(title, fontsize=10, fontweight='bold')
            axes[idx].set_xlabel(col.replace('optimized_', '').replace('_', ' ').title())
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add mean line
            mean_val = results_df[col].mean()
            axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            axes[idx].legend()
    
    plt.suptitle('ColiFormer: Optimization Metrics Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics distribution plot saved to {os.path.join(fig_dir, 'metrics_distribution.png')}")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Benchmark ColiFormer on E. coli sequences")
    parser.add_argument("--excel_path", type=str, default="Benchmark 80 sequences.xlsx",
                        help="Path to benchmark Excel file")
    parser.add_argument("--checkpoint_path", type=str, default="models/ecoli-codon-optimizer/finetune_best.ckpt",
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--natural_sequences_path", type=str, default="data/ecoli_processed_genes.csv",
                        help="Path to natural E. coli sequences for CAI calculation")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Directory to save results")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("COLIFORMER BENCHMARK EVALUATION")
    print("="*60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Load sequences from Excel
    print(f"\nLoading sequences from {args.excel_path}...")
    sequences = parse_excel_sequences(args.excel_path)
    print(f"Loaded {len(sequences)} sequences")
    
    # Load model
    print("\nLoading ColiFormer model...")
    model = load_model(model_path=args.checkpoint_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
    print("Model loaded successfully")
    
    # Prepare evaluation utilities
    print("\nPreparing evaluation utilities...")
    
    # Load natural sequences for CAI weights
    natural_df = pd.read_csv(args.natural_sequences_path)
    ref_sequences = natural_df['dna_sequence'].tolist()
    cai_weights = relative_adaptiveness(sequences=ref_sequences)
    print("CAI weights generated")
    
    # tAI weights
    tai_weights = get_ecoli_tai_weights()
    print("tAI weights loaded")
    
    # Codon frequencies
    try:
        codon_frequencies = download_codon_frequencies_from_kazusa(taxonomy_id=83333)
        print("Codon frequencies loaded from Kazusa")
    except Exception as e:
        print(f"Warning: Kazusa download failed ({e}). Using local frequencies.")
        codon_frequencies = get_codon_frequencies(
            ref_sequences, organism="Escherichia coli general"
        )
    
    # Reference profile for DTW (simplified for now)
    reference_profile = []
    
    # Run model on sequences
    print("\n" + "="*60)
    print("RUNNING OPTIMIZATION...")
    print("="*60)
    
    results_df = run_model_on_sequences(
        sequences=sequences,
        model=model,
        tokenizer=tokenizer,
        device=device,
        cai_weights=cai_weights,
        tai_weights=tai_weights,
        codon_frequencies=codon_frequencies,
        reference_profile=reference_profile,
        output_dir=output_dir
    )
    
    # Save raw results
    results_path = os.path.join(output_dir, 'optimization_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nRaw results saved to {results_path}")
    
    # Generate visualizations and summary
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS...")
    print("="*60)
    
    summary_df = generate_visualizations(results_df, output_dir)
    
    print("\n" + "="*60)
    print("BENCHMARK EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"Total sequences processed: {len(results_df)}")
    print(f"Average runtime per sequence: {results_df['runtime'].mean():.2f} seconds")
    print(f"Total runtime: {results_df['runtime'].sum():.2f} seconds")


if __name__ == "__main__":
    main()