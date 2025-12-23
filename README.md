# ColiFormer: A Transformer-Based Codon Optimization Model Balancing Multiple Objectives for Enhanced E. coli Gene Expression


<p align="center">
  <a href="https://huggingface.co/saketh11/ColiFormer"><img src="https://img.shields.io/badge/HuggingFace-Model-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Model"></a>
  <a href="https://huggingface.co/datasets/saketh11/ColiFormer-Data"><img src="https://img.shields.io/badge/HuggingFace-Data-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Dataset"></a>
</p>

## Abstract

ColiFormer is a transformer-based model for codon optimization of protein sequences in *Escherichia coli*. Built on top of CodonTransformer (a multi-species BigBird model trained on over 1 million DNA–protein pairs), ColiFormer is fine-tuned specifically for E. coli codon preferences using 3,676 high-expression E. coli genes curated from NCBI.

ColiFormer balances multiple objectives (CAI, GC content, tAI, RNA stability, and minimization of negative cis-regulatory elements) and uses an **Augmented-Lagrangian Method (ALM)** to enforce GC content control during training. Performance was evaluated on 37,053 native E. coli genes and 80 recombinant protein targets, demonstrating strong improvements in in silico expression metrics while maintaining biologically appropriate constraints.

## Paper Reference

**ColiFormer: A Transformer-Based Codon Optimization Model Balancing Multiple Objectives for Enhanced E. coli Gene Expression**

Saketh Baddam, Omar Emam, Abdelrahman Elfikky, Francesco Cavarretta, George Luka, Ibrahim Farag, Yasser Sanad

bioRxiv preprint (not peer-reviewed): `https://doi.org/10.1101/2025.11.26.690826`

**What does “preprint and not peer-reviewed” mean?** A preprint is a publicly available manuscript shared before formal journal peer review. It can be cited, but its claims have not yet been evaluated by journal referees.

### Citation

If you use ColiFormer in your research, please cite:

```bibtex
@article{coliformer2025,
  title={ColiFormer: A Transformer-Based Codon Optimization Model Balancing Multiple Objectives for Enhanced E. coli Gene Expression},
  author={Baddam, Saketh and Emam, Omar and Elfikky, Abdelrahman and Cavarretta, Francesco and Luka, George and Farag, Ibrahim and Sanad, Yasser},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.11.26.690826},
  url={https://doi.org/10.1101/2025.11.26.690826},
  note={Preprint (not peer-reviewed)}
}
```

## Quick Start

Optimize a protein sequence in just a few lines:

```python
import torch
from transformers import AutoTokenizer
from CodonTransformer.CodonPrediction import load_model, predict_dna_sequence
from huggingface_hub import hf_hub_download

# Load model from Hugging Face
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = hf_hub_download(
    repo_id="saketh11/ColiFormer",
    filename="balanced_alm_finetune.ckpt",
    cache_dir="./hf_cache"
)
model = load_model(model_path=checkpoint_path, device=device)
tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")

# Optimize a protein sequence
protein = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGG"
output = predict_dna_sequence(
    protein=protein,
    organism="Escherichia coli general",
    device=device,
    model=model,
    tokenizer=tokenizer,
    deterministic=True,
    match_protein=True
)

print(f"Optimized DNA: {output.predicted_dna}")
```

Or use the command-line interface:

```bash
python scripts/optimize_sequence.py \
    --input "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGG" \
    --output optimized.fasta
```

## Installation

### Requirements

- Python >= 3.9
- CUDA-capable GPU (recommended for training, optional for inference)

### Setup

1. **Clone the repository:**

```bash
git clone https://github.com/SAKETH11111/ColiFormer.git
cd ColiFormer
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

The installation takes approximately 10-30 seconds depending on your system and existing packages.

## Data Preparation

### Preparing E. coli Training Data

To prepare training data from raw E. coli gene sequences:

1. **Place your data files in the `data/` directory:**
   - `data/CAI.csv` - CSV file with columns: gene_id, cai_score, dna_sequence
   - `data/Database 3_4300 gene.csv` - CSV file with high-CAI sequences (column: dna_sequence)

2. **Run the preprocessing script:**

```bash
python scripts/preprocess_data.py
```

This will:
- Validate and process DNA sequences
- Create `data/ecoli_processed_genes.csv` with validated sequences
- Generate `data/finetune_set.json` for training (high-CAI sequences)
- Generate `data/test_set.json` for evaluation (100 random sequences)

**Custom paths:**

```bash
python scripts/preprocess_data.py \
    --cai_csv data/my_cai_data.csv \
    --high_cai_csv data/my_high_cai_data.csv \
    --output_dir my_data \
    --test_size 200
```

### Dataset Structure

The processed dataset includes:
- **Training set**: 4,300 high-CAI E. coli sequences (from `Database 3_4300 gene.csv`)
- **Test set**: 100 randomly sampled sequences (for evaluation)
- **Reference sequences**: 50,000+ E. coli genes for CAI/tAI calculation

The complete dataset is available at [saketh11/ColiFormer-Data](https://huggingface.co/datasets/saketh11/ColiFormer-Data) on Hugging Face.

## Training

### Quick Start Training

Train ColiFormer with the default ALM configuration:

```bash
python scripts/train.py --config configs/train_ecoli_alm.yaml
```

### Configuration Files

We provide three configuration files:

1. **`configs/train_ecoli_alm.yaml`** - Main training configuration with ALM GC control
   - 15 epochs, batch size 6, 4 GPUs
   - ALM enabled with GC target 52%
   - Curriculum learning: 3 warm-up epochs

2. **`configs/train_ecoli_quick.yaml`** - Quick sanity check
   - 1 epoch, batch size 2, CPU-only
   - Useful for testing your setup

3. **`configs/benchmark.yaml`** - Benchmark evaluation settings

### Training Parameters

Key parameters in the config files:

- **`training.batch_size`**: Batch size (default: 6)
- **`training.max_epochs`**: Number of training epochs (default: 15)
- **`training.learning_rate`**: Learning rate (default: 5e-5)
- **`training.num_gpus`**: Number of GPUs (default: 4)
- **`alm.enabled`**: Enable ALM GC control (default: true)
- **`alm.gc_target`**: Target GC content (default: 0.52 for E. coli)
- **`alm.curriculum_epochs`**: Warm-up epochs before enforcing GC constraint (default: 3)

### Override Config Values

You can override config values from the command line:

```bash
python scripts/train.py \
    --config configs/train_ecoli_alm.yaml \
    --num_gpus 2 \
    --batch_size 4 \
    --max_epochs 10
```

### Training Output

Checkpoints are saved to the directory specified in `checkpoint.checkpoint_dir`:
- Model state dict: `balanced_alm_finetune.ckpt`
- Training logs: TensorBoard logs in the checkpoint directory

Monitor training progress:

```bash
tensorboard --logdir models/alm-enhanced-training
```

## Inference / Sequence Optimization

### Single Sequence Optimization

Optimize a single protein sequence:

```bash
python scripts/optimize_sequence.py \
    --input "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGG" \
    --output optimized.fasta
```

### Batch Processing

Process multiple sequences from a FASTA file:

```bash
python scripts/optimize_sequence.py \
    --input sequences.fasta \
    --output optimized.fasta \
    --batch
```

### GC Content Constraints

Specify GC content bounds:

```bash
python scripts/optimize_sequence.py \
    --input protein.fasta \
    --output optimized.fasta \
    --gc-min 0.45 \
    --gc-max 0.55
```

### Using Custom Checkpoint

```bash
python scripts/optimize_sequence.py \
    --input protein.fasta \
    --output optimized.fasta \
    --checkpoint models/my_model.ckpt
```

### Python API

For programmatic use:

```python
from CodonTransformer.CodonPrediction import load_model, predict_dna_sequence
from transformers import AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(model_path="models/alm-enhanced-training/balanced_alm_finetune.ckpt", device=device)
tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")

output = predict_dna_sequence(
    protein="MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGG",
    organism="Escherichia coli general",
    device=device,
    model=model,
    tokenizer=tokenizer,
    deterministic=True,
    match_protein=True,
    use_constrained_search=True,
    gc_bounds=(0.45, 0.55),
    beam_size=20
)

print(f"Optimized DNA: {output.predicted_dna}")
```

## Reproducing Paper Results

### Benchmark Evaluation

To reproduce the benchmark results from the paper:

1. **Prepare benchmark sequences:**

   Place your benchmark sequences in an Excel file (see `Benchmark 80 sequences.xlsx` for format).

2. **Run benchmark evaluation:**

```bash
python scripts/run_benchmarks.py --config configs/benchmark.yaml
```

This will:
- Load the fine-tuned ColiFormer model
- Optimize all sequences in the benchmark file
- Calculate metrics (CAI, tAI, GC content, CFD, negative cis-elements)
- Generate comparison plots and summary statistics
- Save results to `benchmark_results/run_TIMESTAMP/`

### Expected Results

On the benchmark set of 80 sequences:
- **CAI improvement**: +6.2% vs base CodonTransformer
- **tAI improvement**: +8.6% vs base CodonTransformer
- **GC content**: Mean 52.1% (target: 52%)
- **Runtime**: ~1-3 seconds per sequence (GPU)

### Custom Benchmark

```bash
python scripts/run_benchmarks.py \
    --excel_path my_benchmark.xlsx \
    --checkpoint_path models/my_model.ckpt \
    --output_dir my_results \
    --use_gpu
```

## Model Architecture

### Base Model

ColiFormer is built on CodonTransformer, a BigBird transformer model:
- **Architecture**: BigBirdForMaskedLM (89.6M parameters)
- **Pre-training**: 1M+ DNA-protein pairs from 164 organisms
- **Context length**: 2048 tokens
- **Attention**: Block-sparse attention for efficiency

### Fine-tuning

ColiFormer is fine-tuned on E. coli-specific data:
- **Training data**: 4,300 high-CAI E. coli sequences
- **Loss function**: Masked Language Modeling (MLM) + GC constraint
- **Optimizer**: AdamW with CosineAnnealingWarmRestarts scheduler
- **Learning rate**: 5e-5 with 10% warmup

### Augmented-Lagrangian Method (ALM)

The ALM approach enforces GC content constraints during training:

**Objective function:**
```
L = L_MLM + λ·(GC - μ) + (ρ/2)(GC - μ)²
```

Where:
- `L_MLM`: Masked language modeling loss
- `λ`: Lagrangian multiplier (updated adaptively)
- `ρ`: Penalty coefficient (self-tuning)
- `GC`: Mean GC content (sliding window of 50 codons)
- `μ`: Target GC content (0.52 for E. coli)

**Key features:**
- **Curriculum learning**: 3 warm-up epochs before enforcing GC constraint
- **Adaptive penalty**: Penalty coefficient increases if constraint violation doesn't improve
- **Self-tuning**: Lagrangian multiplier and penalty updated every 20 steps

This approach allows the model to learn codon preferences while maintaining precise GC content control, critical for synthesis and expression in E. coli.

## Evaluation Metrics

ColiFormer computes comprehensive metrics for optimized sequences:

- **CAI (Codon Adaptation Index)**: Measures similarity to highly expressed genes (0-1, higher is better)
- **tAI (tRNA Adaptation Index)**: Measures tRNA availability (0-1, higher is better)
- **GC Content**: Percentage of G+C nucleotides (target: 52% for E. coli)
- **CFD (Codon Frequency Distribution)**: Similarity to reference codon frequencies
- **Negative cis-elements**: Count of problematic sequence motifs
- **Homopolymer runs**: Long repeats that can cause synthesis issues

## Project Structure

```
coliformer/
├── configs/                    # YAML configuration files
│   ├── train_ecoli_alm.yaml   # Main training config
│   ├── train_ecoli_quick.yaml # Quick test config
│   └── benchmark.yaml         # Benchmark config
├── scripts/                    # Entry-point scripts
│   ├── preprocess_data.py     # Data preparation
│   ├── train.py               # Training wrapper
│   ├── optimize_sequence.py   # Sequence optimization
│   └── run_benchmarks.py      # Benchmark evaluation
├── CodonTransformer/          # Core module (custom, not PyPI)
│   ├── CodonPrediction.py     # Model loading & inference
│   ├── CodonEvaluation.py     # Metrics calculation
│   ├── CodonData.py           # Data preprocessing
│   └── ...
├── data/                       # Datasets
│   ├── finetune_set.json      # Training data
│   ├── test_set.json          # Test data
│   └── ecoli_processed_genes.csv  # Reference sequences
├── models/                     # Model checkpoints
├── notebooks/                  # Jupyter notebooks
├── tests/                      # Test suite
├── streamlit_gui/             # Streamlit web interface
├── finetune.py                # Training script (original)
├── benchmark_evaluation.py    # Evaluation script (original)
└── README.md                  # This file
```

## Troubleshooting

### Common Issues

**1. CUDA out of memory:**
- Reduce `batch_size` in config file
- Use gradient accumulation: increase `accumulate_grad_batches`

**2. Model checkpoint not found:**
- The script will auto-download from Hugging Face if local checkpoint missing
- Ensure you have internet connection for first run

**3. Data preprocessing errors:**
- Verify CSV files have correct column names
- Check that DNA sequences are valid (divisible by 3, proper start/stop codons)

**4. Import errors:**
- Ensure you've activated the virtual environment
- Run `pip install -r requirements.txt` again

### Getting Help

- **Issues**: Open an issue on GitHub
- **Questions**: Check the documentation or contact the authors

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **CodonTransformer**: Base model from [adibvafa/CodonTransformer](https://github.com/adibvafa/CodonTransformer)
- **Hugging Face**: Model hosting and distribution
- **E. coli data**: NCBI and Kazusa codon usage databases

## Citation

If you use ColiFormer in your research, please cite:

```bibtex
@article{coliformer2025,
  title={ColiFormer: A Transformer-Based Codon Optimization Model Balancing Multiple Objectives for Enhanced E. coli Gene Expression},
  author={Baddam, Saketh and Emam, Omar and Elfikky, Abdelrahman and Cavarretta, Francesco and Luka, George and Farag, Ibrahim and Sanad, Yasser},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.11.26.690826},
  url={https://doi.org/10.1101/2025.11.26.690826},
  note={Preprint (not peer-reviewed)}
}
```

---

**ColiFormer** - State-of-the-art codon optimization for E. coli expression systems.
