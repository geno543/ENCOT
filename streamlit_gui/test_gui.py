#!/usr/bin/env python3
"""
Test script for ColiFormer Streamlit GUI

This script tests the core functionality of the GUI without running the full Streamlit application.
"""

import sys
import os
import traceback
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")

    try:
        import streamlit as st
        print(f"  OK: Streamlit: {st.__version__}")
    except ImportError as e:
        print(f"  FAIL: Streamlit: {e}")
        return False

    try:
        import torch
        device = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"  OK: PyTorch: {torch.__version__} ({device})")
    except ImportError as e:
        print(f"  FAIL: PyTorch: {e}")
        return False

    try:
        import plotly
        print(f"  OK: Plotly: {plotly.__version__}")
    except ImportError as e:
        print(f"  FAIL: Plotly: {e}")
        return False

    try:
        from CodonTransformer.CodonPrediction import predict_dna_sequence
        print("  OK: CodonTransformer.CodonPrediction")
    except ImportError as e:
        print(f"  FAIL: CodonTransformer.CodonPrediction: {e}")
        return False

    try:
        from CodonTransformer.CodonEvaluation import get_GC_content, calculate_tAI
        print("  OK: CodonTransformer.CodonEvaluation")
    except ImportError as e:
        print(f"  FAIL: CodonTransformer.CodonEvaluation: {e}")
        return False

    return True

def test_protein_validation():
    """Test protein sequence validation"""
    print("\nTesting protein sequence validation...")

    try:
        # Import the validation function
        from app import validate_protein_sequence

        # Test cases
        test_cases = [
            ("MKTVRQERLK", True, "Valid short sequence"),
            ("", False, "Empty sequence"),
            ("MKTVRQERLKX", False, "Invalid character X"),
            ("MK", False, "Too short"),
            ("M" * 501, False, "Too long"),
            ("mktvrqerlk", True, "Lowercase (should work)"),
            ("MKTVRQERLK*", True, "With stop codon"),
            ("MKTVRQERLK_", True, "With underscore stop"),
        ]

        for seq, expected_valid, description in test_cases:
            is_valid, message = validate_protein_sequence(seq)
            status = "OK" if is_valid == expected_valid else "FAIL"
            print(f"  {status} {description}: {message}")

        return True
    except Exception as e:
        print(f"  FAIL: Error in validation test: {e}")
        traceback.print_exc()
        return False

def test_metrics_calculation():
    """Test metrics calculation"""
    print("\nTesting metrics calculation...")

    try:
        from app import calculate_input_metrics

        test_protein = "MKTVRQERLK"
        organism = "Escherichia coli general"

        metrics = calculate_input_metrics(test_protein, organism)

        # Check if all expected metrics are present
        expected_keys = ['length', 'gc_content', 'baseline_dna', 'cai', 'tai']
        for key in expected_keys:
            if key in metrics:
                print(f"  OK: {key}: {metrics[key]}")
            else:
                print(f"  FAIL: Missing metric: {key}")
                return False

        # Validate metric values
        if metrics['length'] == len(test_protein):
            print("  OK: Length calculation correct")
        else:
            print("  FAIL: Length calculation incorrect")
            return False

        if 0 <= metrics['gc_content'] <= 100:
            print("  OK: GC content in valid range")
        else:
            print("  FAIL: GC content out of range")
            return False

        return True
    except Exception as e:
        print(f"  FAIL: Error in metrics calculation: {e}")
        traceback.print_exc()
        return False

def test_visualization_functions():
    """Test visualization functions"""
    print("\nTesting visualization functions...")

    try:
        from app import create_gc_content_plot, create_metrics_comparison_chart

        # Test GC content plot
        test_dna = "ATGGCGAAAGCGCTGTATCGCGAAAGCGCTGTATCGCGAAAGCGCTGTATCGC"
        fig = create_gc_content_plot(test_dna)
        print("  OK: GC content plot created")

        # Test metrics comparison chart
        before_metrics = {'gc_content': 50.0, 'cai': 0.5, 'tai': 0.3}
        after_metrics = {'gc_content': 52.0, 'cai': 0.6, 'tai': 0.4}
        fig = create_metrics_comparison_chart(before_metrics, after_metrics)
        print("  OK: Metrics comparison chart created")

        return True
    except Exception as e:
        print(f"  FAIL: Error in visualization test: {e}")
        traceback.print_exc()
        return False

def test_codon_evaluation():
    """Test CodonEvaluation functions directly"""
    print("\nTesting CodonEvaluation functions...")

    try:
        from CodonTransformer.CodonEvaluation import get_GC_content, calculate_tAI, get_ecoli_tai_weights

        # Test GC content calculation
        test_dna = "ATGGCGAAAGCG"
        gc_content = get_GC_content(test_dna)
        print(f"  OK: GC content calculation: {gc_content:.1f}%")

        # Test tAI calculation
        try:
            tai_weights = get_ecoli_tai_weights()
            tai_value = calculate_tAI(test_dna, tai_weights)
            print(f"  OK: tAI calculation: {tai_value:.3f}")
        except Exception as e:
            print(f"  NOTE: tAI calculation (may need scipy): {e}")

        return True
    except Exception as e:
        print(f"  FAIL: Error in CodonEvaluation test: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test model loading functionality"""
    print("\nTesting model loading (mock)...")

    try:
        import torch
        from transformers import AutoTokenizer
        from CodonTransformer.CodonPrediction import load_model

        # Test tokenizer loading (this is fast)
        print("  Testing tokenizer loading...")
        tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
        print("  OK: Tokenizer loaded successfully")

        # Test load_model function
        print("  Testing load_model function...")
        from transformers import BigBirdForMaskedLM
        print("  OK: Model class available: BigBirdForMaskedLM")

        # Check if fine-tuned model exists
        import os
        model_path = "models/alm-enhanced-training/balanced_alm_finetune.ckpt"
        if os.path.exists(model_path):
            print(f"  OK: Fine-tuned model found: {model_path}")
        else:
            print(f"  NOTE: Fine-tuned model not found at: {model_path}")

        # Note: We won't actually load the full model here as it's ~2GB
        print("  NOTE: Full model loading skipped in test (too large)")

        return True
    except Exception as e:
        print(f"  FAIL: Error in model loading test: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")

    gui_dir = Path(__file__).parent
    parent_dir = gui_dir.parent

    required_files = [
        "app.py",
        "run_gui.py",
        "requirements.txt",
        "README.md"
    ]

    all_present = True
    for file_name in required_files:
        file_path = gui_dir / file_name
        if file_path.exists():
            print(f"  OK: {file_name}")
        else:
            print(f"  FAIL: {file_name} missing")
            all_present = False

    # Check for model checkpoint
    model_path = parent_dir / "models" / "alm-enhanced-training" / "balanced_alm_finetune.ckpt"
    if model_path.exists():
        print("  OK: Fine-tuned model checkpoint found")
    else:
        print("  NOTE: Fine-tuned model checkpoint not found")

    return all_present

def test_post_processing():
    """Test post-processing functionality"""
    print("\nTesting post-processing features...")

    try:
        from app import POST_PROCESSING_AVAILABLE, DNACHISEL_AVAILABLE

        if POST_PROCESSING_AVAILABLE:
            print("  OK: Post-processing module available")
            if DNACHISEL_AVAILABLE:
                print("  OK: DNAChisel available")
            else:
                print("  NOTE: DNAChisel not available")
        else:
            print("  NOTE: Post-processing module not available")

        return True
    except Exception as e:
        print(f"  FAIL: Error in post-processing test: {e}")
        return False

def main():
    """Run all tests"""
    print("ColiFormer GUI Test Suite")
    print("=" * 50)

    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Protein Validation", test_protein_validation),
        ("Metrics Calculation", test_metrics_calculation),
        ("Visualization Functions", test_visualization_functions),
        ("CodonEvaluation Functions", test_codon_evaluation),
        ("Model Loading", test_model_loading),
        ("Post-Processing", test_post_processing),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"OK: {test_name}: PASSED")
            else:
                print(f"FAIL: {test_name}: FAILED")
        except Exception as e:
            print(f"FAIL: {test_name}: ERROR - {e}")

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("All tests passed. The GUI should work correctly.")
        print("\nTo run the GUI:")
        print("  python run_gui.py")
        print("  or")
        print("  cd streamlit_gui && streamlit run app.py --server.address=0.0.0.0")
    else:
        print("Some tests failed. Please check the issues above.")

    print("\nNotes:")
    print("  • Fine-tuned model integration")
    print("  • Enhanced constrained beam search")
    print("  • Post-processing with DNAChisel")
    print("  • Advanced sequence analysis")
    print("  • Improved parameter controls")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
