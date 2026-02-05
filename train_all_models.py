"""
Train All Models Script
Runs all experiments: multiple models, with/without class weights, and baselines.
"""

import os
import subprocess
import json
from datetime import datetime


def run_command(cmd, description):
    """Run a command and print output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60 + "\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0


def main():
    print("="*60)
    print("Vietnamese Sentiment Analysis - Full Experiment Suite")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    experiments = []
    
    # 1. Traditional ML Baselines
    print("\n" + "="*60)
    print("PHASE 1: Traditional ML Baselines")
    print("="*60)
    
    # SVM baseline
    run_command(
        "python train_baseline.py --model svm",
        "SVM + TF-IDF Baseline"
    )
    experiments.append(("SVM + TF-IDF", "svm_baseline"))
    
    # Logistic Regression baseline
    run_command(
        "python train_baseline.py --model logistic",
        "Logistic Regression + TF-IDF Baseline"
    )
    experiments.append(("Logistic Regression + TF-IDF", "logistic_baseline"))
    
    # 2. Deep Learning Models
    print("\n" + "="*60)
    print("PHASE 2: Deep Learning Models")
    print("="*60)
    
    models = ["phobert-base", "phobert-large", "visobert"]
    
    for model in models:
        # Standard training (train only, validate on dev)
        run_command(
            f"python train_improved.py --model {model}",
            f"{model} (Standard Training)"
        )
        experiments.append((f"{model} (standard)", model.replace("-", "_")))
        
        # With class-weighted loss
        run_command(
            f"python train_improved.py --model {model} --weighted-loss",
            f"{model} (With Class Weights)"
        )
        experiments.append((f"{model} (weighted)", f"{model.replace('-', '_')}_weighted"))
    
    # 3. Best model with merged train+dev
    print("\n" + "="*60)
    print("PHASE 3: Final Training with Merged Data")
    print("="*60)
    
    run_command(
        "python train_improved.py --model phobert-large --weighted-loss --merge-train-dev",
        "PhoBERT-large (Merged + Weighted)"
    )
    experiments.append(("PhoBERT-large (merged+weighted)", "phobert_large_final"))
    
    # 4. Hybrid Feature Fusion Model
    print("\n" + "="*60)
    print("PHASE 4: Hybrid Feature Fusion Model")
    print("="*60)
    
    run_command(
        "python train_hybrid.py",
        "Hybrid (BERT + TF-IDF + Sentiment Lexicon)"
    )
    experiments.append(("Hybrid Fusion", "hybrid_fusion"))
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"\nCompleted {len(experiments)} experiments:")
    for name, folder in experiments:
        print(f"  - {name}: results/{folder}/")
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nRun 'python evaluate_all.py' to generate comparison report.")


if __name__ == "__main__":
    main()
