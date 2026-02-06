"""
Comprehensive Evaluation Script
Collects results from all experiments and generates comparison tables and visualizations.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration
RESULTS_DIR = "results"
OUTPUT_DIR = "evaluation_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def collect_results():
    """Collect results from all experiment directories"""
    results = []
    
    # Exclude V3 models
    EXCLUDE_PATTERNS = ["v3", "V3"]
    
    def should_exclude(name):
        return any(p in name for p in EXCLUDE_PATTERNS)
    
    # Scan results directory for model folders
    for item in os.listdir(RESULTS_DIR):
        item_path = os.path.join(RESULTS_DIR, item)
        if os.path.isdir(item_path) and item != "ensembles":
            # Skip V3 models
            if should_exclude(item):
                print(f"  Skipped: {item} (excluded)")
                continue
                
            # Check for test_results.json
            results_file = os.path.join(item_path, "test_results.json")
            if os.path.exists(results_file):
                with open(results_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Use experiment name from folder
                    data["experiment"] = item
                    results.append(data)
                    print(f"  Found: {item}")
    
    # Also check ensembles folder
    ensemble_dir = os.path.join(RESULTS_DIR, "ensembles")
    if os.path.exists(ensemble_dir):
        for item in os.listdir(ensemble_dir):
            if item.endswith(".json"):
                # Skip V3 ensembles
                if should_exclude(item):
                    print(f"  Skipped: {item} (excluded)")
                    continue
                    
                filepath = os.path.join(ensemble_dir, item)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Create readable name from filename
                    name = item.replace('.json', '').replace('_', ' ').title()
                    data["experiment"] = f"Ensemble: {name}"
                    results.append(data)
                    print(f"  Found: Ensemble: {name}")
    
    return results


def create_comparison_table(results):
    """Create comparison table from results"""
    rows = []
    
    for r in results:
        experiment = r.get("experiment", r.get("model_name", r.get("ensemble_name", "Unknown")))
        row = {
            "Experiment": experiment,
            "Accuracy": r.get("test_accuracy", 0),
            "Weighted F1": r.get("test_weighted_f1", 0),
            "Macro F1": r.get("test_macro_f1", 0),
        }
        
        # Per-class metrics
        if "per_class" in r:
            for cls_name in ["Negative", "Neutral", "Positive"]:
                if cls_name in r["per_class"]:
                    row[f"{cls_name} F1"] = r["per_class"][cls_name].get("f1-score", 0)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values("Weighted F1", ascending=False)
    
    return df


def plot_comparison_bar_chart(df):
    """Create bar chart comparing all models"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Overall metrics
    ax1 = axes[0]
    metrics = ["Accuracy", "Weighted F1", "Macro F1"]
    x = np.arange(len(df))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        ax1.bar(x + i*width, df[metric], width, label=metric)
    
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Score")
    ax1.set_title("Overall Performance Comparison")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(df["Experiment"], rotation=45, ha="right")
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", alpha=0.3)
    
    # Per-class F1
    ax2 = axes[1]
    class_metrics = ["Negative F1", "Neutral F1", "Positive F1"]
    colors = ["#e74c3c", "#95a5a6", "#27ae60"]
    
    for i, (metric, color) in enumerate(zip(class_metrics, colors)):
        if metric in df.columns:
            ax2.bar(x + i*width, df[metric], width, label=metric.replace(" F1", ""), color=color)
    
    ax2.set_xlabel("Model")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Per-Class F1 Score Comparison")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(df["Experiment"], rotation=45, ha="right")
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'model_comparison.png')}")


def plot_heatmap(df):
    """Create heatmap of all metrics"""
    # Select numeric columns
    numeric_cols = [c for c in df.columns if c != "Experiment"]
    
    plt.figure(figsize=(10, max(6, len(df) * 0.5)))
    
    heatmap_data = df.set_index("Experiment")[numeric_cols]
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        linewidths=0.5
    )
    
    plt.title("Performance Metrics Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "metrics_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'metrics_heatmap.png')}")


def plot_neutral_focus(df):
    """Focus on Neutral class performance (the challenging class)"""
    if "Neutral F1" not in df.columns:
        return
    
    plt.figure(figsize=(10, 6))
    
    df_sorted = df.sort_values("Neutral F1", ascending=True)
    
    colors = plt.cm.RdYlGn(df_sorted["Neutral F1"])
    
    bars = plt.barh(df_sorted["Experiment"], df_sorted["Neutral F1"], color=colors)
    
    plt.xlabel("F1 Score")
    plt.title("Neutral Class Performance (Challenge: Imbalanced Data)")
    plt.xlim(0, 1)
    
    # Add value labels
    for bar, val in zip(bars, df_sorted["Neutral F1"]):
        plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{val:.3f}", va="center")
    
    plt.axvline(x=0.61, color="red", linestyle="--", label="Baseline (0.61)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "neutral_class_focus.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'neutral_class_focus.png')}")


def main():
    print("="*60)
    print("Collecting and Evaluating All Results")
    print("="*60)
    
    # Collect results
    results = collect_results()
    
    if not results:
        print("No results found. Run experiments first.")
        return
    
    print(f"\nFound {len(results)} experiment results")
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Print table
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    print(df.to_string(index=False))
    
    # Save to CSV
    csv_path = os.path.join(RESULTS_DIR, "comparison_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_comparison_bar_chart(df)
    plot_heatmap(df)
    plot_neutral_focus(df)
    
    # Find best model
    print("\n" + "="*60)
    print("BEST MODELS")
    print("="*60)
    
    best_weighted_f1 = df.loc[df["Weighted F1"].idxmax()]
    print(f"Best Weighted F1: {best_weighted_f1['Experiment']} ({best_weighted_f1['Weighted F1']:.4f})")
    
    if "Neutral F1" in df.columns:
        best_neutral = df.loc[df["Neutral F1"].idxmax()]
        print(f"Best Neutral F1: {best_neutral['Experiment']} ({best_neutral['Neutral F1']:.4f})")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Visualizations saved to: {OUTPUT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
