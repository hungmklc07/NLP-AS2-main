"""
Traditional ML Baseline for Vietnamese Sentiment Analysis
Uses TF-IDF + SVM/Logistic Regression for comparison with deep learning models.
"""

import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Configuration
DATA_DIR = "./"
RESULTS_DIR = "results"
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


def load_data(data_dir):
    """Load texts and labels from data directory"""
    sents_path = os.path.join(data_dir, "sents.txt")
    sentiments_path = os.path.join(data_dir, "sentiments.txt")
    
    with open(sents_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines()]
    
    with open(sentiments_path, "r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f.readlines()]
    
    return texts, labels


def train_baseline(model_type="svm", use_balanced=True):
    """Train traditional ML baseline model"""
    
    print(f"\n{'='*60}")
    print(f"Training Baseline: {model_type.upper()}")
    print(f"Class Balanced: {use_balanced}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    train_texts, train_labels = load_data(os.path.join(DATA_DIR, "train"))
    dev_texts, dev_labels = load_data(os.path.join(DATA_DIR, "dev"))
    test_texts, test_labels = load_data(os.path.join(DATA_DIR, "test"))
    
    # Merge train + dev for training
    all_train_texts = train_texts + dev_texts
    all_train_labels = train_labels + dev_labels
    
    print(f"Train samples: {len(all_train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # Create pipeline
    if model_type == "svm":
        classifier = LinearSVC(
            class_weight='balanced' if use_balanced else None,
            max_iter=10000,
            random_state=42
        )
    else:  # logistic regression
        classifier = LogisticRegression(
            class_weight='balanced' if use_balanced else None,
            max_iter=1000,
            random_state=42,
            multi_class='multinomial'
        )
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            min_df=2,
            sublinear_tf=True
        )),
        ('clf', classifier)
    ])
    
    # Train
    print("Training model...")
    pipeline.fit(all_train_texts, all_train_labels)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_preds = pipeline.predict(test_texts)
    
    # Generate report
    report = classification_report(
        test_labels, test_preds,
        target_names=LABEL_NAMES,
        output_dict=True
    )
    report_str = classification_report(
        test_labels, test_preds,
        target_names=LABEL_NAMES
    )
    
    print("\nTest Set Results:")
    print(report_str)
    
    # Save results
    output_dir = os.path.join(RESULTS_DIR, f"{model_type}_baseline")
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "model_type": model_type,
        "use_balanced": use_balanced,
        "test_accuracy": report["accuracy"],
        "test_weighted_f1": report["weighted avg"]["f1-score"],
        "test_macro_f1": report["macro avg"]["f1-score"],
        "per_class": {
            "Negative": report["Negative"],
            "Neutral": report["Neutral"],
            "Positive": report["Positive"],
        }
    }
    
    # Save JSON results
    with open(os.path.join(output_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save classification report
    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_str)
    
    # Save model
    joblib.dump(pipeline, os.path.join(output_dir, "model.joblib"))
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train baseline ML model")
    parser.add_argument("--model", type=str, default="svm",
                        choices=["svm", "logistic"],
                        help="Model type")
    parser.add_argument("--no-balanced", action="store_true",
                        help="Disable class balancing")
    
    args = parser.parse_args()
    
    results = train_baseline(
        model_type=args.model,
        use_balanced=not args.no_balanced
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print(f"  Accuracy: {results['test_accuracy']:.4f}")
    print(f"  Weighted F1: {results['test_weighted_f1']:.4f}")
    print(f"  Macro F1: {results['test_macro_f1']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
