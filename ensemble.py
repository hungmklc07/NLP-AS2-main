"""
Ensemble Methods for Vietnamese Sentiment Analysis
Combines predictions from multiple models for improved performance.
"""

import os
import json
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

# Configuration
RESULTS_DIR = "results"
DATA_DIR = "./"
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


def load_test_data():
    """Load test data"""
    sents_path = os.path.join(DATA_DIR, "test", "sents.txt")
    sentiments_path = os.path.join(DATA_DIR, "test", "sentiments.txt")
    
    with open(sents_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines()]
    
    with open(sentiments_path, "r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f.readlines()]
    
    return texts, labels


def get_transformer_predictions(model_name, pretrained_name, texts, batch_size=32):
    """Get predictions and probabilities from a transformer model"""
    print(f"Loading {model_name}...")
    
    model_dir = os.path.join(RESULTS_DIR, model_name)
    
    # Try to load from checkpoint
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint")]
    if checkpoints:
        model_path = os.path.join(model_dir, sorted(checkpoints)[-1])
    else:
        model_path = model_dir
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = probs.argmax(dim=-1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return np.array(all_preds), np.array(all_probs)


def get_baseline_predictions(model_name, texts):
    """Get predictions from baseline ML model"""
    print(f"Loading {model_name}...")
    
    model_dir = os.path.join(RESULTS_DIR, model_name)
    model_path = os.path.join(model_dir, "model.joblib")
    
    pipeline = joblib.load(model_path)
    
    preds = pipeline.predict(texts)
    
    # Get probabilities if available
    if hasattr(pipeline.named_steps['clf'], 'predict_proba'):
        probs = pipeline.predict_proba(texts)
    else:
        # For SVM, use decision function and softmax
        decision = pipeline.decision_function(texts)
        probs = np.exp(decision) / np.exp(decision).sum(axis=1, keepdims=True)
    
    return np.array(preds), np.array(probs)


def majority_voting(predictions_list):
    """Hard voting: majority vote from multiple models"""
    stacked = np.stack(predictions_list, axis=1)
    final_preds = stats.mode(stacked, axis=1, keepdims=False)[0]
    return final_preds


def soft_voting(probabilities_list, weights=None):
    """Soft voting: weighted average of probabilities"""
    if weights is None:
        weights = [1.0 / len(probabilities_list)] * len(probabilities_list)
    
    weighted_probs = np.zeros_like(probabilities_list[0])
    for probs, weight in zip(probabilities_list, weights):
        weighted_probs += weight * probs
    
    final_preds = weighted_probs.argmax(axis=1)
    return final_preds, weighted_probs


def evaluate_ensemble(preds, labels, ensemble_name):
    """Evaluate and save ensemble results"""
    report = classification_report(
        labels, preds,
        target_names=LABEL_NAMES,
        output_dict=True
    )
    report_str = classification_report(
        labels, preds,
        target_names=LABEL_NAMES
    )
    
    print(f"\n{ensemble_name} Results:")
    print(report_str)
    
    # Save results
    output_dir = os.path.join(RESULTS_DIR, "ensembles")
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "ensemble_name": ensemble_name,
        "test_accuracy": report["accuracy"],
        "test_weighted_f1": report["weighted avg"]["f1-score"],
        "test_macro_f1": report["macro avg"]["f1-score"],
        "per_class": {
            "Negative": report["Negative"],
            "Neutral": report["Neutral"],
            "Positive": report["Positive"],
        }
    }
    
    with open(os.path.join(output_dir, f"{ensemble_name.lower().replace(' ', '_')}.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    print("="*60)
    print("Ensemble Predictions")
    print("="*60)
    
    # Load test data
    texts, labels = load_test_data()
    labels = np.array(labels)
    print(f"Test samples: {len(texts)}")
    
    # Collect predictions from available models
    all_preds = {}
    all_probs = {}
    
    # Check which models are available
    available_models = []
    
    # Transformer models
    transformer_models = [
        ("phobert_large", "vinai/phobert-large"),
        ("phobert_base", "vinai/phobert-base"),
        ("visobert", "uitnlp/visobert"),
    ]
    
    for model_name, pretrained_name in transformer_models:
        model_dir = os.path.join(RESULTS_DIR, model_name)
        if os.path.exists(model_dir):
            try:
                preds, probs = get_transformer_predictions(model_name, pretrained_name, texts)
                all_preds[model_name] = preds
                all_probs[model_name] = probs
                available_models.append(model_name)
                print(f"  Loaded {model_name}: Accuracy = {accuracy_score(labels, preds):.4f}")
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")
    
    # Baseline models
    baseline_models = ["svm_baseline", "logistic_baseline"]
    
    for model_name in baseline_models:
        model_dir = os.path.join(RESULTS_DIR, model_name)
        if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "model.joblib")):
            try:
                preds, probs = get_baseline_predictions(model_name, texts)
                all_preds[model_name] = preds
                all_probs[model_name] = probs
                available_models.append(model_name)
                print(f"  Loaded {model_name}: Accuracy = {accuracy_score(labels, preds):.4f}")
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")
    
    if len(available_models) < 2:
        print("\nNeed at least 2 models for ensemble. Train more models first.")
        return
    
    print(f"\nAvailable models for ensemble: {available_models}")
    
    # Ensemble 1: Majority Voting (all models)
    print("\n" + "-"*40)
    preds_list = [all_preds[m] for m in available_models]
    voting_preds = majority_voting(preds_list)
    evaluate_ensemble(voting_preds, labels, "Majority_Voting_All")
    
    # Ensemble 2: Soft Voting (all models with equal weights)
    probs_list = [all_probs[m] for m in available_models]
    soft_preds, _ = soft_voting(probs_list)
    evaluate_ensemble(soft_preds, labels, "Soft_Voting_All")
    
    # Ensemble 3: Transformer-only ensemble (if available)
    transformer_available = [m for m in available_models if m in ["phobert_large", "phobert_base", "visobert"]]
    if len(transformer_available) >= 2:
        probs_list = [all_probs[m] for m in transformer_available]
        # Weight by expected performance (PhoBERT-large > ViSoBERT > PhoBERT-base)
        weights = {"phobert_large": 0.5, "phobert_base": 0.2, "visobert": 0.3}
        w = [weights.get(m, 1.0/len(transformer_available)) for m in transformer_available]
        w = [x/sum(w) for x in w]  # Normalize
        
        weighted_preds, _ = soft_voting(probs_list, weights=w)
        evaluate_ensemble(weighted_preds, labels, "Weighted_Transformer_Ensemble")
    
    print("\n" + "="*60)
    print("Ensemble results saved to: results/ensembles/")
    print("="*60)


if __name__ == "__main__":
    main()
