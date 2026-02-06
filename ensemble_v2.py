"""
Ensemble V2: Stacking Meta-Learner
Uses output probabilities from Base Models (Level 0) to train a Meta-Learner (Level 1).
Meta-Learner: Logistic Regression or XGBoost.
Training Strategy: Train Meta-Learner on DEV set, Evaluate on TEST set.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Config
DATA_DIR = "./"
RESULTS_DIR = "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Negative", "Neutral", "Positive"]

def load_data(split):
    sents_path = os.path.join(DATA_DIR, split, "sents.txt")
    sentiments_path = os.path.join(DATA_DIR, split, "sentiments.txt")
    with open(sents_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines()]
    with open(sentiments_path, "r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f.readlines()]
    return texts, np.array(labels)

def get_model_probs(model_name, texts, batch_size=32):
    """Get soft probabilities from a trained Transformer model"""
    print(f"Generating predictions for {model_name}...")
    model_dir = os.path.join(RESULTS_DIR, model_name)
    
    # 1. Load Tokenizer & Model
    # Try to find config to identify base model type
    try:
        # Heuristic: Check config.json or use known mapping
        if "phobert-base" in model_name: base_name = "vinai/phobert-base"
        elif "phobert-large" in model_name: base_name = "vinai/phobert-large"
        elif "visobert" in model_name: base_name = "uitnlp/visobert"
        elif "hybrid" in model_name: 
            print(f"Skipping {model_name} (Hybrid loading not supported in generic function)")
            return None
        else: base_name = "vinai/phobert-base" # Fallback
        
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        
        # Find checkpoint
        checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint")]
        if checkpoints:
            load_path = os.path.join(model_dir, sorted(checkpoints)[-1])
        else:
            load_path = model_dir # Maybe saved directly
            
        model = AutoModelForSequenceClassification.from_pretrained(load_path)
        model.to(DEVICE)
        model.eval()
        
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

    # 2. Inference
    all_probs = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            
    return np.concatenate(all_probs, axis=0)

def train_stacking_ensemble():
    print("="*60)
    print("Training Stacking Ensemble (V2)")
    print("="*60)
    
    # 1. Load Data
    dev_texts, dev_labels = load_data("dev")
    test_texts, test_labels = load_data("test")
    
    # 2. Select Base Models
    # Prioritize: weighted models, then standard
    candidates = [
        "phobert_large_weighted", 
        "phobert_base_weighted", 
        "visobert_weighted",
        "phobert_large", # fallback
        "phobert_base"   # fallback
    ]
    
    # Check what exists
    base_models = []
    for c in candidates:
        if os.path.exists(os.path.join(RESULTS_DIR, c)):
            base_models.append(c)
    
    # We remove duplicates (e.g. if we have both weighted and standard, keep weighted)
    # Actually Stacking benefits from diversity. Let's keep distinct variants.
    
    if len(base_models) < 2:
        print("Not enough models for Stacking. Need at least 2.")
        return

    print(f"Selected Base Models: {base_models}")
    
    # 3. Generate Level-1 Features (Probs)
    # X_meta_train = Probs on Dev Set
    # X_meta_test  = Probs on Test Set
    
    X_meta_train = []
    X_meta_test = []
    
    valid_models = []
    
    for model_name in base_models:
        # Dev Probs
        dev_probs = get_model_probs(model_name, dev_texts)
        if dev_probs is None: continue
        
        # Test Probs
        test_probs = get_model_probs(model_name, test_texts)
        if test_probs is None: continue
        
        X_meta_train.append(dev_probs)
        X_meta_test.append(test_probs)
        valid_models.append(model_name)
        
    if not valid_models:
        print("Failed to generate features for any model.")
        return

    # Stack features: [n_samples, n_models * 3]
    X_train_meta = np.concatenate(X_meta_train, axis=1)
    X_test_meta = np.concatenate(X_meta_test, axis=1)
    
    print(f"\nMeta-features shape: Train {X_train_meta.shape}, Test {X_test_meta.shape}")
    
    # 4. Train Meta-Learner
    print("\nTraining Meta-Learner (Logistic Regression)...")
    # Using Logistic Regression as it interprets weights well
    meta_clf = LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0)
    meta_clf.fit(X_train_meta, dev_labels)
    
    # Analyze Weights (Interpretation)
    print("\nMeta-Learner Weights (Importance):")
    feature_names = []
    for m in valid_models:
        feature_names.extend([f"{m}_Neg", f"{m}_Neu", f"{m}_Pos"])
        
    # We look at weights for each class
    # Just printing mean absolute weight for each model to see who is trusted
    weights = np.abs(meta_clf.coef_) # [3 classes, n_features]
    weights_per_model = {}
    
    start_idx = 0
    for m in valid_models:
        # 3 features per model
        model_w = weights[:, start_idx:start_idx+3].mean()
        weights_per_model[m] = model_w
        start_idx += 3
        
    for m, w in sorted(weights_per_model.items(), key=lambda x: x[1], reverse=True):
        print(f"  {m}: {w:.4f}")

    # 5. Evaluate on Test
    print("\nEvaluating Stacking Ensemble on Test Set...")
    final_preds = meta_clf.predict(X_test_meta)
    
    report = classification_report(test_labels, final_preds, target_names=LABEL_NAMES, output_dict=True)
    report_str = classification_report(test_labels, final_preds, target_names=LABEL_NAMES)
    
    print("\nStacking Results:")
    print(report_str)
    
    # Save Results
    save_path = os.path.join(RESULTS_DIR, "ensembles")
    os.makedirs(save_path, exist_ok=True)
    
    result_data = {
        "ensemble_name": "Stacking_Ensemble_V2",
        "base_models": valid_models,
        "test_accuracy": report["accuracy"],
        "test_weighted_f1": report["weighted avg"]["f1-score"],
        "test_macro_f1": report["macro avg"]["f1-score"],
        "per_class": {
            "Negative": report["Negative"],
            "Neutral": report["Neutral"],
            "Positive": report["Positive"],
        }
    }
    
    with open(os.path.join(save_path, "stacking_v2.json"), "w") as f:
        json.dump(result_data, f, indent=2)
        
    # Save Meta-model
    joblib.dump(meta_clf, os.path.join(save_path, "stacking_meta_model.joblib"))
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    train_stacking_ensemble()
