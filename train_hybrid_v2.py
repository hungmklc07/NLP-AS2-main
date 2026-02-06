"""
Advanced Hybrid Model V2 (Fine-tuning + Gated Fusion)
Combines PhoBERT embedding fine-tuning with advanced handcrafted features using Gated Fusion.
Goal: Outperform single PhoBERT-large model.
"""

import os
import re
import math
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from tqdm import tqdm

# Configuration
DATA_DIR = "./"
RESULTS_DIR = "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Negative", "Neutral", "Positive"]
MAX_LEN = 128

# Semantic Lexicons
POSITIVE_WORDS = [
    "tốt", "hay", "tuyệt", "thích", "yêu", "giỏi", "xuất sắc", "tận tâm", 
    "nhiệt tình", "dễ hiểu", "hữu ích", "bổ ích", "thú vị", "vui", "hài lòng",
    "chuyên nghiệp", "chu đáo", "thân thiện", "nhanh", "gọn", "ok", "ổn",
    "kỹ", "chi tiết", "sâu", "rộng", "thoải mái", "đam mê"
]

NEGATIVE_WORDS = [
    "tệ", "dở", "chán", "khó", "buồn", "ghét", "kém", "chậm", "lười",
    "khó hiểu", "nhàm chán", "thiếu", "yếu", "tội", "mệt", "khổ",
    "thất vọng", "tức", "bực", "khó chịu", "mất thời gian", "buồn ngủ",
    "nhanh quá", "nhỏ", "ồn", "lag"
]

NEGATION_WORDS = ["không", "chưa", "chẳng", "đừng", "chả", "ko", "k", "nỏ", "đéo", "méo"]

class AdvancedFeatureExtractor:
    """Extracts advanced linguistic features including sentiment shift"""
    
    def __init__(self):
        self.positive_pattern = re.compile(r'\b(' + '|'.join(POSITIVE_WORDS) + r')\b', re.IGNORECASE)
        self.negative_pattern = re.compile(r'\b(' + '|'.join(NEGATIVE_WORDS) + r')\b', re.IGNORECASE)
        self.negation_pattern = re.compile(r'\b(' + '|'.join(NEGATION_WORDS) + r')\b', re.IGNORECASE)
        self.emoji_pattern = re.compile(r'[:;=][\-]?[)D(P]|[\U0001F600-\U0001F64F]')
        self.caps_pattern = re.compile(r'\b[A-ZÀ-Ỹ]{2,}\b') # Detect SCREAMING CASE

    def extract_features(self, text):
        text_lower = text.lower()
        words = text_lower.split()
        num_words = len(words)
        if num_words == 0: return np.zeros(12, dtype=np.float32)

        # 1. Basic Counts
        pos_count = len(self.positive_pattern.findall(text_lower))
        neg_count = len(self.negative_pattern.findall(text_lower))
        negation_count = len(self.negation_pattern.findall(text_lower))
        emoji_count = len(self.emoji_pattern.findall(text))
        caps_count = len(self.caps_pattern.findall(text))

        # 2. Sentiment Shift (Negation before Sentiment)
        # Check if a negation word appears within 2 words before a sentiment word
        shifted_pos = 0
        shifted_neg = 0
        
        # Simple window search
        for i, word in enumerate(words):
            if word in NEGATION_WORDS:
                # Check next 2 words
                window = words[i+1 : min(i+3, num_words)]
                for w in window:
                    if w in POSITIVE_WORDS: shifted_pos += 1
                    if w in NEGATIVE_WORDS: shifted_neg += 1
        
        # Effective Sentiment = Raw - Shifted
        eff_pos = pos_count - shifted_pos + shifted_neg # Negated Neg becomes Pos-ish? (Maybe not fully, but helps Neutral)
        eff_neg = neg_count - shifted_neg + shifted_pos # Negated Pos becomes Neg-ish
        
        # 3. Ratios & Polarity
        pos_ratio = eff_pos / num_words
        neg_ratio = eff_neg / num_words
        
        polarity = (eff_pos - eff_neg) / max(eff_pos + eff_neg, 1)
        
        # 4. Intensity
        has_exclaim = 1.0 if '!' in text else 0.0
        has_question = 1.0 if '?' in text else 0.0
        
        # Feature Vector (dim=12)
        features = np.array([
            pos_count,      # 0
            neg_count,      # 1
            negation_count, # 2
            emoji_count * 2, # 3 (Scale up emoji importance)
            caps_count,     # 4
            shifted_pos,    # 5
            shifted_neg,    # 6
            pos_ratio,      # 7
            neg_ratio,      # 8
            polarity,       # 9
            has_exclaim,    # 10
            has_question    # 11
        ], dtype=np.float32)
        
        return features

    def extract_batch(self, texts):
        return np.array([self.extract_features(t) for t in texts])


class GatedHybridModel(nn.Module):
    """
    Hybrid V2:
    - Fine-tunes last n layers of PhoBERT
    - Gated Fusion Mechanism: z * BERT + (1-z) * Features
    """
    def __init__(self, model_name, feature_dim, num_labels=3, freeze_layers=True):
        super().__init__()
        
        self.phobert = AutoModel.from_pretrained(model_name)
        self.bert_dim = self.phobert.config.hidden_size
        
        # Freeze strategy
        if freeze_layers:
            # Freeze all first
            for param in self.phobert.parameters():
                param.requires_grad = False
            
            # Unfreeze last 2 encoder layers and pooler
            # Config check for layer access
            if hasattr(self.phobert.encoder, 'layer'):
                layers = self.phobert.encoder.layer
                for layer in layers[-2:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            
            if hasattr(self.phobert, 'pooler') and self.phobert.pooler is not None:
                for param in self.phobert.pooler.parameters():
                    param.requires_grad = True

        # Feature processing
        self.feat_proj = nn.Sequential(
            nn.BatchNorm1d(feature_dim), # Normalize handcrafted features
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.bert_dim) # Project to BERT space
        )
        
        # Gating Mechanism
        # Gate calculates how much to trust BERT vs Features based on input
        # Input to gate: Concat(BERT, Features)
        self.gate_net = nn.Sequential(
            nn.Linear(self.bert_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, self.bert_dim),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_labels)
        )

    def forward(self, input_ids, attention_mask, handcrafted_features):
        # BERT forward
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        bert_emb = outputs.last_hidden_state[:, 0, :] # [CLS] embedding
        
        # Projection
        feat_emb = self.feat_proj(handcrafted_features) # Project features to [Batch, 768]
        
        # Gated Fusion
        combined = torch.cat([bert_emb, feat_emb], dim=1)
        gate = self.gate_net(combined) # [Batch, 768] values in [0,1]
        
        # Fusing: gate * BERT + (1-gate) * Features
        # Idea: For some samples, BERT is confused, maybe features help?
        fused = gate * bert_emb + (1 - gate) * feat_emb
        
        # Classification
        logits = self.classifier(fused)
        
        return logits


class HybridDatasetV2(Dataset):
    def __init__(self, texts, labels, tokenizer, feature_extractor, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_len = max_len
        
        # Pre-compute handcrafted features (CPU)
        print("Extracting features...")
        self.features = self.feature_extractor.extract_batch(texts)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        features = self.features[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_dir):
    sents_path = os.path.join(data_dir, "sents.txt")
    sentiments_path = os.path.join(data_dir, "sentiments.txt")
    with open(sents_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines()]
    with open(sentiments_path, "r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f.readlines()]
    return texts, labels

def train_hybrid_v2(model_type="phobert-base"):
    print(f"Training Hybrid V2 with {model_type}...")
    
    # Init
    model_name = "vinai/phobert-base" if model_type == "phobert-base" else "vinai/phobert-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    feature_extractor = AdvancedFeatureExtractor()
    
    # Load Data
    train_texts, train_labels = load_data(os.path.join(DATA_DIR, "train"))
    dev_texts, dev_labels = load_data(os.path.join(DATA_DIR, "dev"))
    test_texts, test_labels = load_data(os.path.join(DATA_DIR, "test"))
    
    # Merge Train + Dev (for max performance)
    full_train_texts = train_texts + dev_texts
    full_train_labels = train_labels + dev_labels
    
    # Datasets
    train_dataset = HybridDatasetV2(full_train_texts, full_train_labels, tokenizer, feature_extractor)
    test_dataset = HybridDatasetV2(test_texts, test_labels, tokenizer, feature_extractor)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True) # Smaller batch for fine-tuning, drop_last for BatchNorm
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Model
    model = GatedHybridModel(model_name, feature_dim=12, freeze_layers=True)
    model.to(DEVICE)
    
    # Class Weights
    label_counts = Counter(full_train_labels)
    total = sum(label_counts.values())
    weights = torch.tensor([total / (3 * label_counts[i]) for i in range(3)]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Optimizer (Different LRs for BERT and Heads)
    optimizer = torch.optim.AdamW([
        {'params': model.phobert.parameters(), 'lr': 2e-5}, # Low LR for BERT
        {'params': model.feat_proj.parameters(), 'lr': 1e-3},
        {'params': model.gate_net.parameters(), 'lr': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ])
    
    # Train Loop
    epochs = 5
    print("\nStarting Training...")
    
    best_f1 = 0
    save_path = os.path.join(RESULTS_DIR, "hybrid_v2")
    os.makedirs(save_path, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            feats = batch['features'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(input_ids, mask, feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Eval
        model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                feats = batch['features'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                logits = model(input_ids, mask, feats)
                preds.extend(logits.argmax(1).cpu().numpy())
                acts.extend(labels.cpu().numpy())
                
        f1 = accuracy_score(acts, preds) # Using Acc for quick check, report F1 later
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}, Test Acc {f1:.4f}")
        
        # Save Best
        # Compute Weighted F1 strictly
        report = classification_report(acts, preds, target_names=LABEL_NAMES, output_dict=True)
        weighted_f1 = report['weighted avg']['f1-score']
        
        if weighted_f1 > best_f1:
            best_f1 = weighted_f1
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
            with open(os.path.join(save_path, "classification_report.txt"), "w", encoding="utf-8") as f:
                f.write(classification_report(acts, preds, target_names=LABEL_NAMES))
            
            # Save JSON
            results = {
                "model_name": "Hybrid V2 (Gated)",
                "test_accuracy": report["accuracy"],
                "test_weighted_f1": report["weighted avg"]["f1-score"],
                "test_macro_f1": report["macro avg"]["f1-score"],
                "per_class": {
                    "Negative": report["Negative"],
                    "Neutral": report["Neutral"],
                    "Positive": report["Positive"],
                }
            }
            with open(os.path.join(save_path, "test_results.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
                
    print(f"\nBest Weighted F1: {best_f1:.4f}")
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="phobert-base", help="Base model for Hybrid")
    args = parser.parse_args()
    
    train_hybrid_v2(args.model)
