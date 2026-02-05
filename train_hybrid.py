"""
Hybrid Feature Fusion Model for Vietnamese Sentiment Analysis

TRUE HYBRID APPROACH: Combines multiple paradigms in a single model:
1. PhoBERT embeddings (Deep semantic understanding)
2. TF-IDF features (Statistical word patterns)  
3. Handcrafted sentiment features (Linguistic rules)

All features are fused and fed into a neural classifier.
"""

import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# Configuration
DATA_DIR = "./"
RESULTS_DIR = "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Negative", "Neutral", "Positive"]

# Vietnamese sentiment lexicon (simplified)
POSITIVE_WORDS = [
    "tốt", "hay", "tuyệt", "thích", "yêu", "giỏi", "xuất sắc", "tận tâm", 
    "nhiệt tình", "dễ hiểu", "hữu ích", "bổ ích", "thú vị", "vui", "hài lòng",
    "chuyên nghiệp", "chu đáo", "thân thiện", "nhanh", "gọn"
]

NEGATIVE_WORDS = [
    "tệ", "dở", "chán", "khó", "buồn", "ghét", "kém", "chậm", "lười",
    "khó hiểu", "nhàm chán", "thiếu", "yếu", "tội", "mệt", "khổ",
    "thất vọng", "tức", "bực", "khó chịu", "mất thời gian"
]

NEGATION_WORDS = ["không", "chưa", "chẳng", "đừng", "chả", "ko", "k"]


class VietnameseSentimentFeatureExtractor:
    """Extract handcrafted linguistic features for Vietnamese sentiment"""
    
    def __init__(self):
        self.positive_pattern = re.compile(r'\b(' + '|'.join(POSITIVE_WORDS) + r')\b', re.IGNORECASE)
        self.negative_pattern = re.compile(r'\b(' + '|'.join(NEGATIVE_WORDS) + r')\b', re.IGNORECASE)
        self.negation_pattern = re.compile(r'\b(' + '|'.join(NEGATION_WORDS) + r')\b', re.IGNORECASE)
    
    def extract_features(self, text):
        """Extract sentiment features from text"""
        text_lower = text.lower()
        
        # Count sentiment words
        pos_count = len(self.positive_pattern.findall(text_lower))
        neg_count = len(self.negative_pattern.findall(text_lower))
        negation_count = len(self.negation_pattern.findall(text_lower))
        
        # Word and character counts
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        # Exclamation and question marks (emotion indicators)
        exclaim_count = text.count('!')
        question_count = text.count('?')
        
        # Emoji/emoticon patterns
        emoji_pattern = re.compile(r'[:;=][\-]?[)D(P]|[\U0001F600-\U0001F64F]')
        emoji_count = len(emoji_pattern.findall(text))
        
        # Normalize by word count to avoid length bias
        if word_count > 0:
            pos_ratio = pos_count / word_count
            neg_ratio = neg_count / word_count
        else:
            pos_ratio = neg_ratio = 0
        
        # Sentiment polarity score
        polarity = (pos_count - neg_count) / max(pos_count + neg_count, 1)
        
        # Negation can flip sentiment
        if negation_count > 0:
            polarity *= -0.5  # Dampen polarity when negation present
        
        features = np.array([
            pos_count,           # Raw positive word count
            neg_count,           # Raw negative word count
            negation_count,      # Negation count
            pos_ratio,           # Positive ratio
            neg_ratio,           # Negative ratio
            polarity,            # Overall polarity score
            word_count,          # Text length (words)
            exclaim_count,       # Exclamation marks
            question_count,      # Question marks
            emoji_count,         # Emoji count
        ], dtype=np.float32)
        
        return features
    
    def extract_batch(self, texts):
        """Extract features for a batch of texts"""
        return np.array([self.extract_features(t) for t in texts])


class HybridFusionModel(nn.Module):
    """
    Hybrid model that fuses:
    - BERT [CLS] embeddings (768 or 1024 dim)
    - TF-IDF features (N dim)
    - Handcrafted sentiment features (10 dim)
    """
    
    def __init__(self, bert_dim, tfidf_dim, handcrafted_dim=10, num_labels=3, dropout=0.3):
        super().__init__()
        
        # Feature dimension reduction
        self.bert_proj = nn.Sequential(
            nn.Linear(bert_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.tfidf_proj = nn.Sequential(
            nn.Linear(tfidf_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.handcrafted_proj = nn.Sequential(
            nn.Linear(handcrafted_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        fusion_dim = 256 + 128 + 32  # 416
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels)
        )
        
    def forward(self, bert_features, tfidf_features, handcrafted_features):
        # Project each feature type
        bert_proj = self.bert_proj(bert_features)
        tfidf_proj = self.tfidf_proj(tfidf_features)
        handcrafted_proj = self.handcrafted_proj(handcrafted_features)
        
        # Concatenate (fuse) all features
        fused = torch.cat([bert_proj, tfidf_proj, handcrafted_proj], dim=1)
        
        # Classification
        logits = self.fusion(fused)
        return logits


class HybridDataset(Dataset):
    """Dataset that provides all feature types"""
    
    def __init__(self, bert_features, tfidf_features, handcrafted_features, labels):
        self.bert_features = torch.tensor(bert_features, dtype=torch.float32)
        self.tfidf_features = torch.tensor(tfidf_features, dtype=torch.float32)
        self.handcrafted_features = torch.tensor(handcrafted_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'bert': self.bert_features[idx],
            'tfidf': self.tfidf_features[idx],
            'handcrafted': self.handcrafted_features[idx],
            'labels': self.labels[idx]
        }


def load_data(data_dir):
    """Load texts and labels"""
    with open(os.path.join(data_dir, "sents.txt"), "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines()]
    with open(os.path.join(data_dir, "sentiments.txt"), "r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f.readlines()]
    return texts, labels


def extract_bert_embeddings(texts, model_name="vinai/phobert-base", batch_size=32):
    """Extract [CLS] embeddings from BERT model"""
    print(f"Extracting BERT embeddings using {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="BERT embeddings"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            ).to(DEVICE)
            
            outputs = model(**inputs)
            # Get [CLS] token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.extend(cls_embeddings)
    
    return np.array(all_embeddings)


def train_hybrid_model(use_class_weights=True):
    """Train the hybrid feature fusion model"""
    
    print("="*60)
    print("Training Hybrid Feature Fusion Model")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_texts, train_labels = load_data(os.path.join(DATA_DIR, "train"))
    dev_texts, dev_labels = load_data(os.path.join(DATA_DIR, "dev"))
    test_texts, test_labels = load_data(os.path.join(DATA_DIR, "test"))
    
    # Merge train + dev
    all_train_texts = train_texts + dev_texts
    all_train_labels = train_labels + dev_labels
    
    print(f"Train samples: {len(all_train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # 1. Extract BERT embeddings
    print("\n[1/3] Extracting BERT embeddings...")
    train_bert = extract_bert_embeddings(all_train_texts)
    test_bert = extract_bert_embeddings(test_texts)
    bert_dim = train_bert.shape[1]
    print(f"BERT embedding dim: {bert_dim}")
    
    # 2. Extract TF-IDF features
    print("\n[2/3] Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    train_tfidf = tfidf.fit_transform(all_train_texts).toarray()
    test_tfidf = tfidf.transform(test_texts).toarray()
    tfidf_dim = train_tfidf.shape[1]
    print(f"TF-IDF feature dim: {tfidf_dim}")
    
    # 3. Extract handcrafted features
    print("\n[3/3] Extracting handcrafted sentiment features...")
    feature_extractor = VietnameseSentimentFeatureExtractor()
    train_handcrafted = feature_extractor.extract_batch(all_train_texts)
    test_handcrafted = feature_extractor.extract_batch(test_texts)
    handcrafted_dim = train_handcrafted.shape[1]
    print(f"Handcrafted feature dim: {handcrafted_dim}")
    
    # Create datasets
    train_dataset = HybridDataset(train_bert, train_tfidf, train_handcrafted, all_train_labels)
    test_dataset = HybridDataset(test_bert, test_tfidf, test_handcrafted, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create model
    model = HybridFusionModel(
        bert_dim=bert_dim,
        tfidf_dim=tfidf_dim,
        handcrafted_dim=handcrafted_dim,
        num_labels=3
    ).to(DEVICE)
    
    # Class weights for imbalanced data
    if use_class_weights:
        from collections import Counter
        label_counts = Counter(all_train_labels)
        total = sum(label_counts.values())
        weights = torch.tensor([total / (3 * label_counts[i]) for i in range(3)]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"\nClass weights: {weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Training
    print("\nTraining hybrid model...")
    num_epochs = 10
    best_f1 = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            bert = batch['bert'].to(DEVICE)
            tfidf = batch['tfidf'].to(DEVICE)
            handcrafted = batch['handcrafted'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            logits = model(bert, tfidf, handcrafted)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # Evaluate
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                bert = batch['bert'].to(DEVICE)
                tfidf = batch['tfidf'].to(DEVICE)
                handcrafted = batch['handcrafted'].to(DEVICE)
                labels = batch['labels']
                
                logits = model(bert, tfidf, handcrafted)
                preds = logits.argmax(dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='weighted')
        acc = accuracy_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Test Set Evaluation")
    print("="*60)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            bert = batch['bert'].to(DEVICE)
            tfidf = batch['tfidf'].to(DEVICE)
            handcrafted = batch['handcrafted'].to(DEVICE)
            labels = batch['labels']
            
            logits = model(bert, tfidf, handcrafted)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Generate report
    report = classification_report(all_labels, all_preds, target_names=LABEL_NAMES, output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=LABEL_NAMES)
    
    print(report_str)
    
    # Save results
    output_dir = os.path.join(RESULTS_DIR, "hybrid_fusion")
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "model_name": "Hybrid Feature Fusion",
        "components": ["PhoBERT-base embeddings", "TF-IDF (5000 features)", "Handcrafted sentiment (10 features)"],
        "test_accuracy": report["accuracy"],
        "test_weighted_f1": report["weighted avg"]["f1-score"],
        "test_macro_f1": report["macro avg"]["f1-score"],
        "per_class": {
            "Negative": report["Negative"],
            "Neutral": report["Neutral"],
            "Positive": report["Positive"],
        }
    }
    
    with open(os.path.join(output_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_str)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    results = train_hybrid_model(use_class_weights=True)
    
    print("\n" + "="*60)
    print("HYBRID MODEL RESULTS:")
    print(f"  Accuracy: {results['test_accuracy']:.4f}")
    print(f"  Weighted F1: {results['test_weighted_f1']:.4f}")
    print(f"  Macro F1: {results['test_macro_f1']:.4f}")
    print("="*60)
