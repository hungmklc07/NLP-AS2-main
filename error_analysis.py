"""
Error Analysis Script for PhoBERT-large
Analyzes misclassified examples and generates detailed error report.
Run on Colab after training.
"""

import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Config
DATA_DIR = "./"
RESULTS_DIR = "results"
MODEL_NAME = "phobert_large"
BASE_MODEL = "vinai/phobert-large"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


def load_test_data():
    """Load test set"""
    sents_path = os.path.join(DATA_DIR, "test", "sents.txt")
    labels_path = os.path.join(DATA_DIR, "test", "sentiments.txt")
    
    with open(sents_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines()]
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f.readlines()]
    
    return texts, labels


def load_model():
    """Load trained PhoBERT-large model"""
    model_dir = os.path.join(RESULTS_DIR, MODEL_NAME)
    
    # Find checkpoint
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint")]
    if checkpoints:
        load_path = os.path.join(model_dir, sorted(checkpoints)[-1])
    else:
        load_path = model_dir
    
    print(f"Loading model from: {load_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(load_path)
    model.to(DEVICE)
    model.eval()
    
    return tokenizer, model


def predict(texts, tokenizer, model, batch_size=32):
    """Get predictions for all texts"""
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            ).to(DEVICE)
            
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = probs.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (i // batch_size) % 10 == 0:
                print(f"Processed {i + len(batch)}/{len(texts)}")
    
    return all_preds, all_probs


def analyze_errors(texts, labels, preds, probs):
    """Analyze misclassified examples"""
    
    errors = []
    for i, (text, true_label, pred_label, prob) in enumerate(zip(texts, labels, preds, probs)):
        if true_label != pred_label:
            errors.append({
                "id": i,
                "text": text,
                "true_label": LABEL_NAMES[true_label],
                "pred_label": LABEL_NAMES[pred_label],
                "confidence": prob[pred_label],
                "true_prob": prob[true_label],
            })
    
    return errors


def generate_report(texts, labels, preds, probs, errors):
    """Generate comprehensive error analysis report"""
    
    output_dir = os.path.join(RESULTS_DIR, "error_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - PhoBERT-large')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")
    
    # 2. Error Statistics
    error_types = Counter()
    for e in errors:
        key = f"{e['true_label']} → {e['pred_label']}"
        error_types[key] += 1
    
    # 3. Generate Markdown Report
    report = []
    report.append("# Phân tích lỗi chi tiết - PhoBERT-large\n")
    report.append(f"**Tổng số mẫu test:** {len(texts)}\n")
    report.append(f"**Số mẫu đúng:** {len(texts) - len(errors)}\n")
    report.append(f"**Số mẫu sai:** {len(errors)}\n")
    report.append(f"**Accuracy:** {(len(texts) - len(errors)) / len(texts) * 100:.2f}%\n\n")
    
    # Confusion Matrix
    report.append("## Confusion Matrix\n")
    report.append("![Confusion Matrix](confusion_matrix.png)\n\n")
    
    # Error breakdown
    report.append("## Phân loại lỗi theo loại\n")
    report.append("| Loại lỗi | Số lượng | Tỷ lệ |\n")
    report.append("|----------|----------|-------|\n")
    for error_type, count in error_types.most_common():
        pct = count / len(errors) * 100
        report.append(f"| {error_type} | {count} | {pct:.1f}% |\n")
    report.append("\n")
    
    # Sample errors by type
    report.append("## Ví dụ các câu bị phân loại sai\n\n")
    
    for error_type in ["Neutral → Positive", "Neutral → Negative", 
                       "Positive → Negative", "Negative → Positive"]:
        samples = [e for e in errors if f"{e['true_label']} → {e['pred_label']}" == error_type][:5]
        if samples:
            report.append(f"### {error_type}\n")
            report.append("| Câu | Confidence | Phân tích |\n")
            report.append("|-----|------------|------------|\n")
            for s in samples:
                text = s['text'][:80] + "..." if len(s['text']) > 80 else s['text']
                text = text.replace("|", "\\|")
                report.append(f"| {text} | {s['confidence']:.2f} | - |\n")
            report.append("\n")
    
    # High confidence errors (model rất tự tin nhưng sai)
    report.append("## Lỗi với độ tự tin cao (>0.9)\n")
    report.append("*Model rất tự tin nhưng sai - Cần phân tích kỹ*\n\n")
    high_conf_errors = [e for e in errors if e['confidence'] > 0.9]
    report.append(f"**Số lượng:** {len(high_conf_errors)}\n\n")
    
    if high_conf_errors[:10]:
        report.append("| Câu | Thật | Dự đoán | Confidence |\n")
        report.append("|-----|------|---------|------------|\n")
        for e in high_conf_errors[:10]:
            text = e['text'][:60] + "..." if len(e['text']) > 60 else e['text']
            text = text.replace("|", "\\|")
            report.append(f"| {text} | {e['true_label']} | {e['pred_label']} | {e['confidence']:.2f} |\n")
    report.append("\n")
    
    # Neutral analysis
    report.append("## Phân tích lớp Neutral\n")
    neutral_errors = [e for e in errors if e['true_label'] == 'Neutral']
    report.append(f"**Tổng Neutral trong test:** {sum(1 for l in labels if l == 1)}\n")
    report.append(f"**Neutral bị nhầm:** {len(neutral_errors)}\n\n")
    
    if neutral_errors[:10]:
        report.append("### Các câu Neutral bị nhầm\n")
        report.append("| Câu | Bị nhầm thành | Confidence |\n")
        report.append("|-----|---------------|------------|\n")
        for e in neutral_errors[:10]:
            text = e['text'][:70] + "..." if len(e['text']) > 70 else e['text']
            text = text.replace("|", "\\|")
            report.append(f"| {text} | {e['pred_label']} | {e['confidence']:.2f} |\n")
    
    # Save report
    report_path = os.path.join(output_dir, "error_analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"Saved: {report_path}")
    
    # Save all errors to CSV
    errors_df = pd.DataFrame(errors)
    errors_df.to_csv(os.path.join(output_dir, "all_errors.csv"), index=False, encoding="utf-8-sig")
    print("Saved: all_errors.csv")
    
    return output_dir


def main():
    print("="*60)
    print("Error Analysis for PhoBERT-large")
    print("="*60)
    
    # Load data
    print("\nLoading test data...")
    texts, labels = load_test_data()
    print(f"Test samples: {len(texts)}")
    
    # Load model
    print("\nLoading model...")
    tokenizer, model = load_model()
    
    # Predict
    print("\nGenerating predictions...")
    preds, probs = predict(texts, tokenizer, model)
    
    # Analyze errors
    print("\nAnalyzing errors...")
    errors = analyze_errors(texts, labels, preds, probs)
    print(f"Total errors: {len(errors)}")
    
    # Generate report
    print("\nGenerating report...")
    output_dir = generate_report(texts, labels, preds, probs, errors)
    
    print("\n" + "="*60)
    print(f"Error analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
