
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import Dataset

# Constants
DATA_DIR = "./"
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = "./results/checkpoint-4070" # Use the last checkpoint
OUTPUT_DIR = "./evaluation_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

class VSFC_Dataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.labels = []
        
        sents_path = os.path.join(data_dir, "sents.txt")
        sentiments_path = os.path.join(data_dir, "sentiments.txt")
        
        with open(sents_path, "r", encoding="utf-8") as f:
            self.texts = [line.strip() for line in f.readlines()]
            
        with open(sentiments_path, "r", encoding="utf-8") as f:
            self.labels = [int(line.strip()) for line in f.readlines()]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label)
        return item

def plot_confusion_matrix(y_true, y_pred, labels, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()

def main():
    print("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        # Fallback to base loading if needed (unlikely)
        return

    test_dataset = VSFC_Dataset(TEST_DIR, tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=32,
        no_cuda=False if torch.cuda.is_available() else True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args
    )
    
    print("Predicting on Test set...")
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    
    target_names = ["Negative", "Neutral", "Positive"]
    
    # 1. Confusion Matrix
    print("Generating Confusion Matrix...")
    plot_confusion_matrix(labels, preds, target_names, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    
    # 2. Classification Report
    print("Generating Report...")
    report_dict = classification_report(labels, preds, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(OUTPUT_DIR, "detailed_metrics.csv"))
    
    # 3. Accuracy Bar Plot
    # plt.figure(figsize=(6, 4))
    # sns.barplot(x=target_names, y=[report_dict[cls]['f1-score'] for cls in target_names])
    # plt.title("F1-Score per Class")
    # plt.ylabel("F1 Score")
    # plt.savefig(os.path.join(OUTPUT_DIR, "f1_scores.png"))
    # plt.close()
    
    print(f"Evaluation finished. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
