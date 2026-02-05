"""
Vietnamese Sentiment Analysis - Improved Training Script
Fixes data leakage and supports multiple models with proper evaluation.
"""

import os
import argparse
import torch
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from collections import Counter
import json

# Import config
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.config import MODELS, TRAINING_CONFIG, DATA_DIR, TRAIN_DIR, DEV_DIR, TEST_DIR, RESULTS_DIR


class VSFC_Dataset(Dataset):
    """Vietnamese Students' Feedback Corpus Dataset"""
    
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
            
        assert len(self.texts) == len(self.labels), f"Mismatch in {data_dir}"

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class WeightedTrainer(Trainer):
    """Custom Trainer with class-weighted loss for imbalanced data"""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        if self.class_weights is not None:
            loss_fct = CrossEntropyLoss(weight=self.class_weights.to(labels.device))
        else:
            loss_fct = CrossEntropyLoss()
            
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(pred):
    """Compute evaluation metrics"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    macro_f1 = precision_recall_fscore_support(labels, preds, average="macro")[2]
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "weighted_f1": f1,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall
    }


def compute_class_weights(labels):
    """Compute inverse frequency class weights"""
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    weights = torch.tensor([total / (len(label_counts) * label_counts[i]) for i in range(len(label_counts))])
    return weights


def train_model(model_name, use_weighted_loss=False, merge_train_dev=False):
    """Train a single model with proper train/val/test split"""
    
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Weighted Loss: {use_weighted_loss}")
    print(f"Merge Train+Dev: {merge_train_dev}")
    print(f"{'='*60}\n")
    
    # Get model config
    model_config = MODELS[model_name]
    pretrained_name = model_config["pretrained_name"]
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    
    # Load datasets
    print("Preparing datasets...")
    train_ds = VSFC_Dataset(os.path.join(DATA_DIR, TRAIN_DIR), tokenizer, TRAINING_CONFIG["max_length"])
    dev_ds = VSFC_Dataset(os.path.join(DATA_DIR, DEV_DIR), tokenizer, TRAINING_CONFIG["max_length"])
    test_ds = VSFC_Dataset(os.path.join(DATA_DIR, TEST_DIR), tokenizer, TRAINING_CONFIG["max_length"])
    
    # Proper split strategy
    if merge_train_dev:
        # Merge train+dev for training, but DON'T use test for validation
        # Instead, we train without early stopping based on validation
        train_dataset = torch.utils.data.ConcatDataset([train_ds, dev_ds])
        eval_dataset = None  # No validation during training
        all_labels = train_ds.labels + dev_ds.labels
    else:
        # Standard: train on train, validate on dev
        train_dataset = train_ds
        eval_dataset = dev_ds
        all_labels = train_ds.labels
    
    # Compute class weights if needed
    class_weights = None
    if use_weighted_loss:
        class_weights = compute_class_weights(all_labels)
        print(f"Class weights: {class_weights.tolist()}")
    
    # Load model
    print(f"Loading model: {pretrained_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_name, 
        num_labels=TRAINING_CONFIG["num_labels"]
    )
    
    # Output directory for this model
    output_dir = os.path.join(RESULTS_DIR, model_name.replace("-", "_"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=TRAINING_CONFIG["num_epochs"],
        per_device_train_batch_size=model_config["batch_size"],
        gradient_accumulation_steps=model_config["gradient_accumulation_steps"],
        per_device_eval_batch_size=16,
        learning_rate=model_config["learning_rate"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="weighted_f1" if eval_dataset else None,
        no_cuda=not torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    # Create trainer
    TrainerClass = WeightedTrainer if use_weighted_loss else Trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "compute_metrics": compute_metrics,
    }
    
    if use_weighted_loss:
        trainer_kwargs["class_weights"] = class_weights
        
    trainer = TrainerClass(**trainer_kwargs)
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate on test set (ONLY for final evaluation, not model selection)
    print("\nEvaluating on Test Set...")
    predictions = trainer.predict(test_ds)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    
    # Generate detailed report
    report = classification_report(
        labels, preds, 
        target_names=TRAINING_CONFIG["label_names"],
        output_dict=True
    )
    report_str = classification_report(
        labels, preds, 
        target_names=TRAINING_CONFIG["label_names"]
    )
    
    print("\nTest Set Results:")
    print(report_str)
    
    # Save results
    results = {
        "model_name": model_name,
        "pretrained_name": pretrained_name,
        "use_weighted_loss": use_weighted_loss,
        "merge_train_dev": merge_train_dev,
        "test_accuracy": report["accuracy"],
        "test_weighted_f1": report["weighted avg"]["f1-score"],
        "test_macro_f1": report["macro avg"]["f1-score"],
        "per_class": {
            "Negative": report["Negative"],
            "Neutral": report["Neutral"],
            "Positive": report["Positive"],
        }
    }
    
    # Save to JSON
    results_file = os.path.join(output_dir, "test_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save classification report
    report_file = os.path.join(output_dir, "classification_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_str)
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Vietnamese Sentiment Analysis Model")
    parser.add_argument("--model", type=str, default="phobert-large", 
                        choices=list(MODELS.keys()),
                        help="Model to train")
    parser.add_argument("--weighted-loss", action="store_true",
                        help="Use class-weighted loss")
    parser.add_argument("--merge-train-dev", action="store_true",
                        help="Merge train and dev sets for training")
    
    args = parser.parse_args()
    
    results = train_model(
        model_name=args.model,
        use_weighted_loss=args.weighted_loss,
        merge_train_dev=args.merge_train_dev
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print(f"  Accuracy: {results['test_accuracy']:.4f}")
    print(f"  Weighted F1: {results['test_weighted_f1']:.4f}")
    print(f"  Macro F1: {results['test_macro_f1']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
