"""
Model configuration for Vietnamese Sentiment Analysis experiments.
Contains model names, hyperparameters, and training settings.
"""

# Available models for experiments
MODELS = {
    "phobert-large": {
        "pretrained_name": "vinai/phobert-large",
        "description": "PhoBERT Large - 370M parameters",
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1.5e-5,
    },
    "phobert-base": {
        "pretrained_name": "vinai/phobert-base",
        "description": "PhoBERT Base - 135M parameters",
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-5,
    },
    "visobert": {
        "pretrained_name": "uitnlp/visobert",
        "description": "ViSoBERT - Vietnamese Social Media BERT",
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-5,
    },
}

# Training configuration
TRAINING_CONFIG = {
    "num_epochs": 5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "max_length": 128,
    "num_labels": 3,
    "label_names": ["Negative", "Neutral", "Positive"],
}

# Paths
DATA_DIR = "./"
TRAIN_DIR = "train"
DEV_DIR = "dev"
TEST_DIR = "test"
RESULTS_DIR = "results"
