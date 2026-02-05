<p align="center">
  <h1 align="center">🇻🇳 Vietnamese Sentiment Analysis with PhoBERT</h1>
  <p align="center">
    <strong>Fine-tuned PhoBERT Large for Student Feedback Classification</strong>
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-results">Results</a> •
    <a href="#-model-architecture">Architecture</a> •
    <a href="#-dataset">Dataset</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-94%25-brightgreen?style=for-the-badge" alt="Accuracy">
  <img src="https://img.shields.io/badge/Weighted_F1-93%25-blue?style=for-the-badge" alt="F1 Score">
  <img src="https://img.shields.io/badge/Model-PhoBERT_Large-orange?style=for-the-badge" alt="Model">
  <img src="https://img.shields.io/badge/License-Apache_2.0-red?style=for-the-badge" alt="License">
</p>

---

## 📋 Overview

This project implements a **state-of-the-art Vietnamese Sentiment Analysis** system using PhoBERT Large, achieving **94% accuracy** on the UIT-VSFC (Vietnamese Students' Feedback Corpus) dataset. The model classifies student feedback into three sentiment categories: **Negative**, **Neutral**, and **Positive**.

### ✨ Key Features

- 🚀 **High Performance**: 94% accuracy with 0.95 F1-score on Positive/Negative classes
- 🔥 **PhoBERT Large**: Leverages 370M parameter Vietnamese language model
- 📊 **Semantic Preprocessing**: Custom text normalization for Vietnamese student feedback
- ⚡ **Optimized Training**: Mixed precision (FP16) + gradient accumulation for efficient GPU usage

---

## 📊 Results

### Performance Metrics

| Class | Precision | Recall | F1-Score | Support |
|:------|:---------:|:------:|:--------:|:-------:|
| **Negative** | 0.95 | 0.96 | **0.95** | 1,409 |
| **Neutral** | 0.71 | 0.54 | 0.61 | 167 |
| **Positive** | 0.95 | 0.96 | **0.95** | 1,590 |
| | | | | |
| **Accuracy** | | | 0.94 | 3,166 |
| **Weighted Avg** | 0.93 | 0.94 | **0.93** | 3,166 |

### Confusion Matrix

<p align="center">
  <img src="./evaluation_plots/confusion_matrix.png" alt="Confusion Matrix" width="500">
</p>

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.9+
pip install torch transformers scikit-learn pandas
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/NLP-AS2.git
cd NLP-AS2

# If optimizer.pt is split, reassemble it
# Extract part1.zip in the results/ directory
```

### Training

```bash
# 1. Preprocess the data (normalize special tokens)
python preprocess.py

# 2. Train the model
python train.py
```

### Project Structure

```
NLP-AS2/
├── 📂 train/                  # Training data
│   ├── sents.txt             # Sentences
│   ├── sentiments.txt        # Labels (0: Neg, 1: Neu, 2: Pos)
│   └── topics.txt            # Topic labels
├── 📂 dev/                    # Development data
├── 📂 test/                   # Test data
├── 📂 results/                # Model checkpoints
├── 📂 evaluation_plots/       # Visualization outputs
├── 🐍 train.py               # Main training script
├── 🐍 preprocess.py          # Data preprocessing
├── 🐍 create_plots.py        # Generate evaluation plots
├── 📄 classification_report.txt  # Test results
└── 📄 Final_Report.md        # Detailed report (Vietnamese)
```

---

## 🏗️ Model Architecture

### PhoBERT Large

We use **[PhoBERT Large](https://github.com/VinAIResearch/PhoBERT)** (`vinai/phobert-large`) as the backbone:

- **Architecture**: RoBERTa-based Transformer
- **Parameters**: 370 million
- **Pre-training Data**: 20GB Vietnamese text
- **Fine-tuning Task**: 3-class sentiment classification

### Training Configuration

| Hyperparameter | Value |
|:---------------|:------|
| Learning Rate | 1.5e-5 |
| Batch Size | 4 (effective: 16 with grad accum) |
| Epochs | 5 |
| Warmup Steps | 500 |
| Weight Decay | 0.01 |
| Optimizer | AdamW |
| Mixed Precision | FP16 |

---

## 📚 Dataset

### UIT-VSFC (Vietnamese Students' Feedback Corpus)

| Split | Samples |
|:------|:-------:|
| Train | ~11,000 |
| Dev | ~3,000 |
| Test | 3,166 |

**Note**: We merge Train + Dev for final training to maximize performance.

### Data Preprocessing

Special tokens in the dataset are normalized to meaningful Vietnamese text:

```python
# Example transformations
"colonlove"  → "yêu thích"   # love emoji → "like/love"
"colonsmile" → "vui vẻ"      # smile emoji → "happy"
"colonsad"   → "buồn"        # sad emoji → "sad"
```

---

## 📈 Analysis

### Strengths

✅ **Excellent Performance on Major Classes**: F1-Score of 0.95 for both Positive and Negative  
✅ **Strong Vietnamese Understanding**: PhoBERT's pre-training on Vietnamese corpus provides deep semantic understanding  
✅ **Efficient Training**: FP16 + gradient accumulation enables training on consumer GPUs  

### Limitations

⚠️ **Neutral Class Challenge**: F1-Score of 0.61 due to:
- Severe class imbalance (only 5% of test data)
- Inherent ambiguity in neutral feedback ("The teacher is enthusiastic but speaks too fast")

### Future Improvements

- [ ] Class-weighted loss function
- [ ] Data augmentation for minority class
- [ ] Ensemble with other Vietnamese models (VisoNLU, ViT5)

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[VinAI Research](https://github.com/VinAIResearch)** for PhoBERT
- **UIT-NLP** for the VSFC dataset
- **Hugging Face** for the Transformers library

---

<p align="center">
  Made with ❤️ by dxpawn
</p>
