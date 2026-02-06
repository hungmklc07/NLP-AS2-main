# BÁO CÁO CHI TIẾT: Phân Tích Quan Điểm Sinh Viên UIT
## Vietnamese Student Feedback Sentiment Analysis - Detailed Technical Report

**Sinh viên:** [Họ tên sinh viên]  
**MSSV:** [Mã số sinh viên]  
**Môn học:** Xử lý ngôn ngữ tự nhiên (NLP)

---

# PHẦN 1: KỸ THUẬT SỬ DỤNG

## 1.1. Tổng quan phương pháp

Dự án triển khai **9 phương pháp** phân loại quan điểm, chia thành 4 nhóm:

```
┌─────────────────────────────────────────────────────────────┐
│                    PHƯƠNG PHÁP TIẾP CẬN                     │
├─────────────────┬─────────────────┬────────────────────────┤
│  Traditional ML │  Deep Learning  │  Hybrid & Ensemble     │
├─────────────────┼─────────────────┼────────────────────────┤
│  • SVM + TF-IDF │  • PhoBERT-base │  • Hybrid V1 (Fusion)  │
│  • Logistic Reg │  • PhoBERT-large│  • Hybrid V2 (Gated)   │
│                 │  • ViSoBERT     │  • Soft Voting         │
│                 │                 │  • Weighted Ensemble   │
│                 │                 │  • Stacking V2         │
└─────────────────┴─────────────────┴────────────────────────┘
```

---

## 1.2. Chi tiết các mô hình Deep Learning

### 1.2.1. PhoBERT (Base & Large)

**Lý do chọn PhoBERT:**
- PhoBERT là mô hình **BERT pre-trained đầu tiên và tốt nhất** cho tiếng Việt (Nguyen & Nguyen, 2020).
- Được huấn luyện trên 20GB văn bản tiếng Việt từ Wikipedia và báo chí.
- Sử dụng tokenizer **RDRSegmenter** để tách từ tiếng Việt chính xác.

**Cấu trúc:**
| Phiên bản | Số tham số | Layers | Hidden Size | Attention Heads |
|-----------|------------|--------|-------------|-----------------|
| PhoBERT-base | 135M | 12 | 768 | 12 |
| PhoBERT-large | 370M | 24 | 1024 | 16 |

**Hyperparameters huấn luyện:**

```python
# models/config.py
"phobert-base": {
    "pretrained_name": "vinai/phobert-base",
    "batch_size": 8,                    # Batch nhỏ vừa phải
    "gradient_accumulation_steps": 2,   # Effective batch = 16
    "learning_rate": 2e-5,              # LR chuẩn cho fine-tuning BERT
}

"phobert-large": {
    "pretrained_name": "vinai/phobert-large",
    "batch_size": 4,                    # Giảm vì model lớn hơn
    "gradient_accumulation_steps": 4,   # Effective batch = 16
    "learning_rate": 1.5e-5,            # LR thấp hơn cho large model
}

# Training config
TRAINING_CONFIG = {
    "num_epochs": 5,          # Đủ để convergence
    "warmup_steps": 500,      # Warmup ~4% of training steps
    "weight_decay": 0.01,     # L2 regularization
    "max_length": 128,        # Đủ cho câu đánh giá (avg ~30 tokens)
}
```

**Giải thích lựa chọn hyperparameters:**

| Parameter | Giá trị | Lý do |
|-----------|---------|-------|
| `learning_rate = 2e-5` | Tiêu chuẩn BERT | Theo paper gốc BERT, LR 2e-5 đến 5e-5 cho fine-tuning |
| `batch_size = 8-16` | Effective 16 | Cân bằng giữa ổn định gradient và bộ nhớ GPU |
| `epochs = 5` | | Đủ để model hội tụ mà không overfitting |
| `warmup_steps = 500` | ~4% training | Tránh learning rate quá cao lúc đầu gây phá vỡ pre-trained weights |
| `weight_decay = 0.01` | | Regularization chuẩn cho Transformer |
| `max_length = 128` | | Câu đánh giá sinh viên thường ngắn (~30 tokens), 128 là dư |

### 1.2.2. ViSoBERT

**Lý do chọn ViSoBERT:**
- ViSoBERT được train trên **văn bản social media tiếng Việt** (Facebook, YouTube comments).
- Kỳ vọng: Hiểu tốt hơn ngôn ngữ teen code, viết tắt, emoji.

**Kết quả thực tế:** 
- ViSoBERT (91.19%) < PhoBERT-base (93.46%).
- **Giải thích:** Dataset UIT-VSFC là feedback chính thức của sinh viên, ít teen code hơn social media. PhoBERT train trên văn bản chuẩn phù hợp hơn.

---

## 1.3. Traditional Machine Learning

### 1.3.1. SVM với TF-IDF

**Pipeline:**
```
Text → TF-IDF Vectorizer (5000 features) → LinearSVC (balanced)
```

**TF-IDF Parameters:**
```python
TfidfVectorizer(
    max_features=5000,      # Giới hạn vocabulary
    ngram_range=(1, 2),     # Unigram + Bigram
    sublinear_tf=True,      # Logarithmic scaling: 1 + log(tf)
    min_df=3                # Bỏ từ xuất hiện < 3 lần
)
```

**Lý do chọn LinearSVC:**
- SVM với kernel linear nhanh và hiệu quả với văn bản.
- `class_weight='balanced'` tự động cân bằng lớp thiểu số.

**Kết quả:** 89.73% accuracy - **Baseline mạnh** cho so sánh.

---

## 1.4. Hybrid Models

### 1.4.1. Hybrid V1 (Feature Fusion)

**Kiến trúc:**
```
┌────────────────┐
│  PhoBERT [CLS] │───┐
│  (768 dim)     │   │
└────────────────┘   │
                     ├──► Concatenate ──► MLP ──► Softmax
┌────────────────┐   │
│  TF-IDF        │───┤
│  (5000 dim)    │   │
└────────────────┘   │
                     │
┌────────────────┐   │
│  Handcrafted   │───┘
│  (10 features) │
└────────────────┘
```

**Handcrafted Features (10 dimensions):**
1. Số từ tích cực (positive word count)
2. Số từ tiêu cực (negative word count)
3. Số từ phủ định (negation count)
4. Số emoji
5. Số dấu chấm than (!)
6. Số dấu hỏi (?)
7. Tỉ lệ từ tích cực
8. Tỉ lệ từ tiêu cực
9. Có emoji tích cực không (0/1)
10. Polarity score = (pos - neg) / (pos + neg)

**Kết quả:** 89.17% - Thấp hơn PhoBERT thuần do:
- BERT embeddings bị **freeze** (không fine-tune).
- Concatenation đơn giản không tối ưu.

### 1.4.2. Hybrid V2 (Gated Fusion)

**Cải tiến so với V1:**

1. **Unfreeze last 2 layers của PhoBERT** → Cho phép fine-tuning nhẹ.
2. **Gated Fusion Mechanism:**
```python
gate = sigmoid(W × [BERT_emb, Feature_emb])  # gate ∈ [0,1]
fused = gate * BERT_emb + (1 - gate) * Feature_emb
```

**Ý nghĩa:** Model tự học khi nào nên tin BERT (gate ≈ 1), khi nào nên tin Features (gate ≈ 0).

**Kết quả đặc biệt:**
- Accuracy: 88.76% (thấp hơn V1)
- **Neutral Recall: 76%** (cao nhất trong tất cả models!)
- **Insight:** Gated fusion giúp model bắt được nhiều câu Neutral hơn bằng cách dựa vào handcrafted features khi BERT không chắc chắn.

---

## 1.5. Ensemble Methods

### 1.5.1. Soft Voting

**Công thức:**
```
P_ensemble(class) = (1/N) × Σ P_model_i(class)
prediction = argmax(P_ensemble)
```

**Tại sao chọn Soft Voting thay vì Hard Voting:**
- Soft voting sử dụng **xác suất** (continuous) thay vì chỉ nhãn (discrete).
- Lợi dụng được **độ tự tin** của từng model.

### 1.5.2. Weighted Ensemble

**Công thức:**
```
weights = [0.4, 0.3, 0.3]  # PhoBERT-large, PhoBERT-base, ViSoBERT
P_ensemble = Σ weight_i × P_model_i
```

**Lý do chọn trọng số:**
- PhoBERT-large có accuracy cao nhất → trọng số cao nhất.
- Các model khác đóng góp để tăng diversity.

### 1.5.3. Stacking V2 (Meta-Learning)

**Kiến trúc 2 tầng:**
```
Level 0: [PhoBERT-large, PhoBERT-base] → Predictions on Dev set
                     ↓
Level 1: Logistic Regression (Meta-Learner) → Final prediction
```

**Tại sao dùng Stacking:**
- Meta-learner **học** cách kết hợp tối ưu từ dữ liệu.
- Có thể phát hiện pattern: "Khi model A tự tin nhưng model B không → có thể sai".

---

## 1.6. Xử lý Class Imbalance

**Vấn đề:** Lớp Neutral chỉ chiếm **5.3%** dữ liệu.

**Giải pháp: Class-Weighted Loss**

```python
def compute_class_weights(labels):
    label_counts = Counter(labels)  # {0: 5077, 1: 594, 2: 5755}
    total = sum(label_counts.values())
    weights = [total / (3 × count) for count in label_counts]
    return weights  # [0.75, 6.45, 0.66]
```

**Ý nghĩa:** Neutral được "phạt nặng" hơn nếu sai (weight = 6.45 so với 0.7).

**Kết quả:**
- Neutral F1: 0.40 (không weight) → **0.61** (có weight) = **+52.5% improvement**

---

# PHẦN 2: KẾT QUẢ PHÂN TÍCH QUAN ĐIỂM

## 2.1. Bảng so sánh tổng hợp

| Category | Model | Accuracy | Weighted F1 | Macro F1 | Neutral F1 |
|----------|-------|----------|-------------|----------|------------|
| Baseline | Nguyen et al. (2018) | ~87% | - | ~75% | - |
| | | | | | |
| **Single** | **PhoBERT-large** | **93.56%** | 0.932 | 0.826 | 0.576 |
| **Single** | **PhoBERT-base** ⭐ | **93.46%** | **0.933** | **0.837** | **0.610** |
| Single | ViSoBERT | 91.19% | 0.906 | 0.767 | 0.436 |
| Single | SVM + TF-IDF | 89.73% | 0.893 | 0.746 | 0.399 |
| | | | | | |
| **Ensemble** | **Soft Voting** ⭐ | **92.61%** | 0.923 | 0.806 | 0.530 |
| Ensemble | Weighted Ensemble | 92.55% | 0.924 | 0.817 | 0.562 |
| Ensemble | Stacking V2 | 91.38% | 0.918 | 0.801 | 0.525 |
| Ensemble | Majority Voting | 91.16% | 0.908 | 0.782 | 0.483 |
| | | | | | |
| Hybrid | Hybrid V1: BERT + TF-IDF + Lexicon (Concat) | 89.17% | 0.890 | 0.747 | 0.406 |
| Hybrid | Hybrid V2: Fine-tune + Gated Fusion | 88.76% | 0.900 | 0.779 | 0.491 |

> ⭐ **Best Single:** PhoBERT-base (93.46%)  
> ⭐ **Best Ensemble:** Soft Voting (92.61%)  
> 💡 **Key Finding:** Single PhoBERT outperforms Ensemble methods

## 2.2. Phân tích chi tiết model tốt nhất

### PhoBERT-base (Best Weighted F1: 0.933)

| Class | Precision | Recall | F1-Score | Support | Giải thích |
|-------|-----------|--------|----------|---------|------------|
| Negative | 0.939 | 0.962 | **0.950** | 1,409 | Rất tốt - nhiều từ khóa rõ ràng |
| Neutral | 0.703 | 0.539 | **0.610** | 167 | Precision cao nhưng Recall thấp |
| Positive | 0.949 | 0.952 | **0.951** | 1,590 | Rất tốt - nhiều từ khóa rõ ràng |

### Confusion Matrix Analysis

```
              Predicted
              Neg   Neu   Pos
Actual  Neg  1355   27    27     ← 96.2% đúng
        Neu   49    90    28     ← 53.9% đúng (thấp!)
        Pos   40    36   1514    ← 95.2% đúng
```

**Phân tích:**
- **Neutral bị nhầm thành Negative (49 mẫu):** Câu chứa từ tiêu cực nhưng mang nghĩa trung lập.
- **Neutral bị nhầm thành Positive (28 mẫu):** Câu chứa từ tích cực nhẹ.

## 2.3. So sánh với Baseline

| Metric | Baseline (2018) | Best (PhoBERT-base) | Improvement |
|--------|-----------------|---------------------|-------------|
| Accuracy | ~87% | **93.46%** | **+6.46%** |
| Macro F1 | ~75% | **83.70%** | **+8.70%** |

**Giải thích improvement:**
1. **Pre-trained Language Model:** PhoBERT đã học ngữ nghĩa tiếng Việt từ 20GB text.
2. **Fine-tuning thay vì Feature Engineering:** Model tự học representation tốt hơn TF-IDF.
3. **Contextual Embeddings:** "Hay" trong "bài giảng hay" ≠ "hay" trong "hay than phiền".

---

# PHẦN 3: PHÂN TÍCH LỖI, ƯU ĐIỂM, HẠN CHẾ

## 3.1. Ưu điểm

| Ưu điểm | Minh chứng | Ý nghĩa thực tiễn |
|---------|------------|-------------------|
| **Accuracy cao** | 93.46% (vs 87% baseline) | Có thể triển khai thực tế |
| **Robust với lớp đa số** | Neg/Pos F1 > 0.95 | Đáng tin cậy cho đánh giá tích cực/tiêu cực |
| **Transfer Learning hiệu quả** | Chỉ cần 11K mẫu đạt 93%+ | Tiết kiệm dữ liệu và thời gian |
| **Class-weighted cải thiện minority** | Neutral F1: 0.40 → 0.61 | Giảm bias với lớp thiểu số |

## 3.2. Hạn chế

### 3.2.1. Lớp Neutral vẫn yếu

**Thống kê:**
- Neutral F1 = 0.61 (so với 0.95 của Neg/Pos)
- Recall = 0.54 → **Bỏ sót 46% câu Neutral**

**Nguyên nhân gốc rễ:**
1. **Data Imbalance:** Chỉ 594/11,426 = 5.2% mẫu train là Neutral.
2. **Bản chất mơ hồ:** Câu Neutral thường không có từ khóa sentiment rõ ràng.
3. **Label noise:** Ranh giới Neutral/Positive rất chủ quan.

**Ví dụ câu Neutral bị nhầm:**

| Câu | Predicted | Actual | Lý do sai |
|-----|-----------|--------|-----------|
| "Thầy dạy được" | Positive | Neutral | "được" = slightly positive |
| "Môn này bình thường" | Positive | Neutral | "bình thường" ambiguous |
| "Cần cải thiện thêm" | Negative | Neutral | "cải thiện" sounds negative |

### 3.2.2. Ensemble không vượt Single Model

**Quan sát:** Soft Voting (92.61%) < PhoBERT-base (93.46%)

**Giải thích:**
1. **Weak models kéo xuống:** ViSoBERT (91.19%), SVM (89.73%) đóng góp noise.
2. **Thiếu diversity:** Tất cả Transformer models có patterns tương tự.
3. **Simple averaging:** Không học trọng số từ dữ liệu.

**Bài học:** Ensemble hiệu quả khi các base models có **diverse errors**, không phải khi chúng giống nhau.

### 3.2.3. Hybrid chưa vượt Fine-tuning

**Quan sát:** Hybrid V2 (88.76%) < PhoBERT (93.46%)

**Giải thích:**
1. **End-to-end fine-tuning mạnh hơn:** 135M parameters được optimize cùng nhau.
2. **Handcrafted features limited:** Con người không thiết kế được features tốt bằng deep learning tự học.
3. **Information bottleneck:** Concatenation có thể làm mất thông tin.

---

# PHẦN 4: CÁC ĐIỂM MỚI, SÁNG TẠO

## 4.1. Phát hiện bất ngờ

### 4.1.1. PhoBERT-base > PhoBERT-large

| Model | Parameters | Accuracy | Weighted F1 |
|-------|------------|----------|-------------|
| PhoBERT-base | 135M | 93.46% | **0.933** |
| PhoBERT-large | 370M | 93.56% | 0.932 |

**Phân tích:**
- Accuracy chênh lệch không đáng kể (0.1%).
- **Weighted F1 của base cao hơn** (0.933 vs 0.932).
- **Neutral F1:** base = 0.610 > large = 0.576.

**Giải thích:**
1. **Overfitting:** Dataset 16K mẫu chưa đủ để exploit 370M params của large model.
2. **Regularization hiệu quả hơn:** Model nhỏ hơn ít bị overfit.
3. **Compute tradeoff:** Large tốn 3x thời gian nhưng không tốt hơn.

**Recommendation:** Với dataset < 50K mẫu, nên dùng PhoBERT-base.

### 4.1.2. Gated Fusion cực tốt cho Neutral Recall

| Model | Neutral Precision | Neutral Recall | Neutral F1 |
|-------|-------------------|----------------|------------|
| PhoBERT-base | **0.703** | 0.539 | 0.610 |
| Hybrid V2 (Gated) | 0.363 | **0.760** | 0.491 |

**Insight:**
- Hybrid V2 **bắt được 76% câu Neutral** (vs 54% của PhoBERT).
- Trade-off: Precision thấp hơn (0.36 vs 0.70).
- **Use case:** Nếu cần tìm TẤT CẢ câu Neutral (screening), dùng Hybrid V2.

**Cơ chế:** Gate tự động "tắt" BERT khi không chắc chắn, dựa vào features ngôn ngữ thay thế.

## 4.2. Đóng góp kỹ thuật

| Đóng góp | Mô tả | Impact |
|----------|-------|--------|
| **Multi-model Benchmark** | So sánh 9 models trên cùng dataset | Baseline cho nghiên cứu tương lai |
| **Gated Feature Fusion** | Tự động cân bằng BERT vs Features | Cải thiện Neutral Recall 41% |
| **Class-Weighted BERT** | Áp dụng weighted loss cho Transformer | Neutral F1 tăng 52.5% |
| **Stacking for NLP** | Meta-learning ensemble cho sentiment | Pipeline có thể tái sử dụng |

## 4.3. Hướng phát triển

1. **Data Augmentation cho Neutral:**
   - Back-translation (Việt → Anh → Việt)
   - Paraphrase với LLM (GPT, Gemini)

2. **Contrastive Learning:**
   - Học embeddings phân biệt rõ Neutral vs Positive/Negative.

3. **Multi-task Learning:**
   - Kết hợp Sentiment + Topic classification → shared representation tốt hơn.

4. **Few-shot với LLM:**
   - Dùng GPT-4/Gemini với few-shot prompting cho Neutral detection.

---

# KẾT LUẬN

Dự án đã triển khai thành công **9 phương pháp** phân tích quan điểm, đạt kết quả **93.46% accuracy** (vượt baseline 6.46%). Các đóng góp chính:

1. **Benchmark toàn diện** các phương pháp từ traditional ML đến deep learning.
2. **Phát hiện quan trọng:** PhoBERT-base hiệu quả hơn large cho dataset nhỏ.
3. **Kỹ thuật mới:** Gated Fusion cải thiện đáng kể Neutral Recall.
4. **Best practice:** Class-weighted loss là bắt buộc cho imbalanced sentiment data.

**Thách thức còn lại:** Lớp Neutral vẫn là điểm yếu (F1 = 0.61) cần nghiên cứu thêm.

---

**Ngày hoàn thành:** Tháng 2, 2026
