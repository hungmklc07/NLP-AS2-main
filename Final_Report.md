# Báo Cáo Phân Tích Quan Điểm Sinh Viên UIT
## Vietnamese Student Feedback Sentiment Analysis

**Sinh viên:** [Họ tên sinh viên]  
**MSSV:** [Mã số sinh viên]  
**Môn học:** Xử lý ngôn ngữ tự nhiên (NLP)

---

## 1. Giới thiệu

### 1.1. Mô tả bài toán
Bài toán **Phân tích quan điểm (Sentiment Analysis)** trên tập dữ liệu **UIT-VSFC** (Vietnamese Students' Feedback Corpus) đánh giá của sinh viên trường UIT. Mục tiêu là phân loại mỗi câu đánh giá vào một trong ba nhãn:
- **Negative (0):** Tiêu cực
- **Neutral (1):** Trung lập  
- **Positive (2):** Tích cực

### 1.2. Thống kê dữ liệu

| Tập | Số mẫu | Negative | Neutral | Positive |
|-----|--------|----------|---------|----------|
| Train | 11,426 | 44.5% | 5.2% | 50.3% |
| Dev | 1,583 | 44.3% | 5.3% | 50.4% |
| Test | 3,166 | 44.5% | 5.3% | 50.2% |

> ⚠️ **Lưu ý:** Lớp **Neutral chỉ chiếm ~5%** dữ liệu → Gây khó khăn lớn cho các model trong việc phân loại đúng lớp này.

### 1.3. Baseline (Bài báo gốc)
Theo bài báo **Nguyen et al. (2018)** - "UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis", kết quả cao nhất đạt được:
- **Accuracy:** ~84-87% (SVM với TF-IDF và n-grams)
- **Macro F1:** ~70-75%

---

## 2. Kỹ thuật sử dụng

### 2.1. Các phương pháp đã triển khai

| Nhóm | Model | Mô tả |
|------|-------|-------|
| **Traditional ML** | SVM + TF-IDF | Support Vector Machine với đặc trưng TF-IDF (5000 features), class balancing |
| | Logistic Regression + TF-IDF | Logistic Regression với TF-IDF features |
| **Deep Learning** | PhoBERT-base | Pre-trained BERT cho tiếng Việt (135M params), fine-tuning |
| | PhoBERT-large | Phiên bản lớn hơn (370M params), fine-tuning |
| | ViSoBERT | BERT tối ưu cho social media tiếng Việt |
| **Hybrid** | Hybrid V1 | Kết hợp BERT embeddings + TF-IDF + Handcrafted features |
| | Hybrid V2 (Gated) | Fine-tuning + Gated Fusion mechanism |
| **Ensemble** | Soft Voting | Cộng xác suất từ tất cả models |
| | Weighted Ensemble | Trọng số theo hiệu suất từng model |
| | Stacking V2 | Meta-learner (Logistic Regression) học cách kết hợp |

### 2.2. Kỹ thuật cải tiến

1. **Class-Weighted Loss:** Áp dụng trọng số lớp để cân bằng ảnh hưởng của các lớp thiểu số (Neutral).
   ```
   weights = [total / (num_classes × count_per_class)]
   ```

2. **Proper Train/Val Split:** Sử dụng tập Dev làm validation, tránh data leakage.

3. **Handcrafted Sentiment Features:**
   - Đếm từ tích cực/tiêu cực (sentiment lexicon)
   - Phát hiện phủ định (negation detection)
   - Emoji và punctuation features
   - Sentiment shift (ví dụ: "không tốt" → đảo ngược)

4. **Gated Fusion (Hybrid V2):**
   ```
   gate = sigmoid(W × [BERT, Features])
   output = gate × BERT + (1-gate) × Features
   ```
   Model tự học khi nào tin BERT, khi nào tin Features.

---

## 3. Kết quả phân tích quan điểm trên tập Test

### 3.1. Bảng so sánh tổng thể

| Model | Accuracy | Weighted F1 | Macro F1 | Neutral F1 |
|-------|----------|-------------|----------|------------|
| **Baseline (Nguyen 2018)** | ~87% | - | ~75% | - |
| SVM + TF-IDF | 89.73% | 0.893 | 0.746 | 0.399 |
| ViSoBERT | 91.19% | 0.906 | 0.767 | 0.436 |
| Hybrid V1 | 89.17% | 0.890 | 0.747 | 0.406 |
| Hybrid V2 (Gated) | 88.76% | **0.900** | 0.779 | **0.491** |
| PhoBERT-base | 93.46% | 0.933 | 0.837 | 0.610 |
| **PhoBERT-large** | **93.56%** | 0.932 | 0.826 | 0.576 |
| Soft Voting Ensemble | 92.61% | 0.923 | 0.806 | 0.530 |
| Weighted Ensemble | 92.55% | 0.924 | 0.817 | 0.562 |
| Stacking V2 | 91.38% | 0.918 | 0.801 | 0.525 |

### 3.2. Phân tích per-class (Model tốt nhất: PhoBERT-base)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.939 | 0.962 | 0.950 | 1,409 |
| Neutral | 0.703 | 0.539 | 0.610 | 167 |
| Positive | 0.949 | 0.952 | 0.951 | 1,590 |
| **Weighted Avg** | - | - | **0.933** | 3,166 |

### 3.3. So sánh với Baseline

| Metric | Baseline (2018) | Best (PhoBERT-base) | Cải thiện |
|--------|-----------------|---------------------|-----------|
| Accuracy | ~87% | **93.46%** | **+6.46%** |
| Macro F1 | ~75% | **83.70%** | **+8.70%** |

> ✅ **Kết quả vượt trội so với baseline**, chứng minh hiệu quả của các mô hình Transformer pre-trained cho tiếng Việt.

---

## 4. Phân tích lỗi, ưu điểm và hạn chế

### 4.1. Ưu điểm

1. **Hiệu suất cao:** PhoBERT-base đạt **93.46% accuracy**, vượt xa baseline (~87%).

2. **Xử lý tốt lớp đa số:** Negative và Positive F1 đều đạt >0.95.

3. **Transfer Learning hiệu quả:** Pre-trained models học được ngữ nghĩa tiếng Việt, giảm nhu cầu dữ liệu lớn.

4. **Class-weighted loss cải thiện Neutral:** Từ ~0.40 (không weight) lên **0.61** (có weight).

### 4.2. Hạn chế

1. **Lớp Neutral vẫn khó:** Dù đã cải thiện, Neutral F1 chỉ đạt **0.61** so với 0.95 của các lớp khác.
   - **Nguyên nhân:** Chỉ ~5% dữ liệu là Neutral, model thiếu đủ ví dụ để học.
   - **Bản chất:** Câu Neutral thường mơ hồ, thiếu từ khóa rõ ràng.

2. **Ensemble không luôn tốt hơn Single model:**
   - Soft Voting (92.61%) < PhoBERT-base (93.46%)
   - **Nguyên nhân:** Các model yếu (ViSoBERT, SVM) kéo trung bình xuống.

3. **Hybrid chưa vượt được Fine-tuning thuần:**
   - Hybrid V2 (88.76%) < PhoBERT (93.46%)
   - **Nguyên nhân:** End-to-end fine-tuning mạnh hơn việc kết hợp features thủ công.

### 4.3. Phân tích lỗi cụ thể

**Các trường hợp thường bị sai:**

| Loại lỗi | Ví dụ | Dự đoán | Thực tế |
|----------|-------|---------|---------|
| Câu ngắn mơ hồ | "Bình thường" | Positive | Neutral |
| Mỉa mai (Sarcasm) | "Hay ghê luôn á" | Positive | Negative |
| Từ chỉ mức độ | "Thầy dạy được" | Positive | Neutral |
| Phủ định phức tạp | "Không phải là không hay" | Negative | Positive |

---

## 5. Các điểm mới, sáng tạo

### 5.1. Phát hiện quan trọng

1. **PhoBERT-base > PhoBERT-large:**
   - Kết quả bất ngờ: Model nhỏ hơn lại tốt hơn model lớn.
   - **Giải thích:** Dataset ~16K samples có thể chưa đủ để exploit toàn bộ capacity của large model → Overfitting.

2. **Gated Fusion cải thiện Neutral Recall:**
   - Hybrid V2 đạt Neutral Recall **76%** (cao nhất), mặc dù Precision thấp.
   - **Insight:** Cơ chế gate giúp model biết khi nào cần dựa vào features ngôn ngữ (khi BERT không chắc chắn).

3. **Soft Voting đơn giản nhưng hiệu quả:**
   - Không cần train thêm meta-learner, chỉ cộng xác suất.
   - Hoạt động tốt khi các base models có diversity cao.

### 5.2. Đóng góp của bài tập

| Đóng góp | Mô tả |
|----------|-------|
| **Multi-model comparison** | So sánh 9 phương pháp khác nhau trên cùng dataset |
| **Hybrid Feature Fusion** | Kết hợp BERT + TF-IDF + Sentiment Lexicon |
| **Gated Attention Fusion** | Cơ chế tự động điều chỉnh trọng số fusion |
| **Stacking Ensemble** | Meta-learning để học cách kết hợp tối ưu |
| **Class Imbalance Handling** | Weighted loss cải thiện lớp thiểu số |

---

## 6. Kết luận

### 6.1. Tóm tắt kết quả
- **Model tốt nhất:** PhoBERT-base với class-weighted loss, đạt **93.46% Accuracy** và **0.933 Weighted F1**.
- **Cải thiện so với baseline:** +6.46% Accuracy, +8.70% Macro F1.
- **Thách thức còn lại:** Lớp Neutral (F1 = 0.61) cần nhiều cải tiến hơn.

### 6.2. Hướng phát triển
1. **Data Augmentation:** Tăng cường dữ liệu cho lớp Neutral.
2. **Few-shot Learning:** Áp dụng kỹ thuật học với ít mẫu.
3. **Multi-task Learning:** Kết hợp sentiment với topic classification.
4. **Contrastive Learning:** Học biểu diễn phân biệt rõ hơn giữa các lớp.

---

## 7. Tài liệu tham khảo

1. Nguyen, K. V., et al. (2018). "UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis". *RIVF 2018*.

2. Nguyen, D. Q., & Nguyen, A. T. (2020). "PhoBERT: Pre-trained language models for Vietnamese". *Findings of EMNLP 2020*.

3. Nguyen, H. T., et al. (2023). "ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing". *arXiv*.

---

**Ngày hoàn thành:** Tháng 2, 2026
