# Báo cáo Bài tập lập trình 2: Phân Tích Quan Điểm (Sentiment Analysis) - UIT-VSFC

## 1. Giới thiệu
Báo cáo này trình bày phương pháp và kết quả giải quyết bài toán Phân tích quan điểm (Sentiment Analysis) trên bộ dữ liệu phản hồi sinh viên UIT-VSFC. Mục tiêu là phân loại các phản hồi vào 3 nhóm: **Tiêu cực (Negative)**, **Trung tính (Neutral)**, và **Tích cực (Positive)**.

---

## (1) Kỹ thuật sử dụng
Để giải quyết bài toán, chúng em đã áp dụng các kỹ thuật học sâu (Deep Learning) tiên tiến nhất hiện nay cho xử lý ngôn ngữ tự nhiên tiếng Việt.

### Mô hình Học sâu (Model Architecture)
Chúng tôi sử dụng mô hình **PhoBERT Large** (`vinai/phobert-large`) làm backbone.
*   **PhoBERT** là mô hình ngôn ngữ tiền huấn luyện (Pre-trained Language Model) dựa trên kiến trúc RoBERTa, được huấn luyện trên lượng dữ liệu tiếng Việt khổng lồ (20GB).
*   Phiên bản **Large** (370 triệu parameters) được chọn thay vì Base để nẵm bắt tốt hơn các đặc trưng ngữ nghĩa phức tạp trong tiếng Việt.

### Xử lý dữ liệu (Data Preprocessing)
Trước khi đưa vào mô hình, dữ liệu được tiền xử lý kỹ lưỡng:
1.  **Chuẩn hóa văn bản:** Sử dụng Regular Expressions (Regex) để xử lý các teen-code và ký tự đặc biệt thường gặp trong phản hồi sinh viên.
    *   Ví dụ: `colonlove` $\to$ `yêu thích`, `colonsmile` $\to$ `vui vẻ`.
2.  **Tăng cường dữ liệu (Data Augmentation):** Gộp tập **Train** và tập **Dev** để tăng số lượng mẫu huấn luyện, giúp mô hình học tổng quát hơn.

### Chiến lược Huấn luyện (Training Strategy)
*   **Framework:** Hugging Face Transformers + PyTorch.
*   **Optimizer:** AdamW với learning rate $1.5 \times 10^{-5}$.
*   **Scheduler:** Linear warm-up (500 steps).
*   **Training Trick:** Sử dụng **Gradient Accumulation** để huấn luyện với batch size ảo lớn hơn trên GPU giới hạn, và **Mixed Precision (FP16)** để tăng tốc độ.

---

## (2) Kết quả phân tích trên tập Test
Kết quả thực nghiệm trên tập Test (3166 mẫu) cho thấy hiệu năng vượt trội của phương pháp đề xuất.

### Bảng kết quả chi tiết (Metrics)

| Lớp (Class) | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Negative** | 0.94 | 0.96 | **0.95** | 1409 |
| **Neutral** | 0.68 | 0.55 | **0.61** | 167 |
| **Positive** | 0.95 | 0.96 | **0.95** | 1590 |
| | | | | |
| **Accuracy** | | | **0.936** | 3166 |
| **Weighted Avg**| 0.93 | 0.94 | **0.934** | 3166 |

### Biểu đồ Confusion Matrix
![Confusion Matrix](./evaluation_plots/confusion_matrix.png)

### Nhận xét
*   Độ chính xác toàn cục (Accuracy) đạt **93.6%**.
*   Điểm F1 trung bình có trọng số (Weighted F1) đạt **93.4\%**.
*   Hai lớp quan trọng nhất là **Negative** và **Positive** đều đạt F1-score **0.95**, chứng tỏ mô hình cực kỳ tin cậy trong việc phân định tốt/xấu.

---

## (3) Phân tích lỗi, ưu điểm, hạn chế

### Ưu điểm
*   **Hiệu năng cao:** Việc sử dụng PhoBERT Large giúp mô hình hiểu sâu ngữ cảnh tiếng Việt, vượt trội so với các phương pháp Machine Learning truyền thống (SVM, Naive Bayes).
*   **Độ tin cậy cao:** Kết quả trên hai lớp chính (Pos/Neg) rất cân bằng và cao (>95%).

### Hạn chế và Phân tích lỗi
Hạn chế lớn nhất nằm ở lớp **Neutral (Trung tính)**:
*   F1-score của lớp Neutral chỉ đạt **0.61**, thấp hơn nhiều so với hai lớp còn lại.
*   **Nguyên nhân:**
    1.  **Mất cân bằng dữ liệu nghiêm trọng:** Lớp Neutral chỉ chiếm khoảng 5% tổng dữ liệu (167/3166 mẫu test).
    2.  **Tính chất nhập nhằng:** Các phản hồi trung tính thường chứa cả ý khen và chê nhẹ, hoặc không rõ ràng (ví dụ: "Thầy dạy nhiệt tình nhưng hơi nhanh"), khiến mô hình dễ nhầm lẫn sang Positive hoặc Negative.

---

## (4) Điểm mới và Sáng tạo
Trong quá trình thực hiện, chúng tôi đã áp dụng một số cải tiến sáng tạo:

1.  **Quy trình tiền xử lý ngữ nghĩa (Semantic Preprocessing):** Thay vì chỉ loại bỏ ký tự đặc biệt, chúng em đã phân tích tập dữ liệu và viết script để "dịch" các token đặc biệt (ví dụ: `colonlove`) về dạng ngôn ngữ tự nhiên, giúp PhoBERT hiểu được cảm xúc tích cực ẩn chứa trong các icon này thay vì coi chúng là nhiễu.
2.  **Tối ưu hóa chiến lược dữ liệu:** Quyết định gộp tập Dev vào Train (sau khi đã tìm ra siêu tham số tốt) đã giúp model "nhìn" thấy nhiều dữ liệu hơn, trực tiếp đóng góp vào việc tăng Accuracy từ ~91.9% lên ~93.6%.
