# Vietnamese Sentiment Analysis - Colab Training Notebook

## 📋 Hướng dẫn sử dụng
# 1. Upload folder NLP-AS2 lên Google Drive
# 2. Mở notebook này trên Colab
# 3. Bật GPU: Runtime > Change runtime type > GPU
# 4. Chạy từng cell theo thứ tự

#%% [markdown]
# ## Cell 1: Mount Google Drive

#%%
from google.colab import drive
drive.mount('/content/drive')

#%% [markdown]
# ## Cell 2: Copy project từ Drive và cài dependencies

#%%
import os

# THAY ĐỔI ĐƯỜNG DẪN NÀY CHO PHÙ HỢP
PROJECT_PATH = "/content/drive/MyDrive/NLP-AS2-main"

# Copy project vào Colab runtime (nhanh hơn làm việc trên Drive)
!cp -r "{PROJECT_PATH}" /content/NLP-AS2
os.chdir("/content/NLP-AS2")

# Cài dependencies
!pip install transformers torch scikit-learn pandas matplotlib seaborn tqdm -q

print("✅ Setup complete!")
print(f"Current directory: {os.getcwd()}")
!ls -la

#%% [markdown]
# ## Cell 3: Kiểm tra GPU

#%%
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

#%% [markdown]
# ## Cell 4: Preprocess data

#%%
!python preprocess.py

#%% [markdown]
# ---
# # 🚀 TRAINING MODELS
# Chạy từng model một, hoặc chọn model bạn muốn
# ---

#%% [markdown]
# ## Cell 5A: Train SVM Baseline (~5 phút, không cần GPU)

#%%
!python train_baseline.py --model svm

#%% [markdown]
# ## Cell 5B: Train PhoBERT-base (~30 phút)

#%%
!python train_improved.py --model phobert-base --weighted-loss

#%% [markdown]
# ## Cell 5C: Train ViSoBERT (~30 phút)

#%%
!python train_improved.py --model visobert --weighted-loss

#%% [markdown]
# ## Cell 5D: Train PhoBERT-large (~60 phút, cần GPU mạnh)
# ⚠️ Nếu bị Out of Memory, bỏ qua cell này

#%%
!python train_improved.py --model phobert-large --weighted-loss

#%% [markdown]
# ## Cell 5E: Train Hybrid Model V1 (~20 phút)

#%%
!python train_hybrid.py

#%% [markdown]
# ## Cell 5F: Train Advanced Hybrid Model V2 (~30-45 phút)
# 🔥 Fine-tuning PhoBERT + Advanced Features + Gated Fusion

#%%
!python train_hybrid_v2.py --model phobert-base

#%% [markdown]
# ## Cell 5G: Train Hybrid V3 (~45-60 phút)
# 🔥🔥 Multi-Head Cross-Attention + Focal Loss + PhoBERT-large

#%%
!python train_hybrid_v3.py

#%% [markdown]
# ---
# # 📊 EVALUATION
# ---

#%% [markdown]
# ## Cell 6A: Tạo Simple Ensemble (Voting)

#%%
!python ensemble.py

#%% [markdown]
# ## Cell 6B: Tạo Stacking Ensemble V2 (Meta-Learning)
# 🚀 Học cách kết hợp tối ưu từ các models đã train

#%%
!python ensemble_v2.py

#%% [markdown]
# ## Cell 6C: Tạo Ensemble V3 (Neural Meta-Learner + Temperature Scaling)
# 🔥🔥🔥 Neural network meta-learner with K-Fold CV

#%%
!python ensemble_v3.py

#%% [markdown]
# ## Cell 7: Đánh giá và so sánh tất cả models

#%%
!python evaluate_all.py

#%% [markdown]
# ## Cell 8: Xem kết quả

#%%
import pandas as pd

# Đọc bảng so sánh
df = pd.read_csv("results/comparison_results.csv")
print("📊 BẢNG SO SÁNH CÁC MODELS:")
print("="*80)
display(df)

#%%
# Hiển thị biểu đồ
from IPython.display import Image, display

print("\n📈 BIỂU ĐỒ SO SÁNH:")
display(Image("evaluation_plots/model_comparison.png", width=800))

print("\n🎯 NEUTRAL CLASS FOCUS:")
display(Image("evaluation_plots/neutral_class_focus.png", width=600))

#%% [markdown]
# ## Cell 9: Copy kết quả về Drive

#%%
import shutil

# Tạo folder kết quả trên Drive
output_drive = "/content/drive/MyDrive/NLP-AS2-results"
os.makedirs(output_drive, exist_ok=True)

# Copy results
shutil.copytree("results", f"{output_drive}/results", dirs_exist_ok=True)
shutil.copytree("evaluation_plots", f"{output_drive}/evaluation_plots", dirs_exist_ok=True)

print(f"✅ Kết quả đã được lưu vào: {output_drive}")

#%% [markdown]
# ## Cell 10: Download kết quả về máy (optional)

#%%
!zip -r results.zip results/ evaluation_plots/

from google.colab import files
files.download('results.zip')
