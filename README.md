# BLOOM Sentiment Analysis (ABSA)

> 飯店評論多面向情感分析 - 使用 BLOOM-389M 全量微調

## 📌 專案概述

本專案使用 BLOOM-389M 中文模型進行全量微調（Full Fine-Tuning），訓練一個能夠分析飯店評論中多個面向情感的 AI 系統。相較於傳統情感分析只判斷正負面，ABSA (Aspect-Based Sentiment Analysis) 能夠識別評論中提到的具體面向並分別判斷情感。

## 🎯 專案目標

- 實作 LLM 全量微調 (Supervised Fine-Tuning)
- 學習 ABSA 任務的資料處理流程
- 建立飯店評論情感分析系統
- 提供互動式 Gradio Demo

## 🛠️ 技術棧

- **Framework**: HuggingFace Transformers
- **Base Model**: Langboat/bloom-389m-zh (中文 BLOOM 模型)
- **Fine-tuning Method**: Full SFT (Supervised Fine-Tuning)
- **Dataset**: 飯店評論 ABSA 資料集 (~16K 樣本)
- **UI**: Gradio

## 📊 主要成果

| 指標 | 數值 |
|------|------|
| **Final Loss** | **0.13** |
| **訓練時間** | ~28 min/epoch (T4 GPU) |
| **模型參數** | 389M (全量微調) |
| **訓練資料** | 16K 飯店評論 |

## 🏨 ABSA 分析面向

系統能夠分析以下面向的情感：

- 🛏️ **整潔舒適** - 房間清潔度、舒適度
- 🏢 **設施** - 硬體設施、設備狀況
- 👨‍💼 **服務** - 服務人員態度、服務品質
- 📍 **地點** - 位置、交通便利性
- 💰 **性價比** - 價格合理性
- 📋 **其他** - 其他特殊面向

**情感分類**: 正面、中立、負面

## 🚀 快速開始

### 安裝依賴

```bash
pip install -r requirements.txt
```

依賴套件：
- `transformers>=4.30.0`
- `datasets`
- `torch>=2.0.0`
- `gradio>=4.0.0`
- `accelerate`

### 訓練模型

```bash
# 基本訓練（1 epoch）
python train.py

# 自訂參數
python train.py --epochs 2 --batch-size 4 --learning-rate 2e-5

# 指定輸出路徑
python train.py --output-dir models/my-bloom-absa
```

**訓練參數說明:**
- `--model`: 預訓練模型名稱 (預設: `Langboat/bloom-389m-zh`)
- `--data-dir`: 處理後的資料集路徑 (預設: `data/processed`)
- `--output-dir`: 模型輸出路徑 (預設: `models/bloom-absa`)
- `--epochs`: 訓練 epochs (預設: 1)
- `--batch-size`: Batch size (預設: 4)
- `--learning-rate`: 學習率 (預設: 2e-5)

### 啟動 Gradio Demo

```bash
# 使用預設模型路徑
python demo.py

# 指定模型路徑
python demo.py --model models/bloom-absa

# 開啟公開分享連結
python demo.py --share

# 自訂 port
python demo.py --port 8080
```

Demo 將在 http://localhost:7860 啟動

## 📁 專案結構

```
sentiment-analysis-absa/
├── train.py              # 訓練腳本
├── demo.py              # Gradio 互動介面
├── requirements.txt     # 依賴套件
├── README.md           # 專案說明
├── data/               # 資料集目錄
│   ├── raw/           # 原始資料
│   └── processed/     # 處理後的資料
├── models/            # 儲存訓練好的模型
│   └── bloom-absa/
├── configs/           # 配置檔案
├── notebooks/         # Jupyter notebooks
└── src/              # 核心程式碼模組
```

## 🔬 方法論

### 資料格式

ABSA 資料集格式 (Alpaca 格式):

```json
{
  "instruction": "以下這段文字提到那些面向?是正面還是負面的情緒?",
  "input": "房間很乾淨，服務人員態度很好，地點方便，性價比高",
  "output": "整潔舒適-正面,服務-正面,地點-正面,性價比-正面"
}
```

### 訓練配置

```python
TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=True,  # GPU 加速
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True
)
```

### 模型特點

- **Bloom Tokenizer**: 左側 padding (與 LLaMA 不同)
- **Causal LM**: 自回歸語言模型
- **全量微調**: 更新所有 389M 參數
- **損失函數**: Language Modeling Loss

## 📈 訓練結果

### 訓練曲線

| Step | Train Loss | Eval Loss |
|------|------------|-----------|
| 0    | ~2.5       | ~2.4      |
| 200  | ~0.8       | ~0.6      |
| 400  | ~0.3       | ~0.25     |
| 600  | ~0.15      | **0.13**  |

### 範例輸出

**輸入評論:**
```
"房間很乾淨，服務人員態度很好，但是地點稍微偏遠"
```

**模型分析:**
```
整潔舒適-正面, 服務-正面, 地點-負面
```

## 💡 關鍵洞察

1. **全量微調效果顯著**: 1 epoch 即可收斂到 Loss 0.13
2. **Padding Side 重要**: Bloom 必須使用左側 padding
3. **小 Batch Size**: 受限於 GPU 記憶體，使用 batch_size=4
4. **資料品質關鍵**: ABSA 標註品質直接影響模型表現

## 📦 資料集

**資料來源**: 飯店評論 ABSA 資料集

| Split | 樣本數 |
|-------|--------|
| Train | ~14K   |
| Val   | ~2K    |
| Total | ~16K   |

**資料處理流程**:
1. 原始 CSV 格式 → Alpaca JSON
2. Tokenization
3. 轉換為 HuggingFace Dataset 格式
4. 儲存為 Arrow 格式供訓練使用

## 🎨 Demo 功能

Gradio 介面提供：
- 📝 評論文字輸入框
- 🎛️ 可調參數 (temperature, top_p)
- 📊 即時分析結果顯示
- 🎯 範例評論快速測試
- 📈 技術規格說明

## 🔗 參考資料

- Base Model: [Langboat/bloom-389m-zh](https://huggingface.co/Langboat/bloom-389m-zh)
- BLOOM Paper: [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100)
- ABSA Survey: [Aspect-Based Sentiment Analysis: A Survey of Deep Learning Methods](https://arxiv.org/abs/2203.01054)

## 📝 環境需求

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (GPU 訓練)
- 16GB+ RAM (或使用 GPU)
- ~10GB GPU VRAM (T4/V100/A100)

## 🏆 成就

✅ Loss 0.13 (1 epoch)
✅ 完整的訓練與推理 pipeline
✅ 互動式 Gradio Demo
✅ 支援多面向情感分析

## ⚠️ 注意事項

1. **資料集路徑**: 確保 `data/processed/` 存在並包含處理好的資料集
2. **模型大小**: 訓練完成的模型約 1.5GB
3. **GPU 記憶體**: 全量微調建議至少 10GB VRAM
4. **Tokenizer Padding**: Bloom 使用左側 padding，與 LLaMA 不同

---

*專案建立於 2023-2024 年度深度學習課程*
*微調方法: Full Supervised Fine-Tuning (SFT)*
