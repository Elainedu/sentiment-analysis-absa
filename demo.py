"""
===============================================================================
專案名稱: ABSA Sentiment Analysis System - 飯店評論多面向情感分析系統
===============================================================================

[專案簡介]
這是一個基於 BLOOM 語言模型的 ABSA (Aspect-Based Sentiment Analysis)
情感分析系統。系統可以自動分析飯店評論文字，識別出提到的各個面向
（如整潔度、服務、地點等），並判斷每個面向的情感傾向（正面/中立/負面）。

[核心技術]
- 任務類型: ABSA (Aspect-Based Sentiment Analysis)
- 基礎模型: BLOOM-389M-ZH (中文語言模型)
- 微調方法: Supervised Fine-Tuning (SFT) 全量微調
- 深度學習框架: PyTorch + Transformers
- Web 介面: Gradio

[ABSA 說明]
ABSA 與傳統情感分析的差異:
- 傳統情感分析: 只給出整體正面/負面（例如：「這家飯店很棒」→ 正面）
- ABSA: 細分到各個面向（例如：「房間乾淨但服務差」→ 房間:正面, 服務:負面）

[分析面向]
系統可識別的飯店評論面向:
1. 🛏️ 整潔舒適 - 房間清潔度、床品舒適度、整體衛生
2. 🏢 設施 - 硬體設備、裝潢、維護狀況
3. 👨‍💼 服務 - 服務人員態度、專業度、響應速度
4. 📍 地點 - 地理位置、交通便利性、周邊環境
5. 💰 性價比 - 價格合理性、物超所值程度
6. 📋 其他 - 早餐、停車、網路等其他面向

[情感分類]
每個面向的情感標籤:
- 正面 (Positive): 讚賞、滿意的評價
- 中立 (Neutral): 客觀描述、無明顯傾向
- 負面 (Negative): 批評、不滿的評價

[訓練資料]
- 資料集: 16K 飯店評論 ABSA 標註資料
- 標註格式: <面向, 情感> 對
- 訓練方法: 全量微調 (Full Fine-Tuning)
- 訓練時間: ~28 min/epoch (T4 GPU)
- Final Loss: ~0.13

[模型架構]
- 參數量: 389M (全部參數都參與訓練)
- 輸入格式: "以下這段文字提到那些面向?是正面還是負面的情緒?\n{評論}\n這段文字是屬於:"
- 輸出格式: 生成式回答，列出識別的面向和情感

[啟動方式]
基本啟動（使用預設模型路徑）:
    python demo.py

指定模型路徑:
    python demo.py --model models/bloom-absa

產生公開分享連結:
    python demo.py --share

指定 Port:
    python demo.py --port 8080

[使用說明]
1. 啟動後開啟 http://127.0.0.1:7860
2. 在輸入框輸入飯店評論文字
3. 點擊「[INFO] 分析情感」按鈕
4. 系統會分析並輸出：
   - 識別出的面向
   - 每個面向的情感傾向
   - 支持該判斷的理由
5. 可在「進階設定」調整生成參數

[參數調整]
- Temperature (0.1-1.5): 控制生成的隨機性
  - 低值 (0.1-0.5): 更確定、保守的輸出
  - 高值 (1.0-1.5): 更多樣、創造性的輸出
- Top-p (0.1-1.0): 核採樣參數，控制候選詞的範圍
- 預設值: temperature=0.7, top_p=0.9

[評論範例]
範例 1:
輸入: "房間很乾淨，服務人員態度很好，地點方便，性價比高"
輸出: 整潔舒適-正面, 服務-正面, 地點-正面, 性價比-正面

範例 2:
輸入: "設施老舊，房間有異味，服務態度差"
輸出: 設施-負面, 整潔舒適-負面, 服務-負面

範例 3:
輸入: "地點不錯在市中心，但是房間隔音很差，晚上很吵"
輸出: 地點-正面, 整潔舒適-負面

[訓練模型]
執行訓練:
    python train.py

訓練配置:
- Batch size: 8
- Learning rate: 2e-5
- Epochs: 3-5
- Gradient accumulation: 4

[面試展示重點]
1. **ABSA 概念**: 說明為何細粒度的情感分析比整體分析更有價值
2. **實際應用**: 飯店業者可根據各面向改善服務、電商平台產品評論分析
3. **技術挑戰**:
   - 同一句話可能包含多個面向和不同情感
   - 隱含情感的識別（例如：「還可以」是中立還是略微負面？）
   - 中文語言的特殊性（例如：反諷、委婉表達）
4. **改進方向**:
   - 加入更多面向（如 WiFi、早餐、噪音等）
   - 情感強度評分（1-5星）
   - 跨領域遷移（餐廳、3C產品等）

[檔案結構]
demo.py                          # 本檔案 - Gradio 分析介面
train.py                         # 模型訓練程式
models/bloom-absa/               # 訓練好的模型
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files
data/                            # 訓練資料
    └── hotel_reviews_absa.json

[實際應用場景]
1. 飯店業: 自動分析顧客評論，找出需改善的面向
2. OTA 平台: 為旅客提供細分的評分參考
3. 競爭分析: 比較不同飯店在各面向的表現
4. 服務優化: 根據負面面向優先級進行改善

[開發者]
碩士班課程專案 - 深度學習（進階）
建立日期: 2024
更新日期: 2026-03-10 (修正 emoji 編碼問題)

===============================================================================
"""
import argparse
import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ABSASentimentAnalyzer:
    def __init__(self, model_path):
        """初始化分析器"""
        print(f"[LOAD] 載入模型: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto'
        )

        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[OK] 模型載入完成")

    def analyze(self, review_text, max_length=512, temperature=0.7, top_p=0.9):
        """分析飯店評論的情感"""

        # 構建 prompt
        prompt = f"以下這段文字提到那些面向?是正面還是負面的情緒?\n{review_text}\n這段文字是屬於:"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.model.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 解碼
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取生成的部分
        if "這段文字是屬於:" in full_output:
            result = full_output.split("這段文字是屬於:")[-1].strip()
        else:
            result = full_output

        return result


def create_demo(model_path):
    """建立 Gradio 介面"""
    analyzer = ABSASentimentAnalyzer(model_path)

    def analyze_review(review, temperature, top_p):
        """分析評論"""
        if not review.strip():
            return "請輸入飯店評論文字"

        result = analyzer.analyze(review, temperature=temperature, top_p=top_p)
        return result

    # 範例評論
    examples = [
        ["房間很乾淨，服務人員態度很好，地點方便，性價比高"],
        ["設施老舊，房間有異味，服務態度差"],
        ["地點不錯在市中心，但是房間隔音很差，晚上很吵"],
        ["早餐很豐富，房間舒適，但是價格偏高"],
    ]

    # 建立介面
    with gr.Blocks(theme=gr.themes.Soft(), title="飯店評論情感分析") as demo:
        gr.Markdown("""
        # 🏨 Hotel Review ABSA Sentiment Analysis
        ### 飯店評論多面向情感分析系統

        本系統使用微調的 BLOOM-389M 模型，可自動分析飯店評論中提到的各個面向及其情感。

        **分析面向:**
        - 🛏️ 整潔舒適
        - 🏢 設施
        - 👨‍💼 服務
        - 📍 地點
        - 💰 性價比
        - 📋 其他

        **情感分類:** 正面、中立、負面
        """)

        with gr.Row():
            with gr.Column():
                review_input = gr.Textbox(
                    label="飯店評論",
                    placeholder="請輸入飯店評論...",
                    lines=5
                )

                with gr.Accordion("進階設定", open=False):
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="控制生成的隨機性"
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p",
                        info="核採樣參數"
                    )

                analyze_btn = gr.Button("[INFO] 分析情感", variant="primary")

            with gr.Column():
                output = gr.Textbox(
                    label="分析結果",
                    lines=10,
                    placeholder="分析結果將顯示在這裡..."
                )

        gr.Examples(
            examples=examples,
            inputs=review_input,
            label="範例評論"
        )

        # 事件綁定
        analyze_btn.click(
            fn=analyze_review,
            inputs=[review_input, temperature, top_p],
            outputs=output
        )

        gr.Markdown("""
        ---
        **技術規格:**
        - 模型: BLOOM-389M-ZH (全量微調)
        - 訓練資料: 16K 飯店評論 ABSA 標註
        - 微調方法: Supervised Fine-Tuning (SFT)

        **訓練成果:**
        - Final Loss: ~0.13
        - 訓練時間: ~28 min/epoch (T4 GPU)
        """)

    return demo


def main():
    parser = argparse.ArgumentParser(description='啟動 ABSA 情感分析 Demo')
    parser.add_argument('--model', type=str,
                       default='models/bloom-absa',
                       help='模型路徑')
    parser.add_argument('--share', action='store_true',
                       help='建立公開分享連結')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port 號')

    args = parser.parse_args()

    # 檢查模型
    if not os.path.exists(args.model):
        print(f"[ERROR] 找不到模型: {args.model}")
        print(f"請先執行訓練: python train.py")
        return

    # 建立並啟動 demo
    print(f"[START] 啟動 Gradio Demo...")
    demo = create_demo(args.model)
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )


if __name__ == '__main__':
    main()
