"""
BLOOM ABSA Sentiment Analysis - Gradio Demo
飯店評論情感分析互動介面

用法:
    python demo.py
    python demo.py --model models/bloom-absa
"""
import argparse
import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ABSASentimentAnalyzer:
    def __init__(self, model_path):
        """初始化分析器"""
        print(f"📦 載入模型: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto'
        )

        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✅ 模型載入完成")

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

                analyze_btn = gr.Button("🔍 分析情感", variant="primary")

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
        print(f"❌ 找不到模型: {args.model}")
        print(f"請先執行訓練: python train.py")
        return

    # 建立並啟動 demo
    print(f"🚀 啟動 Gradio Demo...")
    demo = create_demo(args.model)
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )


if __name__ == '__main__':
    main()
