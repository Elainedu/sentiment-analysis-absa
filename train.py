"""
BLOOM Sentiment Analysis (ABSA) - Training Script
使用飯店評論 ABSA 資料集微調 Bloom-389m

用法:
    python train.py
    python train.py --epochs 2 --batch-size 4
"""
import argparse
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk


def load_model_and_tokenizer(model_name='Langboat/bloom-389m-zh'):
    """載入模型和 tokenizer"""
    print(f"[LOAD] 載入模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # 或 torch.float16 if GPU
        device_map='auto'
    )

    # Bloom 使用左側 padding
    tokenizer.padding_side = 'left'

    # 設定 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print(f"[OK] 模型載入完成")
    print(f"   參數量: {model.num_parameters():,}")

    return model, tokenizer


def load_processed_dataset(data_dir='data/processed'):
    """載入已處理的資料集"""
    print(f"[DIR] 載入資料集: {data_dir}")

    try:
        dataset = load_from_disk(data_dir)
        print(f"[OK] 資料集載入完成")
        print(f"   訓練樣本: {len(dataset['train'])}")
        print(f"   驗證樣本: {len(dataset['val'])}")
        return dataset
    except Exception as e:
        print(f"[ERROR] 資料集載入失敗: {e}")
        print(f"請先執行資料前處理: python preprocess.py")
        return None


def train(
    model_name='Langboat/bloom-389m-zh',
    data_dir='data/processed',
    output_dir='models/bloom-absa',
    epochs=1,
    batch_size=4,
    learning_rate=2e-5,
    max_steps=-1
):
    """訓練模型"""
    print("[START] 開始訓練...")

    # 載入模型和 tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # 載入資料集
    dataset = load_processed_dataset(data_dir)
    if dataset is None:
        return

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM
    )

    # 訓練參數
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
        max_steps=max_steps
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # 訓練
    print(f"🏋️ 開始訓練...")
    train_result = trainer.train()

    # 儲存模型
    print(f"[SAVE] 儲存模型至 {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 顯示結果
    print(f"\n[OK] 訓練完成！")
    print(f"   最終 Loss: {train_result.training_loss:.4f}")
    print(f"   訓練時間: {train_result.metrics['train_runtime']:.2f} 秒")

    return trainer


def main():
    parser = argparse.ArgumentParser(description='訓練 BLOOM ABSA 模型')
    parser.add_argument('--model', type=str,
                       default='Langboat/bloom-389m-zh',
                       help='預訓練模型名稱')
    parser.add_argument('--data-dir', type=str,
                       default='data/processed',
                       help='處理後的資料集路徑')
    parser.add_argument('--output-dir', type=str,
                       default='models/bloom-absa',
                       help='模型輸出路徑')
    parser.add_argument('--epochs', type=int, default=1,
                       help='訓練 epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='學習率')
    parser.add_argument('--max-steps', type=int, default=-1,
                       help='最大訓練步數 (-1 為不限制)')

    args = parser.parse_args()

    train(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps
    )

    print("[DONE] 完成！")


if __name__ == '__main__':
    main()
