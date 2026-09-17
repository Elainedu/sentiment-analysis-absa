# sentiment-analysis-absa

**Aspect-Based Sentiment Analysis (ABSA)** for Chinese hotel reviews,
built on a full-SFT fine-tuned `Langboat/bloom-389m-zh`. Generates
structured `aspect-sentiment` pairs (six aspects × positive/neutral/
negative) for any given review.

---

## Table of Contents

- [What This Does](#what-this-does)
- [Model & Dataset](#model--dataset)
- [Training Setup](#training-setup)
- [System Architecture](#system-architecture)
- [Repository Layout](#repository-layout)
- [Key Files](#key-files)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Reproducing Training](#reproducing-training)
- [Running Inference](#running-inference)
- [Notes](#notes)
- [Files Not in Repo](#files-not-in-repo)
- [References](#references)
- [License](#license)

---

## What This Does

Traditional sentiment analysis assigns a single polarity to a whole
review. **ABSA** goes further: it identifies the individual **aspects**
mentioned in a review and predicts the sentiment for each one
separately.

This project fully fine-tunes `Langboat/bloom-389m-zh` on a
Chinese hotel-review ABSA dataset (~16 K samples in Alpaca format) so
that the model can return structured aspect–sentiment pairs.

Example (Chinese input/output, English gloss):

```
Input : The room was very clean and the staff were friendly,
        but the location is a bit remote.
Output: cleanliness-positive, service-positive, location-negative
```

**Six aspects** the dataset covers:

- Cleanliness / comfort
- Facilities
- Service
- Location
- Value for money
- Other

Each aspect is classified as **positive**, **neutral**, or **negative**.

---

## Model & Dataset

| Item              | Value |
| ----------------- | ----- |
| Base model        | `Langboat/bloom-389m-zh` |
| Fine-tuning       | Full SFT (all 389 M params updated) |
| Task type         | Causal LM (autoregressive generation) |
| Dataset           | Chinese hotel review ABSA (~16 K samples) |
| Train / Val split | ~14 K / ~2 K |
| Data format       | Alpaca-style JSON |
| Framework         | PyTorch + HuggingFace Transformers |
| UI                | Gradio |

**Data format (Alpaca-style, Chinese fields shown as English gloss):**

```json
{
  "instruction": "Which aspects are mentioned in the text and what is the sentiment of each?",
  "input":  "The room was clean, service was great, location was convenient, good value for money",
  "output": "cleanliness-positive, service-positive, location-positive, value-positive"
}
```

(In the real dataset, `instruction` / `input` / `output` are written in
Simplified Chinese; the model is trained to reproduce the same Chinese
aspect–sentiment vocabulary.)

Raw CSV → Alpaca JSON → tokenised HF `Dataset` (Arrow). The preprocessing
notebooks under `notebooks/` produce `data/processed/` as the training
input.

---

## Training Setup

Method: **Full SFT** (no LoRA / no quantisation) — every parameter is
updated.

Actual `TrainingArguments` from `train.py`:

```python
TrainingArguments(
    output_dir              = "models/bloom-absa",
    num_train_epochs        = 1,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size  = 4,
    learning_rate           = 2e-5,
    weight_decay            = 0.01,
    warmup_steps            = 100,
    logging_steps           = 50,
    eval_strategy           = "steps",
    eval_steps              = 200,
    save_strategy           = "steps",
    save_steps              = 200,
    save_total_limit        = 3,
    load_best_model_at_end  = True,
    metric_for_best_model   = "eval_loss",
    fp16                    = torch.cuda.is_available(),
    report_to               = "none",
)
```

CLI flags (`--epochs`, `--batch-size`, `--learning-rate`,
`--output-dir`, `--data-dir`, `--model`, `--max-steps`) let you override
the defaults.

| Setting              | Value |
| -------------------- | ----- |
| Epochs               | 1 (default) |
| Per-device batch     | 4 |
| Learning rate        | 2e-5 |
| Weight decay         | 0.01 |
| Warm-up steps        | 100 |
| LR schedule          | default linear w/ warm-up |
| Precision            | fp16 on CUDA, fp32 otherwise |
| Padding side         | **left** (BLOOM requirement) |
| GPU tested           | NVIDIA T4 (~10 GB VRAM) |
| Wall time            | ~28 min / epoch on a T4 |
| Best-checkpoint loss | ~0.13 (train), ~0.13 (eval) at ~step 600 |

Reference training curve:

| Step | Train Loss | Eval Loss |
| ---- | ---------- | --------- |
| 0    | ~2.5       | ~2.4      |
| 200  | ~0.8       | ~0.6      |
| 400  | ~0.3       | ~0.25     |
| 600  | ~0.15      | **0.13**  |

---

## System Architecture

```
                ┌──────────────────────────────────────┐
                │  Raw hotel-review CSV (~16K rows)    │
                │  data/raw/hotel-review-combined.csv  │
                └───────────────┬──────────────────────┘
                                │  notebooks/05-*-alpaca-json
                                ▼
                ┌──────────────────────────────────────┐
                │  Alpaca-style JSON                   │
                │  (instruction / input / output)      │
                └───────────────┬──────────────────────┘
                                │  notebooks/10-14-*-bloom-tokenize
                                ▼
                ┌──────────────────────────────────────┐
                │  HF Arrow dataset                    │
                │  data/processed/  (train / val)      │
                └───────────────┬──────────────────────┘
                                │  BloomTokenizerFast (left pad)
                                ▼
    ┌───────────────────────────────────────────────────────┐
    │  train.py                                             │
    │   - AutoModelForCausalLM(Langboat/bloom-389m-zh)      │
    │   - DataCollatorForLanguageModeling(mlm=False)        │
    │   - TrainingArguments(fp16, lr=2e-5, epochs=1, bs=4)  │
    │   - Trainer.train() -> save_model(models/bloom-absa)  │
    └───────────────┬───────────────────────────────────────┘
                    │  checkpoint (~1.5 GB)
                    ▼
    ┌───────────────────────────────────────────────────────┐
    │  demo.py (Gradio)                                     │
    │   - AutoModelForCausalLM.from_pretrained(models/...)  │
    │   - Instruction-style prompt template                 │
    │   - Sliders: temperature, top_p, max_new_tokens       │
    └───────────────┬───────────────────────────────────────┘
                    │
                    ▼
             http://localhost:7860
```

---

## Repository Layout

```
sentiment-analysis-absa/
├── train.py                        # SFT training entry point
├── demo.py                         # Gradio ABSA demo
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                        # original hotel-review CSV
│   └── processed/                  # tokenised HF Arrow dataset (gitignored)
├── models/
│   └── bloom-absa/                 # trained checkpoints (~1.5 GB, gitignored)
├── configs/                        # YAML configs (optional overrides)
├── src/                            # data / model utility modules
└── notebooks/
    ├── 05-hotel-review-absa-to-alpaca-json.ipynb           # CSV -> Alpaca JSON
    ├── 05-hotel-review-absa-to-alpaca-json-copy.ipynb      # variant
    ├── 10-bloom-tokenizer-alpaca-chinese-setup.ipynb       # tokenizer setup
    ├── 10-14-hotel-review-absa-to-bloom-tokenize.ipynb     # tokenisation -> Arrow
    ├── bloom389m-sft-full-finetune-human-assistant-dialog.ipynb  # reference SFT notebook
    ├── hotel-review-combined-16k.csv                       # cleaned dataset (~2 MB)
    ├── hotel-review-<CN-original-name>.csv                  # original CSV filename (Chinese, kept as shipped)
    ├── alpaca_gpt4_data_zh.json / (1).json                 # auxiliary Chinese Alpaca data
    ├── TraditionalChinese.json                             # trad-Chinese variant
    ├── output_results.json / .txt / copy.json              # sample generations
    └── train_dataset/                                      # scratch tokenised split
```

---

## Key Files

| File | Purpose |
| ---- | ------- |
| `train.py` | Full-SFT trainer. Loads `Langboat/bloom-389m-zh`, sets `tokenizer.padding_side = 'left'`, loads `data/processed` via `datasets.load_from_disk`, runs `Trainer` with fp16-on-CUDA, saves to `models/bloom-absa/`. CLI flags: `--model`, `--data-dir`, `--output-dir`, `--epochs`, `--batch-size`, `--learning-rate`, `--max-steps`. |
| `demo.py` | Gradio demo. Loads a trained ABSA checkpoint, wraps user text in a Chinese instruction template ("Which aspects are mentioned and what is the sentiment of each?"), and streams generation. CLI flags: `--model`, `--share`, `--port`. |
| `requirements.txt` | Minimum runtime + training deps. |
| `notebooks/05-hotel-review-absa-to-alpaca-json*.ipynb` | Turn the raw hotel-review CSV into Alpaca-format JSON (`instruction`/`input`/`output`). |
| `notebooks/10-bloom-tokenizer-alpaca-chinese-setup.ipynb` | BLOOM tokenizer sanity checks for Chinese Alpaca inputs. |
| `notebooks/10-14-hotel-review-absa-to-bloom-tokenize.ipynb` | Tokenises the Alpaca JSON with `BloomTokenizerFast`, writes the HF Arrow dataset to `data/processed/`. |
| `notebooks/bloom389m-sft-full-finetune-human-assistant-dialog.ipynb` | Reference notebook for the same full-SFT recipe on general dialog — kept for comparison with the ABSA task. |

---

## Requirements

From `requirements.txt`:

```
torch>=2.0.0
transformers>=4.32.0
datasets>=2.14.0
accelerate>=0.23.0
gradio>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
pyyaml>=6.0
```

Hardware:

- Python 3.10+, CUDA 11.8+ (for training)
- **~10 GB GPU VRAM** for full-SFT training (T4 / V100 / A100)
- 16 GB+ system RAM
- CPU inference works but is slow

Install:

```bash
pip install -r requirements.txt
```

---

## Configuration

No environment variables. All settings are either CLI flags to
`train.py` / `demo.py` or hard-coded defaults inside `train.py`:

| Flag             | Default | Description |
| ---------------- | ------- | ----------- |
| `--model`        | `Langboat/bloom-389m-zh` | Pretrained base model |
| `--data-dir`     | `data/processed`         | Tokenised HF Arrow dataset |
| `--output-dir`   | `models/bloom-absa`      | Where checkpoints go |
| `--epochs`       | `1`                      | Training epochs |
| `--batch-size`   | `4`                      | Per-device batch size |
| `--learning-rate`| `2e-5`                   | LR (linear + warm-up) |
| `--max-steps`    | `-1`                     | Cap total training steps (`-1` = uncapped) |

`configs/` contains YAML overrides that can be loaded by user code (the
default trainer reads its config from CLI flags, not from YAML).

---

## Reproducing Training

```bash
# 1. Clone
git clone https://github.com/Elainedu/sentiment-analysis-absa.git
cd sentiment-analysis-absa

# 2. Install deps
pip install -r requirements.txt

# 3. Prepare the dataset
#    Run these notebooks in order:
#      notebooks/05-hotel-review-absa-to-alpaca-json.ipynb
#      notebooks/10-14-hotel-review-absa-to-bloom-tokenize.ipynb
#    They write the tokenised Arrow dataset into data/processed/
#    (must contain a "train" and "val" split).

# 4. Train
python train.py                           # 1 epoch, bs=4, lr=2e-5, defaults
python train.py --epochs 2 --batch-size 4 # custom
python train.py --output-dir models/my-bloom-absa

# 5. Serve
python demo.py --model models/bloom-absa
```

Training writes checkpoints to `models/bloom-absa/checkpoint-*` and, at
the end, `save_model()` puts the best checkpoint at the root of the
output directory.

---

## Running Inference

```bash
# Default model path (models/bloom-absa)
python demo.py

# Custom model
python demo.py --model models/my-bloom-absa

# Public share link
python demo.py --share

# Custom port
python demo.py --port 8080
```

The demo runs on `http://localhost:7860` and exposes an input textbox,
temperature / top-p sliders, quick example reviews, and a live result
panel.

Prompt template used by `demo.py` (English gloss of the Chinese
template that the model actually receives):

```
Which aspects are mentioned in the following text?
Is the sentiment positive or negative?
{review}
This text belongs to:
```

The model then generates a short comma-separated list of
`aspect-sentiment` pairs.

---

## Notes

- **Padding side.** BLOOM tokenizer requires **left**-side padding for
  causal-LM training — do **not** flip it to right (unlike LLaMA-style
  models).
- **Batch size.** Kept at 4 because of GPU memory constraints for full
  fine-tuning; use gradient accumulation to raise the effective batch.
- **Convergence.** Loss drops from ~2.5 to ~0.13 within a single epoch
  (~28 min on a T4). One epoch is generally enough.
- **Checkpoint size.** A trained checkpoint is around **1.5 GB**, so
  weights are **not** committed to the repo — retrain locally to reproduce.
- **Data quality matters.** ABSA labels are relatively noisy; cleaning
  the raw annotations improves final F1 notably.

---

## Files Not in Repo

| Excluded                         | Size    | How to obtain / regenerate |
| -------------------------------- | ------- | -------------------------- |
| `models/bloom-absa/` checkpoint  | ~1.5 GB | Retrain via `python train.py`. |
| `data/processed/` Arrow dataset  | ~50 MB  | Rerun the two preprocessing notebooks under `notebooks/`. |
| `data/raw/` hotel-review CSV     | ~2 MB   | A copy exists at `notebooks/hotel-review-combined-16k.csv`. |
| `Langboat/bloom-389m-zh` weights | ~1.65 GB| Auto-downloaded by `transformers` on first run. |

---

## References

- Base model: <https://huggingface.co/Langboat/bloom-389m-zh>
- Le Scao et al. **BLOOM: A 176B-Parameter Open-Access Multilingual
  Language Model.** arXiv:2211.05100. <https://arxiv.org/abs/2211.05100>
- Zhang et al. **A Survey on Aspect-Based Sentiment Analysis.**
  arXiv:2203.01054. <https://arxiv.org/abs/2203.01054>
- Alpaca instruction format:
  <https://github.com/tatsu-lab/stanford_alpaca>
- HuggingFace `Trainer`:
  <https://huggingface.co/docs/transformers/main_classes/trainer>

---

## License

Released for educational use as part of a deep-learning course project.
Underlying model and dataset licenses belong to their original authors
(BLOOM RAIL License for BLOOM-derived checkpoints; dataset terms follow
the hotel-review corpus source).
