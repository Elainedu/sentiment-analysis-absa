# BLOOM Sentiment Analysis (ABSA)

Aspect-Based Sentiment Analysis for hotel reviews, powered by a fully
fine-tuned BLOOM-389M Chinese language model.

## Overview

Traditional sentiment analysis only assigns a single polarity to a whole
review. ABSA (Aspect-Based Sentiment Analysis) goes further: it identifies
the individual aspects mentioned in a review and predicts the sentiment for
each of them separately.

This project fine-tunes `Langboat/bloom-389m-zh` on a Chinese hotel review
ABSA dataset (~16K samples) so that the model can return structured
`aspect-sentiment` pairs for any given review.

Example:

```
Input : "The room was very clean and the staff were friendly,
         but the location is a bit remote."
Output: cleanliness-positive, service-positive, location-negative
```

The six aspects covered by the dataset:

- Cleanliness / comfort
- Facilities
- Service
- Location
- Value for money
- Other

Each aspect is classified as **positive**, **neutral**, or **negative**.

## Model and Dataset

| Item | Value |
|------|-------|
| Base model | `Langboat/bloom-389m-zh` |
| Fine-tuning | Full Supervised Fine-Tuning (SFT) |
| Task type | Causal LM (autoregressive generation) |
| Dataset | Chinese hotel review ABSA (~16K samples) |
| Train / Val split | ~14K / ~2K |
| Data format | Alpaca-style JSON |
| Framework | PyTorch + HuggingFace Transformers |
| UI | Gradio |

Training data uses the Alpaca instruction format:

```json
{
  "instruction": "Which aspects are mentioned in the text and what is the sentiment of each?",
  "input": "The room was clean, service was great, location was convenient, good value for money",
  "output": "cleanliness-positive, service-positive, location-positive, value-positive"
}
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (recommended for training)
- ~10 GB GPU VRAM (T4 / V100 / A100)
- 16 GB+ system RAM

Python dependencies (see `requirements.txt`):

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

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

Place the processed HuggingFace `Dataset` under `data/processed/`. The
training script loads it via `datasets.load_from_disk`, so both the raw
CSV -> Alpaca JSON conversion and tokenization must be run beforehand.

### 3. Train

```bash
# Default: 1 epoch, batch size 4, lr 2e-5
python train.py

# Custom hyperparameters
python train.py --epochs 2 --batch-size 4 --learning-rate 2e-5

# Custom output directory
python train.py --output-dir models/my-bloom-absa
```

Training arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Langboat/bloom-389m-zh` | Pretrained model name |
| `--data-dir` | `data/processed` | Processed dataset path |
| `--output-dir` | `models/bloom-absa` | Where to save checkpoints |
| `--epochs` | `1` | Number of training epochs |
| `--batch-size` | `4` | Per-device train batch size |
| `--learning-rate` | `2e-5` | Learning rate |

Underlying `TrainingArguments`:

```python
TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=True,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
)
```

### 4. Launch the Gradio demo

```bash
# Default model path
python demo.py

# Custom model
python demo.py --model models/bloom-absa

# Public share link
python demo.py --share

# Custom port
python demo.py --port 8080
```

The demo runs on `http://localhost:7860` by default and exposes an input
textbox, temperature / top-p sliders, quick example reviews, and a live
result panel.

## Files

```
sentiment-analysis-absa/
|-- train.py            # Training entry point
|-- demo.py             # Gradio demo
|-- requirements.txt
|-- README.md
|-- configs/            # YAML configs
|-- data/
|   |-- raw/            # Original CSV
|   `-- processed/      # HuggingFace Arrow dataset
|-- models/
|   `-- bloom-absa/     # Trained checkpoints
|-- notebooks/          # EDA + experiment notebooks
`-- src/                # Data / model utility modules
```

## Notes

- **Padding side**: BLOOM tokenizer requires **left-side padding** for
  causal LM training (different from LLaMA-style right padding).
- **Batch size**: kept at 4 because of GPU memory constraints for full
  fine-tuning; use gradient accumulation to raise the effective batch.
- **Convergence**: with this data volume, loss drops from ~2.5 to ~0.13
  within a single epoch (~28 min/epoch on a T4 GPU).
- **Checkpoint size**: a trained checkpoint is around 1.5 GB, so model
  weights are **not** committed to the repo. Retrain locally to reproduce.
- **Data quality matters**: ABSA labels are relatively noisy; cleaning the
  raw annotations improves final F1 notably.

Reference training curve:

| Step | Train Loss | Eval Loss |
|------|------------|-----------|
| 0    | ~2.5       | ~2.4      |
| 200  | ~0.8       | ~0.6      |
| 400  | ~0.3       | ~0.25     |
| 600  | ~0.15      | **0.13**  |

## References

- Base model: <https://huggingface.co/Langboat/bloom-389m-zh>
- BLOOM paper: <https://arxiv.org/abs/2211.05100>
- ABSA survey: <https://arxiv.org/abs/2203.01054>

## License

Released for educational use as part of a deep learning course project.
Underlying model and dataset licenses belong to their original authors.
