# Senior Training — Hotel Review ABSA to Alpaca JSON Preprocessing

## Task Overview

Convert a structured hotel review ABSA (Aspect-Based Sentiment Analysis) CSV dataset into the **Alpaca instruction-tuning JSON format** so it can be used to fine-tune a language model.

**Task type**: Data preprocessing / dataset format conversion
**Input**: Hotel review CSV with multi-label sentiment annotations (16,027 rows)
**Output**: Alpaca-format JSON ready for LLM instruction fine-tuning

---

## Background

A hotel review CSV was manually annotated with the following columns:

| Column | Description |
|---|---|
| `comment_text` | Raw hotel review text |
| `整潔舒適面向` | Cleanliness/comfort aspect present (0/1) |
| `設施面向` | Facilities aspect present (0/1) |
| `服務面向` | Service aspect present (0/1) |
| `地點面向` | Location aspect present (0/1) |
| `性價比面向` | Value-for-money aspect present (0/1) |
| `其他面向` | Other aspects (0/1) |
| `整潔舒適情緒` / `設施情緒` / ... | Sentiment: 1=positive, 2=neutral, 3=negative |
| `整體情緒` | Overall sentiment |

**Total rows**: 16,027 hotel reviews, no null values in the target columns.

---

## Pipeline

### Step 1 — Load and Inspect CSV

```python
import pandas as pd
df = pd.read_csv('hotel-review-飯店留言合併1.6萬筆.csv', delimiter=',')
print(df.columns)
# Index(['comment_text', '整潔舒適面向', '設施面向', '服務面向', '地點面向', '性價比面向', '其他面向',
#        '整潔舒適情緒', '設施情緒', '服務情緒', '地點情緒', '性價比情緒', '其他情緒', '整體情緒'])
```

### Step 2 — Format Aspect-Sentiment Output

Each row is converted to a structured answer string:

```python
emotion_mapping = {0: "無情緒", 1: "正面", 2: "中立", 3: "負面"}

for index, row in df.iterrows():
    result = "這段文字是屬於:"
    for category in categories:
        if row[category] != 0:
            if '面向' in category:
                result += f"{category},"           # aspect present
            elif '情緒' in category:
                emotion = emotion_mapping[row[category]]
                result += f"{category}: {emotion} " # sentiment label
```

Example output for a row:

`設施面向, 設施情緒: 負面, 整體情緒: 負面`

### Step 3 — Save to `output_results.txt`

All 16,027 formatted lines written to a text file (one per row).

### Step 4 — Merge into Alpaca JSON

The Alpaca format has three fields per entry: `instruction`, `input`, `output`.

```python
# instruction: fixed question prompt
item['instruction'] = "以下這段文字提到那些面向?是正面還是負面的情緒?"

# input: hotel review text from CSV comment_text column
item['input'] = comments[i]

# output: formatted aspect-sentiment string from output_results.txt
item['output'] = output_results_line
```

The base JSON template (`alpaca_gpt4_data_zh_copy.json`) is updated in-place.

---

## File Structure

```
senior-training/
├── 05-hotel-review-absa-to-alpaca-json.ipynb        # main preprocessing notebook
└── 05-hotel-review-absa-to-alpaca-json-copy.ipynb   # copy/variant
```

External files referenced (not in repo — too large):
- `hotel-review-飯店留言合併1.6萬筆.csv` (~hotel reviews CSV, 16K rows)
- `alpaca_gpt4_data_zh_copy.json` (~34 MB Alpaca template)
- `output_results.txt` (intermediate formatted output)

---

## How to Run

1. Place the hotel review CSV and Alpaca JSON template in the paths specified in the notebook
2. Run `05-hotel-review-absa-to-alpaca-json.ipynb` top to bottom
3. Output: updated `alpaca_gpt4_data_zh_copy.json` with hotel review ABSA training pairs

---

## Key Observations

- The CSV has 16,027 rows but the Alpaca JSON template may have a different number of entries. The script uses `min(i, len(data))` to avoid index out of range.
- Some cells have `0` for all sentiment columns (neutral / no specific aspect mentioned) — these rows produce minimal output strings.
- The notebook was developed across multiple machines (paths reference `a0936` and `USER` home directories) — update paths before running.
- This dataset was used as the `input/output` pairs for fine-tuning Bloom-389m on ABSA tasks (see `assignment-03`).
