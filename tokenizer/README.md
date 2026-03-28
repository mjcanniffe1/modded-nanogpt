# Morphological BPE Tokenizer — Quick Start

A BPE tokenizer that force-merges ~80 English morphemes (prefixes/suffixes like `un-`, `-tion`, `-able`) into dedicated tokens before running standard statistical BPE. This costs ~0.2% of the vocabulary but gives the model explicit compositional building blocks for reasoning.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install tiktoken regex openpyxl datasets
```

## 1. Train a Tokenizer

```bash
# Quick test (~5 min on CPU, 10M chars)
python tokenizer/train.py --morphemes --vocab_size=4096 --max_chars=10_000_000

# Full training on FineWeb (~1B chars, matching GPT-2 vocab size)
python tokenizer/train.py --morphemes --vocab_size=50257 --max_chars=1_000_000_000

# Vanilla BPE (no morphemes, for A/B comparison)
python tokenizer/train.py --vocab_size=50257 --output_dir=tokenizer/output_vanilla

# Custom morpheme list
python tokenizer/train.py --morpheme_file=my_morphemes.txt --vocab_size=50257
```

Output goes to `tokenizer/output/` (or `--output_dir`):
- `tokenizer.pkl` — tiktoken encoding (used for tokenization and training)
- `meta.json` — training metadata (vocab size, morpheme list, merge counts)
- `mergeable_ranks.json` — full token vocabulary for inspection

## 2. Evaluate the Tokenizer (no GPU needed)

```bash
# Clone gold-standard morphological data (one time)
git clone https://github.com/hugomailhot/MorphoLex-en.git evals/MorphoLex-en

# Evaluate against MorphoLex + compare to GPT-2 baseline
python evals/eval_tokenizer.py \
    --tokenizer_path=tokenizer/output/tokenizer.pkl \
    --baseline_tokenizer=gpt2
```

This reports:
- **Morphological boundary F1** — do token splits align with real morpheme boundaries?
- **Token fertility** — avg tokens per word (lower = more compact)
- **Compression ratio** — bytes per token
- **Morpheme utilization** — which forced morphemes are actually used vs wasted

## 3. Re-tokenize FineWeb

```bash
python data/tokenize_fineweb_morphological.py \
    --tokenizer_path=tokenizer/output/tokenizer.pkl \
    --version=10B

# Quick test (5 shards only)
python data/tokenize_fineweb_morphological.py \
    --tokenizer_path=tokenizer/output/tokenizer.pkl \
    --version=10B --max_shards=5
```

This produces `.bin` shards in `data/fineweb10B_morph/` in the exact same format that `train_gpt.py` expects.

## 4. Train a Model

`train_gpt.py` is configured via environment variables — no code changes needed:

```bash
# Check meta.json for your tokenizer's eot_token_id
cat tokenizer/output/meta.json | grep eot_token_id

# Train with morphological tokenizer
VOCAB_SIZE=50257 \
BOS_ID=50256 \
TRAIN_FILES="data/fineweb10B_morph/fineweb_train_*.bin" \
VAL_FILES="data/fineweb10B_morph/fineweb_val_*.bin" \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Set `BOS_ID` to the `eot_token_id` from your tokenizer's `meta.json`.

## 5. A/B Comparison

To compare morphological vs vanilla tokenization end-to-end:

```bash
# Train both tokenizers
python tokenizer/train.py --morphemes --vocab_size=50257 --output_dir=tokenizer/output_morpheme
python tokenizer/train.py --vocab_size=50257 --output_dir=tokenizer/output_vanilla

# Intrinsic eval (CPU, ~2 min)
python evals/eval_tokenizer.py \
    --tokenizer_path=tokenizer/output_morpheme/tokenizer.pkl \
    --baseline_tokenizer=gpt2

# Re-tokenize data with each
python data/tokenize_fineweb_morphological.py \
    --tokenizer_path=tokenizer/output_morpheme/tokenizer.pkl \
    --output_dir=data/fineweb10B_morpheme
python data/tokenize_fineweb_morphological.py \
    --tokenizer_path=tokenizer/output_vanilla/tokenizer.pkl \
    --output_dir=data/fineweb10B_vanilla

# Train model A (original GPT-2 tokenizer — just run normally)
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Train model B (morphological tokenizer)
VOCAB_SIZE=50257 BOS_ID=50256 \
TRAIN_FILES="data/fineweb10B_morpheme/fineweb_train_*.bin" \
VAL_FILES="data/fineweb10B_morpheme/fineweb_val_*.bin" \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Compare val loss curves between the two runs
```

## How It Works

Standard BPE builds tokens purely by frequency — "unhappiness" might split as `unh|app|iness`. The morphological tokenizer forces ~80 English morphemes to be merged first:

```
Standard BPE:       "unhappiness" -> ["unh", "app", "iness"]
Morphological BPE:  "unhappiness" -> ["un", "happi", "ness"]
```

The forced merges consume the first ~131 token IDs (256-386), using ~0.2% of a 65K vocabulary. The remaining 99.8% trains statistically as normal BPE.

Morphemes are sorted by byte length before merging, so shorter morphemes like `in` are merged before longer ones like `inter`, enabling prefix sharing (reusing the `in` merge).

## File Overview

```
tokenizer/
  morphological_bpe.py    # BPE trainer with forced morpheme merges
  morphemes.py            # ~80 English morphemes (prefixes + suffixes)
  train.py                # CLI to train a tokenizer
  tests/
    test_morphological_bpe.py  # 8 tests (run: python -m pytest tokenizer/tests/ -v)
data/
  tokenize_fineweb_morphological.py  # Re-tokenize FineWeb with trained tokenizer
evals/
  eval_tokenizer.py       # Tier 1 intrinsic evaluation
train_gpt.py              # Modified: VOCAB_SIZE, BOS_ID, TRAIN_FILES, VAL_FILES env vars
```
