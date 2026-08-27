# Dota Draft AI

Experimental neural-network draft assistant for **Dota 2**.

The current version recommends a fifth hero for your team from an incomplete draft:

```text
Your team: 4 known heroes + one hero to recommend
Enemy:     4 known heroes + one unknown hero
```

The model uses:

- hero embeddings;
- positions 1–5;
- pick order;
- average MMR;
- MMR bracket;
- 25 hero-vs-hero interaction vectors;
- partial-draft training with one masked hero;
- mirrored evaluation to reduce Radiant/Dire side bias.

> This is an experimental ML project, not a production esports drafting engine.

## Architecture

Each hero slot is represented as:

```text
Hero embedding
+
Position embedding
+
Pick-order embedding
```

The full model combines:

```text
10 hero slots
+
25 cross-team hero interactions
+
average MMR
+
rank bracket embedding
```

To support incomplete drafts, the training pipeline randomly hides one hero in many training examples:

```text
hero = 0
position = 0
pick_order = 0
```

This lets the network learn partial draft states instead of seeing only complete 5v5 drafts.

## Repository structure

```text
dota_draft_ai/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   └── workflows/
├── data/
├── models/
├── scripts/
├── src/
│   ├── collect.py
│   ├── collect_detailed.py
│   ├── collect_positions.py
│   ├── hero_roles.py
│   ├── nn_data_v7.py
│   ├── nn_model_v7.py
│   ├── nn_predictor_v8.py
│   ├── recommender_v8.py
│   ├── train_nn_v8.py
│   └── recommend_app_v8.py
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt
```

## Requirements

Recommended:

- Python 3.11 or 3.12
- Windows, Linux, or macOS
- NVIDIA GPU optional
- Internet access for match collection

PyTorch can run on CPU if CUDA is unavailable.

## Installation

Clone your repository:

```bash
git clone https://github.com/YOUR_USERNAME/dota-draft-ai.git
cd dota-draft-ai
```

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Optional OpenDota API key

Copy:

```text
.env.example
```

to:

```text
.env
```

and put your key there:

```text
OPENDOTA_API_KEY=your_key_here
```

Never commit `.env`.

## Data pipeline

### 1. Collect recent matches

The project currently keeps a fixed window of about **30 days**.

```bash
python -m src.collect --pages 300 --sleep 2
```

Output:

```text
data/matches.csv
```

### 2. Collect detailed match data

Quick test:

```bash
python -m src.collect_detailed --matches 500 --sleep 2
```

Larger dataset:

```bash
python -m src.collect_detailed --matches 10000 --sleep 2
```

Detailed responses are cached in:

```text
data/match_details/
```

Already downloaded match details are reused.

Output:

```text
data/detailed_matches.csv
```

### 3. Collect position statistics

```bash
python -m src.collect_positions --matches 500 --sleep 2
```

This is used for candidate role filtering.

## Train the partial-draft model

```bash
python -m src.train_nn_v8
```

Training reports:

```text
train_loss
full_logloss
partial_logloss
partial_auc
partial_acc
```

The most important metrics for the current use case are:

```text
partial_logloss
partial_auc
```

Generated artifacts:

```text
models/draft_nn_v8.pt
models/draft_nn_v8_meta.pt
models/nn_v8_metrics.json
```

These files are ignored by Git because they are local/model artifacts.

## Run the Streamlit UI

```bash
python -m streamlit run src/recommend_app_v8.py
```

Open:

```text
http://localhost:8501
```

The interface asks for:

- 4 heroes on your team;
- positions of those heroes;
- their pick order;
- 4 known enemy heroes;
- enemy positions;
- enemy pick order;
- the target role for your fifth hero;
- average MMR;
- MMR bracket.

The fifth enemy hero remains unknown.

## Metric guide

### `train_loss`

Loss on training data.

Lower is better, but a very low value with worse validation metrics indicates overfitting.

### `partial_logloss`

Probability loss on the incomplete-draft validation set.

Lower is better.

A random 50/50 classifier is around:

```text
0.693
```

### `partial_auc`

Ranking quality:

```text
0.50 = random
0.55 = weak but real signal
0.60+ = increasingly useful
1.00 = perfect
```

### `partial_acc`

Binary accuracy with a 50% decision threshold.

Useful, but less informative than log loss and AUC for probability ranking.

## Early stopping

The trainer saves the best model state according to partial-draft validation loss.

This protects against the common pattern:

```text
train_loss decreases
partial_logloss increases
```

If you experiment with hundreds of epochs, keep restoration of `best_state` enabled.

## Training data size

Very rough development targets:

```text
2,500 matches   = proof of concept
10,000 matches  = useful experiment
25,000+         = better validation
50,000+         = much more interesting
```

More data is usually more valuable than simply adding more epochs.

## Files intentionally excluded from Git

The `.gitignore` excludes:

```text
.venv/
.env
data/*.csv
data/*.json
data/match_details/
models/*.pt
models/*.json
```

This keeps the repository small and prevents accidental publication of credentials or large local artifacts.

If you want to publish a pretrained model later, use **GitHub Releases** rather than committing large `.pt` files directly.

## Data source and disclaimer

The project uses public Dota 2 match information obtained through OpenDota-compatible APIs.

Respect API rate limits and applicable terms of use.

This project is not affiliated with Valve, OpenDota, or Dotabuff.

Dota and Dota 2 are trademarks of Valve Corporation.

## Roadmap

- incremental dataset updater;
- safe parallel match downloader;
- facets/aspects;
- explicit patch features;
- lane matchup features;
- attention/transformer draft encoder;
- probability calibration;
- recommendation explanations;
- automated evaluation dashboard;
- pretrained model releases.

## License

MIT. See `LICENSE`.
