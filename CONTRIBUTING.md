# Contributing

## Setup

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
python -m pip install -r requirements.txt
```

## Pull requests

- Do not commit API keys, datasets, caches, or trained model binaries.
- Keep implementation code inside `src/`.
- If you change the model, include before/after validation metrics.
- For the current use case, report `partial_logloss`, `partial_auc`, and `partial_acc`.
