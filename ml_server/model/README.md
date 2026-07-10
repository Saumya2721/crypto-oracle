# Model Deployment — Manual Workflow

This directory holds the production model files. The Flask server (`app.py`) loads
from `current/` once at startup — **no code changes are needed to swap model versions**.

## Directory Structure

```
model/
├── current/              ← Active model (app.py reads from here)
│   ├── model_v1.json     ← XGBoost model (saved via save_model(), NOT pickle)
│   ├── feature_cols.json ← JSON list of feature column names, in training order
│   └── metadata.json     ← Training metadata (see schema below)
└── archive/              ← Previous model versions (manual rollback)
    └── 2026-07-10T08-48/ ← Example: timestamped subdirectory
        ├── model_v1.json
        ├── feature_cols.json
        └── metadata.json
```

## Deploying a New Model

1. **Archive the current model:**
   ```bash
   # From ml_server/
   mkdir model/archive/$(date +%Y-%m-%dT%H-%M)
   cp model/current/* model/archive/$(date +%Y-%m-%dT%H-%M)/
   ```

2. **Place the new model files** into `model/current/`:
   - `model_v1.json` — XGBoost model saved with `model.save_model('model_v1.json')`
   - `feature_cols.json` — JSON list of feature column names in exact training order
   - `metadata.json` — see schema below

3. **Restart the Flask server** — it re-loads from `model/current/` on startup.

4. **(Optional) Validate first** — run `python validate_model.py` before restarting
   to check the new files are well-formed.

## Rolling Back

Copy the desired version from `model/archive/<timestamp>/` back into `model/current/`,
then restart the Flask server.

## metadata.json Schema

```json
{
  "trained_at": "ISO-8601 timestamp",
  "model_type": "XGBoost",
  "hyperparameters": { ... },
  "target_threshold": 0.01,
  "coins": ["Bitcoin", "Ethereum", ...],
  "test_macro_f1": 0.39,
  "test_accuracy": 0.40,
  "confidence_gating": {
    "0.5": { "accuracy": 0.41, "coverage_pct": 36.7 },
    "0.6": { "accuracy": 0.43, "coverage_pct": 10.5 }
  },
  "label_mapping": { "0": -1, "1": 0, "2": 1 }
}
```

## Important Notes

- This is a **MANUAL** process for this MVP. There is no automated retraining,
  CI/CD pipeline, or scheduled model refresh.
- Training happens in Colab. The resulting files are manually placed here.
- The model file MUST be saved with XGBoost's native `save_model()` format,
  NOT pickle. This ensures cross-platform compatibility.
