"""
Crypto Oracle — Model Validation Script
=========================================
Run this before deploying a new model to verify the files in
model/current/ are well-formed and loadable.

Usage:
    python validate_model.py

Exit codes:
    0 — all checks passed
    1 — one or more checks failed
"""

import os
import sys
import json

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "current")

REQUIRED_KEYS_METADATA = [
    "trained_at", "model_type", "hyperparameters", "target_threshold",
    "coins", "test_macro_f1", "test_accuracy", "confidence_gating",
    "label_mapping",
]


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    print("=" * 60)
    print("MODEL VALIDATION")
    print(f"Directory: {MODEL_DIR}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Check directory exists
    # ------------------------------------------------------------------
    if not os.path.isdir(MODEL_DIR):
        print(f"\n[FAIL] model/current/ directory does not exist: {MODEL_DIR}")
        return 1

    # ------------------------------------------------------------------
    # 2. Check model file is loadable by XGBoost
    # ------------------------------------------------------------------
    # Look for any .json file that could be the model
    model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("model") and f.endswith(".json")]
    if not model_files:
        errors.append("No model*.json file found in model/current/")
    else:
        model_path = os.path.join(MODEL_DIR, model_files[0])
        print(f"\n[CHECK] Model file: {model_files[0]}")
        try:
            import xgboost as xgb
            m = xgb.XGBClassifier()
            m.load_model(model_path)
            n_classes = m.n_classes_
            print(f"  [OK] Loaded successfully -- {n_classes} classes")
        except ImportError:
            warnings.append("xgboost not installed -- cannot validate model loading")
            print("  [WARN] xgboost not installed, skipping load check")
        except Exception as exc:
            errors.append(f"Model failed to load: {exc}")
            print(f"  [FAIL] Load failed: {exc}")

    # ------------------------------------------------------------------
    # 3. Check feature_cols.json
    # ------------------------------------------------------------------
    fc_path = os.path.join(MODEL_DIR, "feature_cols.json")
    print(f"\n[CHECK] feature_cols.json")
    if not os.path.exists(fc_path):
        errors.append("feature_cols.json not found")
        print("  ✗ File not found")
    else:
        try:
            with open(fc_path) as f:
                cols = json.load(f)
            if not isinstance(cols, list):
                errors.append("feature_cols.json is not a JSON list")
                print("  [FAIL] Not a JSON list")
            elif len(cols) == 0:
                errors.append("feature_cols.json is empty")
                print("  [FAIL] Empty list")
            else:
                print(f"  [OK] {len(cols)} feature columns")
                # Check for duplicates
                dupes = [c for c in cols if cols.count(c) > 1]
                if dupes:
                    warnings.append(f"Duplicate feature columns: {set(dupes)}")
                    print(f"  [WARN] Duplicates found: {set(dupes)}")
        except json.JSONDecodeError as exc:
            errors.append(f"feature_cols.json is not valid JSON: {exc}")
            print(f"  [FAIL] Invalid JSON: {exc}")

    # ------------------------------------------------------------------
    # 4. Check metadata.json
    # ------------------------------------------------------------------
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    print(f"\n[CHECK] metadata.json")
    if not os.path.exists(meta_path):
        errors.append("metadata.json not found")
        print("  [FAIL] File not found")
    else:
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if not isinstance(meta, dict):
                errors.append("metadata.json is not a JSON object")
                print("  [FAIL] Not a JSON object")
            else:
                print(f"  [OK] Valid JSON object")
                # Check required keys
                missing_keys = [k for k in REQUIRED_KEYS_METADATA if k not in meta]
                if missing_keys:
                    warnings.append(f"Missing metadata keys: {missing_keys}")
                    print(f"  [WARN] Missing keys: {missing_keys}")
                else:
                    print(f"  [OK] All required keys present")

                # Report key metadata
                if "trained_at" in meta:
                    print(f"  [INFO] Trained at: {meta['trained_at']}")
                if "test_macro_f1" in meta:
                    print(f"  [INFO] Test macro-F1: {meta['test_macro_f1']}")
                if "test_accuracy" in meta:
                    print(f"  [INFO] Test accuracy: {meta['test_accuracy']}")
                if "label_mapping" in meta:
                    print(f"  [INFO] Label mapping: {meta['label_mapping']}")

        except json.JSONDecodeError as exc:
            errors.append(f"metadata.json is not valid JSON: {exc}")
            print(f"  [FAIL] Invalid JSON: {exc}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if errors:
        print(f"RESULT: FAILED -- {len(errors)} error(s)")
        for e in errors:
            print(f"  [FAIL] {e}")
    else:
        print("RESULT: PASSED")

    if warnings:
        print(f"\n  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  [WARN] {w}")

    print("=" * 60)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
