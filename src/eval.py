"""Evaluate trivial line-level bug classifiers on the Herbold dataset.

Self-contained: loads data, computes consensus ground truth, runs 5-fold
cross-validation, prints micro-averaged metrics per classifier.
"""

import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.oracles import (
    heuristic_predict, ml_train, ml_predict, save_model, load_model,
    MODELS_DIR,
)

DATA_PATH = Path("data/herbold.parquet")

MIN_ANNOTATORS = 3
MIN_LINES = 5
MAX_LINES = 200
N_FOLDS = 5
TEST_RATIO = 0.3
SEED = 42


# --- Herbold consensus ground truth ---

def consensus(commit_df: pd.DataFrame) -> list[dict]:
    """Majority-vote consensus labels for a commit's lines."""
    key = ["file", "change_type", "old_line_idx", "new_line_idx", "line_content"]
    results = []
    for group_key, group in commit_df.groupby(key, dropna=False):
        labels = group["label"].tolist()
        counts = Counter(labels)
        top_label, top_count = counts.most_common(1)[0]
        ct = group_key[1]
        idx = group_key[2] if ct == "-" else group_key[3]
        binary = "bug" if top_label == "bugfix" else "unrelated"
        results.append({
            "file": group_key[0],
            "change_type": ct,
            "old_line_idx": group_key[2],
            "new_line_idx": group_key[3],
            "line_content": group_key[4],
            "label": binary,
            "agreement": top_count / len(labels),
            "n_annotators": len(labels),
            "line_idx": int(idx),
        })
    return results


# --- Eligible commits ---

def eligible_commits(df: pd.DataFrame) -> list[str]:
    """Filter to commits with min annotators and reasonable size."""
    df = df[df["error"].isna()].copy()
    df["_old"] = df["old_line_idx"].fillna(-1).astype(int)
    df["_new"] = df["new_line_idx"].fillna(-1).astype(int)
    line_key = ["commit_sha", "file", "change_type", "_old", "_new"]

    ann_per_line = df.groupby(line_key)["annotator"].nunique()
    min_ann = ann_per_line.groupby("commit_sha").min()
    enough_ann = set(min_ann[min_ann >= MIN_ANNOTATORS].index)

    lines_per_commit = df.drop_duplicates(line_key).groupby("commit_sha").size()
    right_size = set(lines_per_commit[
        (lines_per_commit >= MIN_LINES) & (lines_per_commit <= MAX_LINES)
    ].index)

    return sorted(enough_ann & right_size)


# --- Splits ---

def generate_splits(shas: list[str]) -> list[tuple[list[str], list[str]]]:
    """Generate k folds: each independently samples test_ratio for test."""
    rng = random.Random(SEED)
    test_size = int(len(shas) * TEST_RATIO)
    splits = []
    for _ in range(N_FOLDS):
        test = rng.sample(shas, test_size)
        test_set = set(test)
        train = [s for s in shas if s not in test_set]
        splits.append((train, test))
    return splits


# --- Metrics ---

def mcc(tp, fp, fn, tn) -> float:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom != 0 else 0.0


def confusion(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn


def micro_metrics(rows: list[dict]) -> dict:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    tn = sum(r["tn"] for r in rows)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "mcc": mcc(tp, fp, fn, tn)}


# --- Evaluation ---

def evaluate_commit(gt_lines: list[dict], preds: list[dict]) -> dict:
    """Compare predictions to ground truth, return confusion counts."""
    gt_df = pd.DataFrame([
        {"file": l["file"], "status": l["change_type"], "line_idx": l["line_idx"],
         "true": 1 if l["label"] == "bug" else 0}
        for l in gt_lines
    ])
    pred_df = pd.DataFrame(preds)
    joined = pred_df.merge(gt_df, on=["file", "status", "line_idx"], how="inner")
    tp, fp, fn, tn = confusion(joined["true"].values, joined["pred"].values)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_lines": len(joined)}


def print_report(name: str, fold_rows: list[list[dict]]) -> None:
    """Print micro-averaged metrics with per-fold breakdown."""
    n_commits = sum(len(rows) for rows in fold_rows)
    n_lines = sum(r["n_lines"] for rows in fold_rows for r in rows)
    fold_metrics = [micro_metrics(rows) for rows in fold_rows]

    print(f"{name} ({n_commits} commits, {n_lines} lines, {len(fold_rows)} folds)")
    for col in ["precision", "recall", "f1", "mcc"]:
        vals = [m[col] for m in fold_metrics]
        per_fold = "  ".join(f"{v:.3f}" for v in vals)
        print(f"  {col:10s}: {per_fold}  mean={np.mean(vals):.3f} std={np.std(vals):.3f}")
    print()


# --- Main ---

def main():
    print("Loading herbold.parquet...")
    raw = pd.read_parquet(DATA_PATH)

    print("Computing eligible commits...")
    shas = eligible_commits(raw)
    print(f"  {len(shas)} eligible commits")

    print("Computing ground truth consensus...")
    by_commit = {sha: group for sha, group in raw.groupby("commit_sha")}
    gt_cache = {}
    for sha in shas:
        gt_cache[sha] = consensus(by_commit[sha])

    splits = generate_splits(shas)
    print(f"  {N_FOLDS} folds, {TEST_RATIO:.0%} test\n")

    classifiers = ["all_bug", "heuristic", "ml_logreg", "ml_rf"]
    results = {c: [[] for _ in range(N_FOLDS)] for c in classifiers}

    for fold, (train_shas, test_shas) in enumerate(splits):
        print(f"--- Fold {fold} ({len(train_shas)} train, {len(test_shas)} test) ---")

        # Train or load ML models
        ml_models = {}
        for est_name in ["logreg", "rf"]:
            clf_name = f"ml_{est_name}"
            model_path = MODELS_DIR / clf_name / f"fold_{fold}.pkl"
            if model_path.exists():
                print(f"  Loading {clf_name} fold {fold}")
                model = load_model(clf_name, fold)
            else:
                train_lines = [gt_cache[s] for s in train_shas]
                model = ml_train(train_lines, est_name)
                save_model(model, clf_name, fold)
            ml_models[est_name] = model

        # Evaluate on test commits
        for sha in test_shas:
            gt_lines = gt_cache[sha]

            # all_bug: predict everything as bug
            all_bug_preds = [
                {"file": l["file"], "status": l["change_type"],
                 "line_idx": l["line_idx"], "pred": 1}
                for l in gt_lines
            ]
            results["all_bug"][fold].append(evaluate_commit(gt_lines, all_bug_preds))

            # heuristic
            h_preds = heuristic_predict(gt_lines)
            results["heuristic"][fold].append(evaluate_commit(gt_lines, h_preds))

            # ML classifiers
            for est_name, model in ml_models.items():
                clf_name = f"ml_{est_name}"
                m_preds = ml_predict(gt_lines, model)
                results[clf_name][fold].append(evaluate_commit(gt_lines, m_preds))

        print()

    # Report
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    for clf in classifiers:
        print_report(clf, results[clf])


if __name__ == "__main__":
    main()
