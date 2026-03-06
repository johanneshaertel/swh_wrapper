"""Trivial line-level bug classifiers: heuristic rules + ML (logreg, rf).

Self-contained: all feature extraction, training, and prediction in one file.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MODELS_DIR = Path("data/models")

# --- Shared patterns ---

TEST_PATH_PATTERNS = ["/test/", "/tests/", "/test_", "_test.", "Test.java", "Tests.java"]
DOC_PATH_PATTERNS = ["CHANGELOG", "CHANGES", "README", "LICENSE"]
DOC_EXTENSIONS = {".md", ".txt", ".rst", ".html", ".xml", ".json", ".yaml", ".yml", ".cfg", ".ini", ".properties"}
COMMENT_PREFIXES = ("//", "#", "/*", "*", "<!--", "%", "--")
IMPORT_PREFIXES = ("import ", "from ", "#include", "using ", "require(", "require ")


def _is_test_file(path: str) -> bool:
    return any(p in path for p in TEST_PATH_PATTERNS)


def _is_doc_file(path: str) -> bool:
    if any(p in path for p in DOC_PATH_PATTERNS):
        return True
    return any(path.lower().endswith(ext) for ext in DOC_EXTENSIONS)


# --- Heuristic classifier ---

def heuristic_predict(lines: list[dict]) -> list[dict]:
    """Predict using heuristic rules: bug unless test/doc/whitespace/comment/import."""
    results = []
    for l in lines:
        content = l.get("line_content", "") or ""
        stripped = content.strip()
        is_bug = not (
            _is_test_file(l["file"])
            or _is_doc_file(l["file"])
            or stripped == ""
            or stripped.startswith(("//", "#", "/*", "*", "import ", "from "))
        )
        results.append({
            "file": l["file"],
            "status": l["change_type"],
            "line_idx": l["line_idx"],
            "pred": 1 if is_bug else 0,
        })
    return results


# --- ML features ---

FEATURE_COLS = [
    "is_add", "is_del", "line_len", "stripped_len", "is_blank",
    "is_comment", "is_import", "is_test_file", "is_doc_file",
    "is_java", "is_py", "is_xml", "is_config",
]


def extract_features(file: str, change_type: str, line_content: str) -> dict:
    content = line_content or ""
    stripped = content.strip()
    ext = os.path.splitext(file)[1].lower()
    return {
        "is_add": int(change_type == "+"),
        "is_del": int(change_type == "-"),
        "line_len": len(content),
        "stripped_len": len(stripped),
        "is_blank": int(stripped == ""),
        "is_comment": int(stripped.startswith(COMMENT_PREFIXES)),
        "is_import": int(stripped.startswith(IMPORT_PREFIXES)),
        "is_test_file": int(any(p in file for p in TEST_PATH_PATTERNS)),
        "is_doc_file": int(any(p in file for p in DOC_PATH_PATTERNS) or ext in DOC_EXTENSIONS),
        "is_java": int(ext == ".java"),
        "is_py": int(ext == ".py"),
        "is_xml": int(ext == ".xml"),
        "is_config": int(ext in {".cfg", ".ini", ".properties", ".yaml", ".yml", ".toml"}),
    }


def features_for_lines(lines: list[dict]) -> pd.DataFrame:
    """Extract features + labels for ground truth lines."""
    rows = []
    for l in lines:
        feats = extract_features(l["file"], l["change_type"], l["line_content"])
        feats["file"] = l["file"]
        feats["status"] = l["change_type"]
        feats["line_idx"] = l["line_idx"]
        feats["label"] = 1 if l["label"] == "bug" else 0
        rows.append(feats)
    return pd.DataFrame(rows)


# --- ML training and prediction ---

ESTIMATORS = {
    "logreg": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]),
    "rf": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
}


def ml_train(train_lines_per_commit: list[list[dict]], estimator_name: str = "logreg") -> object:
    """Train ML classifier on ground truth lines from multiple commits."""
    frames = [features_for_lines(lines) for lines in train_lines_per_commit if lines]
    df = pd.concat(frames, ignore_index=True)
    X = df[FEATURE_COLS].values
    y = df["label"].values
    print(f"  Training {estimator_name} on {len(df)} lines "
          f"({y.sum()} bug, {len(y) - y.sum()} unrelated)")
    model = ESTIMATORS[estimator_name]()
    model.fit(X, y)
    return model


def ml_predict(lines: list[dict], model: object) -> list[dict]:
    """Predict line labels using a trained ML model."""
    df = features_for_lines(lines)
    X = df[FEATURE_COLS].values
    preds = model.predict(X)
    return [
        {"file": row["file"], "status": row["status"],
         "line_idx": int(row["line_idx"]), "pred": int(p)}
        for (_, row), p in zip(df.iterrows(), preds)
    ]


def save_model(model: object, classifier_name: str, fold: int) -> None:
    path = MODELS_DIR / classifier_name / f"fold_{fold}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(classifier_name: str, fold: int) -> object:
    path = MODELS_DIR / classifier_name / f"fold_{fold}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
