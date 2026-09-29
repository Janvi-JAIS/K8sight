#!/usr/bin/env python3
"""K8sight workload-pattern classifier.

Proof-of-concept classifier for five workload classes:
stable, increasing, periodic, spike, random_spike.

Method follows the Claude design used in the synthetic end-to-end test:
- classification unit = 120-second window (8 samples at 15 s), stride 1
- pattern is label only; timestamp/pod/absolute levels are excluded
- windows never cross a 2-block boundary
- blocked cross-validation across contiguous time blocks
- compare logistic regression and random forest, plus a stratified dummy reference
- label-shuffle control for leakage sanity checking
- save the exact feature code used for evaluation

Run from the project directory:
    python train_workload_classifier.py
or:
    python train_workload_classifier.py --project-dir ~/k8sight-locust
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    import joblib
    import sklearn
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}. Run: pip install pandas scikit-learn joblib matplotlib"
    )

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

CLASS_ORDER = ["stable", "increasing", "periodic", "spike", "random_spike"]
SCHEMA = [
    "timestamp", "pod", "cpu_cores", "cpu_millicores",
    "memory_bytes", "memory_mib", "pattern",
]
STEP_SECONDS = 15
WINDOW = 8
STRIDE = 1
N_FOLDS = 2
SEED = 42
N_SHUFFLE_REPEATS = 20
TIE_MARGIN = 0.02
CANDIDATES = ["logistic_regression", "random_forest"]

FEATURE_INFO = {
    "cpu_cv": ("CPU variability", "Std/mean CPU."),
    "cpu_rel_range": ("CPU variability", "(max-min)/mean CPU."),
    "cpu_rel_slope": ("CPU trend", "Window slope scaled by window length and mean CPU."),
    "cpu_trend_r2": ("CPU trend", "R-squared of the linear trend."),
    "cpu_half_logratio": ("CPU trend", "Log ratio between second-half and first-half CPU means."),
    "cpu_peak_to_mean": ("CPU burstiness", "Peak CPU relative to mean CPU."),
    "cpu_skew": ("CPU shape", "Skewness of CPU values."),
    "cpu_lag1_autocorr": ("CPU smoothness", "Lag-1 autocorrelation."),
    "cpu_diff_std_rel": ("CPU smoothness", "Std of 15 s CPU changes relative to mean."),
    "cpu_max_abs_diff_rel": ("CPU burstiness", "Largest 15 s CPU change relative to mean."),
    "cpu_zero_crossings": ("CPU oscillation", "Crossings of the window mean."),
    "cpu_lowfreq_power_frac": ("CPU spectrum", "Low-frequency spectral power fraction."),
    "cpu_hf_power_frac": ("CPU spectrum", "Higher-frequency spectral power fraction."),
    "mem_rel_range": ("Memory", "(max-min)/mean memory."),
    "mem_rel_slope": ("Memory", "Relative memory slope."),
    "mem_cpu_corr": ("Memory", "Correlation of memory and CPU inside the window."),
}
FEATURE_NAMES = list(FEATURE_INFO)

FEATURE_MODULE_SOURCE = r'''"""Feature extraction used by the K8sight workload classifier."""
import numpy as np
from collections import Counter

STEP_SECONDS = 15
WINDOW = 8
STRIDE = 1
FEATURE_NAMES = [
    "cpu_cv", "cpu_rel_range", "cpu_rel_slope", "cpu_trend_r2",
    "cpu_half_logratio", "cpu_peak_to_mean", "cpu_skew", "cpu_lag1_autocorr",
    "cpu_diff_std_rel", "cpu_max_abs_diff_rel", "cpu_zero_crossings",
    "cpu_lowfreq_power_frac", "cpu_hf_power_frac", "mem_rel_range",
    "mem_rel_slope", "mem_cpu_corr",
]

def _safe_corr(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return c if np.isfinite(c) else 0.0

def extract_window_features(cpu_mc, mem_mib):
    cpu = np.asarray(cpu_mc, dtype=float)
    mem = np.asarray(mem_mib, dtype=float)
    n = len(cpu)
    if n != WINDOW or len(mem) != WINDOW:
        raise ValueError(f"Expected {WINDOW} samples per window")
    x = np.arange(n, dtype=float)
    mean = float(cpu.mean())
    den = max(abs(mean), 1.0)
    sd = float(cpu.std())
    dev = cpu - mean
    slope, intercept = np.polyfit(x, cpu, 1)
    fit = slope * x + intercept
    ss_tot = float((dev ** 2).sum())
    r2 = 1.0 - float(((cpu - fit) ** 2).sum()) / ss_tot if ss_tot > 1e-12 else 0.0
    half = n // 2
    first = float(cpu[:half].mean())
    second = float(cpu[half:].mean())
    power = np.abs(np.fft.rfft(dev)) ** 2
    total = float(power[1:].sum()) + 1e-12
    low = float(power[1:3].sum()) / total
    high = float(power[3:].sum()) / total
    d = np.diff(cpu)
    signs = np.sign(dev)
    signs = signs[signs != 0]
    crossings = int((np.diff(signs) != 0).sum()) if len(signs) > 1 else 0
    skew = float((dev ** 3).mean() / sd ** 3) if sd > 1e-9 else 0.0
    mden = max(abs(float(mem.mean())), 1.0)
    mslope = float(np.polyfit(x, mem, 1)[0])
    return {
        "cpu_cv": sd / den,
        "cpu_rel_range": float(cpu.max() - cpu.min()) / den,
        "cpu_rel_slope": float(slope) * (n - 1) / den,
        "cpu_trend_r2": float(r2),
        "cpu_half_logratio": float(np.log(second + 1.0) - np.log(first + 1.0)),
        "cpu_peak_to_mean": float(cpu.max()) / den,
        "cpu_skew": skew,
        "cpu_lag1_autocorr": _safe_corr(cpu[:-1], cpu[1:]),
        "cpu_diff_std_rel": float(d.std()) / den,
        "cpu_max_abs_diff_rel": float(np.abs(d).max()) / den,
        "cpu_zero_crossings": crossings,
        "cpu_lowfreq_power_frac": low,
        "cpu_hf_power_frac": high,
        "mem_rel_range": float(mem.max() - mem.min()) / mden,
        "mem_rel_slope": mslope * (n - 1) / mden,
        "mem_cpu_corr": _safe_corr(cpu, mem),
    }

def classify_trace(bundle, cpu_millicores, memory_mib):
    cpu = np.asarray(cpu_millicores, dtype=float)
    mem = np.asarray(memory_mib, dtype=float)
    starts = list(range(0, len(cpu) - WINDOW + 1, STRIDE))
    if not starts:
        raise ValueError(f"Need at least {WINDOW} samples")
    names = bundle["config"]["feature_names"]
    X = np.array([
        [extract_window_features(cpu[s:s+WINDOW], mem[s:s+WINDOW])[f] for f in names]
        for s in starts
    ])
    preds = bundle["model"].predict(X)
    counts = Counter(preds)
    label, votes = counts.most_common(1)[0]
    return {
        "window_predictions": [str(p) for p in preds],
        "majority_label": str(label),
        "majority_share": float(votes / len(preds)),
    }
'''


def fail(msg: str) -> None:
    raise SystemExit(f"\n[FAIL] {msg}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_timestamps(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float)
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    if dt.isna().any():
        fail("Some timestamps could not be parsed.")
    return (dt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()


def load_and_validate(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Input not found: {path}")
    df = pd.read_csv(path)
    if list(df.columns) != SCHEMA:
        fail(f"Columns {list(df.columns)} != expected {SCHEMA}")
    problems = []
    if df.isna().any().any():
        problems.append("missing values present")
    if set(df["pattern"].unique()) != set(CLASS_ORDER):
        problems.append(f"labels {sorted(df['pattern'].unique())} != {sorted(CLASS_ORDER)}")
    if (df["cpu_millicores"] < 0).any():
        problems.append("negative CPU values")
    if (df["memory_bytes"] <= 0).any():
        problems.append("non-positive memory values")
    if not np.allclose(df["cpu_cores"] * 1000, df["cpu_millicores"], rtol=1e-6, atol=1e-6):
        problems.append("cpu_millicores != cpu_cores*1000")
    if not np.allclose(df["memory_bytes"] / 1048576, df["memory_mib"], rtol=1e-6, atol=1e-6):
        problems.append("memory_mib != memory_bytes/1048576")
    df["ts"] = parse_timestamps(df["timestamp"])
    if df.duplicated(["ts", "pod", "pattern"]).any():
        problems.append("duplicate timestamp/pod/pattern rows")
    if problems:
        fail("Data validation failed: " + "; ".join(problems))
    return df


def smoothness_table(df: pd.DataFrame) -> dict:
    rows = {}
    for pattern in CLASS_ORDER:
        cpu = df.loc[df["pattern"] == pattern].sort_values("ts")["cpu_millicores"].to_numpy()
        d = np.diff(cpu)
        rows[pattern] = {
            "rows": int(len(cpu)),
            "cpu_mean_mc": float(cpu.mean()),
            "cpu_max_mc": float(cpu.max()),
            "step_std_over_level_std": float(d.std() / (cpu.std() + 1e-9)),
        }
    return rows


def build_runs(df: pd.DataFrame):
    runs = {}
    notes = []
    for (pattern, pod), group in df.groupby(["pattern", "pod"]):
        g = group.sort_values("ts").reset_index(drop=True)
        dts = np.diff(g["ts"].to_numpy())
        breaks = np.where(np.abs(dts - STEP_SECONDS) > 1.5)[0] + 1
        segments = np.split(np.arange(len(g)), breaks)
        if len(segments) > 1:
            notes.append(f"{pattern}/{pod}: {len(segments)-1} timestamp gap(s); windows never span a gap")
        runs[f"{pattern}|{pod}"] = (pattern, g, segments)
    return runs, notes


def build_windows(runs: dict, feature_module):
    records = []
    for run_id, (pattern, g, segments) in runs.items():
        n = len(g)
        blocks = np.array_split(np.arange(n), N_FOLDS)
        block_id = np.empty(n, dtype=int)
        for b, ids in enumerate(blocks):
            block_id[ids] = b
        cpu = g["cpu_millicores"].to_numpy(float)
        mem = g["memory_mib"].to_numpy(float)
        for segment in segments:
            for start in range(0, len(segment) - WINDOW + 1, STRIDE):
                idx = segment[start:start + WINDOW]
                # Never create a window crossing the train/test block boundary.
                if block_id[idx[0]] != block_id[idx[-1]]:
                    continue
                features = feature_module.extract_window_features(cpu[idx], mem[idx])
                rec = {
                    "run": run_id,
                    "label": pattern,
                    "block": int(block_id[idx[0]]),
                    "row_start": int(idx[0]),
                    "row_end": int(idx[-1]) + 1,
                }
                rec.update(features)
                records.append(rec)
    win = pd.DataFrame(records)
    if win.empty:
        fail("No windows could be constructed.")
    counts = win.groupby(["run", "block"]).size()
    for run_id in runs:
        for b in range(N_FOLDS):
            if counts.get((run_id, b), 0) < 1:
                fail(f"{run_id} has no window in block {b}; reduce WINDOW/N_FOLDS or collect longer traces.")
    if not np.isfinite(win[FEATURE_NAMES].to_numpy()).all():
        fail("Non-finite feature values found.")
    return win


def make_model(kind: str):
    if kind == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED),
        )
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=1,
        )
    if kind == "dummy_stratified":
        return DummyClassifier(strategy="stratified", random_state=SEED)
    raise ValueError(kind)


def blocked_cv(win: pd.DataFrame, kind: str, shuffle_seed: int | None = None) -> dict:
    rng = np.random.RandomState(shuffle_seed) if shuffle_seed is not None else None
    y_true, y_pred, folds = [], [], []
    for fold in range(N_FOLDS):
        train = win[win["block"] != fold]
        test = win[win["block"] == fold]
        if set(train["label"]) != set(CLASS_ORDER) or set(test["label"]) != set(CLASS_ORDER):
            fail(f"Fold {fold}: every class must occur in both train and test blocks.")
        y_train = train["label"].to_numpy()
        if rng is not None:
            y_train = rng.permutation(y_train)
        model = make_model(kind)
        model.fit(train[FEATURE_NAMES].to_numpy(), y_train)
        pred = model.predict(test[FEATURE_NAMES].to_numpy())
        y_true.extend(test["label"].tolist())
        y_pred.extend(pred.tolist())
        folds.append({
            "fold": fold,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "macro_f1": float(
                f1_score(test["label"], pred, labels=CLASS_ORDER, average="macro", zero_division=0)
            ),
        })
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "folds": folds,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
    }


def strip_predictions(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in {"y_true", "y_pred"}}


def write_readme(path: Path, context: dict) -> None:
    selected = context["selected"]
    results = context["results"]
    dummy = context["dummy"]
    shuffle = context["shuffle"]
    lines = [
        "# K8sight workload-pattern classifier (proof of concept)",
        "",
        "This is a proof-of-concept classifier for five synthetic workload shapes from CPU and memory time series.",
        "It does not establish production or cross-run generalisation.",
        "",
        "## Method",
        f"- Input: `combined_all_patterns.csv`, {context['n_rows']} source rows.",
        f"- Classes: {', '.join(CLASS_ORDER)}.",
        f"- Window: {WINDOW} samples = {WINDOW * STEP_SECONDS} seconds; stride={STRIDE}.",
        f"- Validation: {N_FOLDS} contiguous blocks per run; windows never cross block boundaries.",
        "- Features use CPU millicores and memory MiB only; pattern is the label and timestamp/pod are excluded.",
        "- Candidate models: logistic regression and random forest; stratified dummy is a reference.",
        f"- Selection: higher pooled macro-F1; differences below {TIE_MARGIN} favour logistic regression.",
        f"- Selected model: **{selected}**.",
        "- Selection and reporting use the same blocked validation, so the selected score is slightly optimistic.",
        "",
        "## Important limitation",
        "There is one workload run per class. Therefore the validation measures recognition of held-out segments from the same runs, not generalisation to new runs, services, or production traffic.",
        "",
        "## Results",
        "",
        "| Model | Accuracy | Balanced accuracy | Macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for name in sorted(results):
        r = results[name]
        lines.append(f"| {name} | {r['accuracy']:.3f} | {r['balanced_accuracy']:.3f} | {r['macro_f1']:.3f} |")
    lines.append(
        f"| dummy_stratified (reference) | {dummy['accuracy']:.3f} | {dummy['balanced_accuracy']:.3f} | {dummy['macro_f1']:.3f} |"
    )
    lines += [
        "",
        f"Label-shuffle control: mean macro-F1={shuffle['mean']:.3f}, std={shuffle['std']:.3f}, max={shuffle['max']:.3f} over {N_SHUFFLE_REPEATS} repeats.",
        "",
        "## Features",
        "",
        "| Feature | Group | Why it is useful |",
        "|---|---|---|",
    ]
    for name in FEATURE_NAMES:
        group, why = FEATURE_INFO[name]
        lines.append(f"| `{name}` | {group} | {why} |")
    lines += [
        "",
        "## Files",
        "- `workload_classifier.joblib`: saved model + configuration.",
        "- `workload_features.py`: exact feature-processing code used.",
        "- `window_features.csv`: one row per window.",
        "- `feature_class_means.csv`: mean features per class.",
        "- `feature_weights.csv`: descriptive model feature weights/importances.",
        "- `feature_config.json`: feature/window configuration.",
        "- `classification_report.txt`: per-class precision/recall/F1.",
        "- `confusion_matrix.csv`: confusion matrix counts.",
        "- `confusion_matrix.png`: confusion matrix image when matplotlib is installed.",
        "- `metrics.json`: metrics, controls, and limitations.",
        "- `run_manifest.json`: input hash and package versions.",
        "",
        "## Reproducibility",
        f"Seed: {SEED}. Python {platform.python_version()}, scikit-learn {sklearn.__version__}, pandas {pd.__version__}, numpy {np.__version__}.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=str(Path.home() / "k8sight-locust"))
    args = parser.parse_args()
    project = Path(args.project_dir).expanduser().resolve()
    input_csv = project / "combined_all_patterns.csv"
    out_dir = project / "classifier_output"
    if not project.is_dir():
        fail(f"Project directory not found: {project}")
    out_dir.mkdir(exist_ok=True)

    csv_before = {p.name: sha256(p) for p in project.glob("*.csv")}

    print("\n=== 1/9 Loading and validating data")
    df = load_and_validate(input_csv)
    rows_per_class = df["pattern"].value_counts().reindex(CLASS_ORDER).to_dict()
    print(f"    {len(df)} rows: " + ", ".join(f"{k}={rows_per_class[k]}" for k in CLASS_ORDER))

    print("\n=== 2/9 Consistency check across patterns")
    consistency = smoothness_table(df)
    for pattern in CLASS_ORDER:
        r = consistency[pattern]
        print(
            f"    {pattern:<14} rows={r['rows']:<4} mean={r['cpu_mean_mc']:.1f} mc "
            f"max={r['cpu_max_mc']:.1f} mc step/level={r['step_std_over_level_std']:.2f}"
        )

    print("\n=== 3/9 Writing feature code, building windows and features")
    feature_path = out_dir / "workload_features.py"
    feature_path.write_text(FEATURE_MODULE_SOURCE)
    spec = importlib.util.spec_from_file_location("workload_features", feature_path)
    wf = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(wf)
    runs, notes = build_runs(df)
    for note in notes:
        print(f"    note: {note}")
    win = build_windows(runs, wf)
    win.to_csv(out_dir / "window_features.csv", index=False)
    windows_per_class = win["label"].value_counts().reindex(CLASS_ORDER).to_dict()
    print(f"    window={WINDOW} samples (120s), stride={STRIDE}, features={len(FEATURE_NAMES)}, blocks={N_FOLDS}")
    print("    windows per class: " + ", ".join(f"{k}={windows_per_class[k]}" for k in CLASS_ORDER))
    win.groupby("label")[FEATURE_NAMES].mean().reindex(CLASS_ORDER).to_csv(out_dir / "feature_class_means.csv")

    print("\n=== 4/9 Blocked cross-validation of candidate models")
    results = {name: blocked_cv(win, name) for name in CANDIDATES}
    dummy = blocked_cv(win, "dummy_stratified")
    for name in sorted(results):
        r = results[name]
        print(f"    {name:<22} accuracy={r['accuracy']:.3f} bal_acc={r['balanced_accuracy']:.3f} macro_F1={r['macro_f1']:.3f}")
    print(f"    {'dummy_stratified (ref)':<22} accuracy={dummy['accuracy']:.3f} bal_acc={dummy['balanced_accuracy']:.3f} macro_F1={dummy['macro_f1']:.3f}")

    lr, rf = results["logistic_regression"], results["random_forest"]
    if abs(rf["macro_f1"] - lr["macro_f1"]) < TIE_MARGIN:
        selected = "logistic_regression"
    else:
        selected = "random_forest" if rf["macro_f1"] > lr["macro_f1"] else "logistic_regression"
    print(f"    selected={selected}")

    print("\n=== 5/9 Evaluation of selected model")
    selected_result = results[selected]
    report_txt = classification_report(
        selected_result["y_true"], selected_result["y_pred"],
        labels=CLASS_ORDER, zero_division=0, digits=3,
    )
    print(report_txt)
    cm = confusion_matrix(selected_result["y_true"], selected_result["y_pred"], labels=CLASS_ORDER)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_ORDER], columns=[f"pred_{c}" for c in CLASS_ORDER])
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    print("\n=== 6/9 Label-shuffle control")
    shuffled_scores = [
        blocked_cv(win, selected, shuffle_seed=SEED + i)["macro_f1"]
        for i in range(N_SHUFFLE_REPEATS)
    ]
    shuffle = {
        "mean": float(np.mean(shuffled_scores)),
        "std": float(np.std(shuffled_scores)),
        "max": float(np.max(shuffled_scores)),
        "repeats": N_SHUFFLE_REPEATS,
    }
    print(
        f"    shuffled macro-F1 mean={shuffle['mean']:.3f} "
        f"std={shuffle['std']:.3f} max={shuffle['max']:.3f}"
    )

    print("\n=== 7/9 Training final model and saving artefacts")
    final_model = make_model(selected)
    final_model.fit(win[FEATURE_NAMES].to_numpy(), win["label"].to_numpy())
    if selected == "random_forest":
        weights = final_model.feature_importances_
        weight_type = "random-forest impurity importance"
    else:
        weights = np.abs(final_model.named_steps["logisticregression"].coef_).mean(axis=0)
        weight_type = "mean absolute standardised coefficient"
    pd.DataFrame({
        "feature": FEATURE_NAMES,
        "weight": weights,
        "weight_type": weight_type,
        "group": [FEATURE_INFO[f][0] for f in FEATURE_NAMES],
        "why_useful": [FEATURE_INFO[f][1] for f in FEATURE_NAMES],
    }).sort_values("weight", ascending=False).to_csv(out_dir / "feature_weights.csv", index=False)

    config = {
        "seed": SEED,
        "window_samples": WINDOW,
        "stride": STRIDE,
        "step_seconds": STEP_SECONDS,
        "n_folds": N_FOLDS,
        "selected_model": selected,
        "candidates": CANDIDATES,
        "selection_rule": f"higher pooled macro-F1; difference < {TIE_MARGIN} -> simpler logistic regression",
        "class_order": CLASS_ORDER,
        "feature_names": FEATURE_NAMES,
        "input_columns_used": ["cpu_millicores", "memory_mib"],
        "excluded_from_features": ["pattern (label only)", "timestamp", "pod", "absolute CPU/memory levels"],
    }
    joblib.dump({"model": final_model, "config": config}, out_dir / "workload_classifier.joblib")
    (out_dir / "feature_config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "classification_report.txt").write_text(
        f"Selected model: {selected}\nBlocked validation, pooled over {N_FOLDS} folds.\n"
        "One run per class: within-run generalisation only.\n\n" + report_txt
    )

    if HAVE_MPL:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(CLASS_ORDER)))
        ax.set_yticks(range(len(CLASS_ORDER)))
        ax.set_xticklabels(CLASS_ORDER, rotation=35, ha="right")
        ax.set_yticklabels(CLASS_ORDER)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Out-of-fold confusion matrix ({selected})")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)

    metrics = {
        "validation": {
            "scheme": f"blocked {N_FOLDS}-fold contiguous row blocks; windows never span blocks",
            "pooled_over_folds": True,
        },
        "models": {k: strip_predictions(v) for k, v in results.items()},
        "selection": {
            "selected": selected,
            "note": "Selection and reporting use the same validation; score is slightly optimistic.",
        },
        "selected_model_per_class": classification_report(
            selected_result["y_true"], selected_result["y_pred"], labels=CLASS_ORDER,
            zero_division=0, output_dict=True,
        ),
        "references": {
            "dummy_stratified": strip_predictions(dummy),
            "label_shuffle_control": shuffle,
        },
        "windows_per_class": windows_per_class,
        "rows_per_class": rows_per_class,
        "consistency_table": consistency,
        "limitations": [
            "One run per class: within-run generalisation only.",
            "Small dataset (169 source rows in the current combined file).",
            "Synthetic patterns against one sample service.",
            "Selected-model scores are slightly optimistic because selection and reporting share validation.",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    input_sha = sha256(input_csv)
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "input_csv": str(input_csv),
        "input_sha256": input_sha,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "seed": SEED,
    }, indent=2))

    print("\n=== 8/9 Writing README")
    write_readme(out_dir / "README.md", {
        "selected": selected,
        "results": results,
        "dummy": dummy,
        "shuffle": shuffle,
        "n_rows": len(df),
    })

    print("\n=== 9/9 Verifying outputs")
    required = [
        "workload_classifier.joblib", "workload_features.py", "window_features.csv",
        "feature_class_means.csv", "feature_weights.csv", "feature_config.json",
        "classification_report.txt", "confusion_matrix.csv", "metrics.json",
        "run_manifest.json", "README.md",
    ]
    if HAVE_MPL:
        required.append("confusion_matrix.png")
    ok = True
    for name in required:
        p = out_dir / name
        good = p.exists() and p.stat().st_size > 0
        ok &= good
        print(f"    {'OK' if good else 'BAD'} {name}")

    bundle = joblib.load(out_dir / "workload_classifier.joblib")
    reloaded_pred = bundle["model"].predict(win[FEATURE_NAMES].to_numpy())
    same = bool(np.array_equal(reloaded_pred, final_model.predict(win[FEATURE_NAMES].to_numpy())))
    no_label = "pattern" not in bundle["config"]["feature_names"]
    first_run = next(iter(runs.values()))[1]
    demo = wf.classify_trace(bundle, first_run["cpu_millicores"].to_numpy(), first_run["memory_mib"].to_numpy())
    demo_ok = demo["majority_label"] in CLASS_ORDER
    csv_after = {p.name: sha256(p) for p in project.glob("*.csv")}
    unchanged = csv_before == csv_after

    print(f"    reloaded model reproduces predictions : {'OK' if same else 'BAD'}")
    print(f"    'pattern' absent from feature list    : {'OK' if no_label else 'BAD'}")
    print(f"    saved feature code runs end-to-end   : {'OK' if demo_ok else 'BAD'}")
    print(f"    project CSV files unchanged (SHA-256): {'OK' if unchanged else 'BAD'}")

    ok &= same and no_label and demo_ok and unchanged
    print("\n" + "=" * 64)
    print(f"Selected model: {selected}")
    print(
        f"Blocked validation: accuracy {selected_result['accuracy']:.3f}, "
        f"balanced accuracy {selected_result['balanced_accuracy']:.3f}, "
        f"macro-F1 {selected_result['macro_f1']:.3f}"
    )
    print(
        f"Label shuffle control: mean macro-F1 {shuffle['mean']:.3f}, "
        f"max {shuffle['max']:.3f}"
    )
    print("Scope: one run per class -> held-out segments of the same runs only.")
    print(f"Outputs: {out_dir}")
    print("RESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
