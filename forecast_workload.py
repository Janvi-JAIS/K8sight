#!/usr/bin/env python3
"""
K8sight - forecast_workload.py

Proof-of-concept CPU forecaster, one model per workload pattern, built from
the corrected, ML-ready dataset (default: combined_all_patterns_ml.csv).
No Docker / Kubernetes / Prometheus needed.

Run:  python forecast_workload.py
      (optional) --project-dir /path/to/dir containing the input CSV
      (optional) --input-csv combined_all_patterns_ml.csv   (default; override if needed)

Design, in one paragraph
  Each pattern (stable/increasing/periodic/spike/random_spike) is one
  continuous Locust run against one pod: a single CPU time series, 15s
  samples. Forecasting is done PER PATTERN, never across patterns (mixing
  runs would blend unrelated regimes). Models are backtested with expanding-
  window walk-forward validation: train on every sample up to time t, predict
  t+1..t+HORIZON, slide forward one step, repeat. This never uses a future
  sample to predict the past. Horizon = 4 steps = 60 seconds: the longest
  runs are only ~10 minutes, so forecasting further ahead would extrapolate
  past the data actually collected. This is a real gap against a "15-20
  minute" forecast goal, not a limitation of the model choice.
"""

import argparse
import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    import statsmodels
    import joblib
except ImportError as e:
    sys.exit(f"Missing dependency ({e}). Run: pip install pandas statsmodels matplotlib joblib")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

warnings.filterwarnings("ignore")  # statsmodels convergence chatter on tiny windows

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
CLASS_ORDER = ["stable", "increasing", "periodic", "spike", "random_spike"]
SCHEMA = ["timestamp", "pod", "cpu_cores", "cpu_millicores",
          "memory_bytes", "memory_mib", "pattern"]
STEP_SECONDS = 15
STEP_TOLERANCE = 1.5
HORIZON = 4                 # 4 * 15s = 60s ahead
MIN_TRAIN = 12              # need this many points before the first backtest origin
TIE_MARGIN = 0.05           # if two models' MAE differ by < 5%, keep the simpler one
SIMPLICITY_ORDER = ["naive", "drift", "holt", "seasonal_naive", "arima", "holt_winters"]
SEED = 42


# ----------------------------------------------------------------------
# forecasting models - imported from the shared module so joblib can
# unpickle them later from a different script (e.g. right_sizing_engine.py)
# ----------------------------------------------------------------------
from forecasting_models import (NaiveModel, DriftModel, SeasonalNaiveModel, HoltModel,
                                HoltWintersModel, ArimaModel, model_registry_for, SEASONAL_PERIOD)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def fail(msg):
    print(f"\n[FAIL] {msg}")
    sys.exit(1)


def stage(msg):
    print(f"\n=== {msg}")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def parse_timestamps(series):
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().all():
        return num.astype(float)
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    if dt.isna().any():
        fail("Could not parse some timestamps.")
    return (dt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()


def load_and_validate(input_csv):
    if not input_csv.exists():
        fail(f"Input not found: {input_csv}")
    df = pd.read_csv(input_csv)
    if list(df.columns) != SCHEMA:
        fail(f"columns {list(df.columns)} != {SCHEMA}")
    problems = []
    if df.isna().any().any():
        problems.append("missing values present")
    if set(df["pattern"].unique()) != set(CLASS_ORDER):
        problems.append(f"labels {sorted(df['pattern'].unique())} != {sorted(CLASS_ORDER)}")
    if (df["cpu_millicores"] < 0).any():
        problems.append("negative CPU values")
    df["ts"] = parse_timestamps(df["timestamp"])
    if df.duplicated(["ts", "pod", "pattern"]).any():
        problems.append("duplicate (timestamp, pod, pattern) rows")
    if problems:
        fail("Data validation failed: " + "; ".join(problems))
    return df


def build_segments(df):
    """
    For each pattern: sort by time, then split into continuous, gap-free
    segments wherever the gap between consecutive timestamps deviates from
    STEP_SECONDS by more than STEP_TOLERANCE. Segments are returned in
    chronological order. A gap is never bridged - each segment is a series
    with a genuine, uniform 15s spacing throughout.
    """
    segments, notes = {}, []
    for pattern, g in df.groupby("pattern"):
        g = g.sort_values("ts").reset_index(drop=True)
        pods = g["pod"].unique()
        if len(pods) != 1:
            fail(f"{pattern}: expected one pod, found {list(pods)}")
        ts = g["ts"].to_numpy()
        cpu = g["cpu_millicores"].to_numpy(dtype=float)
        dts = np.diff(ts)
        break_after = np.where(np.abs(dts - STEP_SECONDS) > STEP_TOLERANCE)[0]
        bounds = [0] + [b + 1 for b in break_after] + [len(g)]
        segs = []
        for s, e in zip(bounds[:-1], bounds[1:]):
            segs.append({"y": cpu[s:e], "ts_start": float(ts[s]), "ts_end": float(ts[e - 1]),
                        "n": e - s})
        if break_after.size:
            gap_desc = ", ".join(f"{dts[b]:.0f}s" for b in break_after)
            notes.append(f"{pattern}: {break_after.size} timestamp gap(s) ({gap_desc}); "
                        f"split into {len(segs)} continuous segment(s), no window trained or "
                        f"forecast across a gap")
        segments[pattern] = segs
    return segments, notes


# ----------------------------------------------------------------------
# backtest
# ----------------------------------------------------------------------
def walk_forward_backtest(pattern, segs):
    """Runs the walk-forward backtest independently within each continuous
    segment (never across a gap), then pools the results for the pattern."""
    registry = model_registry_for(pattern)
    rows = []
    n_origins = 0
    usable_segs, skipped_segs = 0, []
    for si, seg in enumerate(segs):
        y = seg["y"]
        n = len(y)
        max_origin = n - HORIZON - 1
        if max_origin < MIN_TRAIN - 1:
            skipped_segs.append({"segment": si, "n_points": n})
            continue
        usable_segs += 1
        for origin in range(MIN_TRAIN - 1, max_origin + 1):
            train = y[:origin + 1]
            actual = y[origin + 1: origin + 1 + HORIZON]
            for mname, mcls in registry.items():
                model = mcls().fit(train)
                pred = np.asarray(model.predict(HORIZON), dtype=float)
                for h in range(HORIZON):
                    rows.append({"pattern": pattern, "segment": si, "origin": origin,
                                "horizon_step": h + 1, "model": mname,
                                "actual": float(actual[h]), "predicted": float(pred[h]),
                                "abs_error": float(abs(actual[h] - pred[h])),
                                "sq_error": float((actual[h] - pred[h]) ** 2)})
            n_origins += 1
    if n_origins == 0:
        fail(f"{pattern}: no segment has {MIN_TRAIN + HORIZON}+ points; cannot backtest. "
             f"Segment sizes: {[s['n'] for s in segs]}")
    return pd.DataFrame(rows), n_origins, usable_segs, skipped_segs


def select_model(bt_pattern):
    agg = bt_pattern.groupby("model").agg(mae=("abs_error", "mean"),
                                          rmse=("sq_error", lambda s: float(np.sqrt(s.mean())))).reset_index()
    best_mae = agg["mae"].min()
    within_margin = agg[agg["mae"] <= best_mae * (1 + TIE_MARGIN)]["model"].tolist()
    order = {m: i for i, m in enumerate(SIMPLICITY_ORDER)}
    selected = sorted(within_margin, key=lambda m: order.get(m, 99))[0]
    return selected, agg.set_index("model").to_dict("index")


# ----------------------------------------------------------------------
# README
# ----------------------------------------------------------------------
def write_readme(path, ctx):
    L = []
    a = L.append
    a("# K8sight - CPU forecaster (proof of concept)")
    a("")
    a("Produced by `forecast_workload.py`. One forecasting model per workload pattern, "
      f"predicting `cpu_millicores` {HORIZON} steps ({HORIZON * STEP_SECONDS}s) ahead. "
      "**Scope note:** the original K8sight target was a 15-20 minute forecast horizon; the longest "
      "collected run is under 10 minutes, so a 15-20 minute horizon would extrapolate past all available "
      "data. This stage forecasts 60 seconds ahead instead, and the gap should be listed as an open item, "
      "to be closed by collecting longer continuous runs.")
    a("")
    a("## Data")
    a(f"- Input: `{ctx['input_name']}` (SHA-256 `{ctx['input_sha'][:16]}...`).")
    a("- Target: `cpu_millicores`. Memory is not forecast in this stage.")
    a("- One continuous series per pattern (one pod, one Locust run each), split into gap-free segments "
      "where needed (see Gap handling below). Series are never mixed across patterns: forecasting is "
      "per-pattern, matching the assumption that the pattern is already known (from the classifier, or "
      "from which experiment is being run) at forecast time.")
    a("")
    a("## Gap handling")
    a("Some Prometheus collections have missing samples (observations legitimately unavailable, not just "
      "slow scrapes). A gap is detected wherever the time between two consecutive rows differs from the "
      f"15s step by more than {STEP_TOLERANCE}s. Each pattern's series is split at every such gap into "
      "continuous segments; a gap is never bridged, and is never treated as if it were a normal 15s step.")
    a("")
    if ctx.get("seg_notes"):
        for n in ctx["seg_notes"]:
            a(f"- {n}")
    else:
        a("- No timestamp gaps were detected in this input.")
    a("")
    a(f"- **Backtest:** training and forecasting happen independently within each segment; a segment "
      f"shorter than MIN_TRAIN+HORIZON={MIN_TRAIN + HORIZON} points is excluded from the backtest and "
      "does not contribute any origins.")
    a("- **Final saved model:** fit on each pattern's *latest* continuous segment only (the most recent "
      "usable history), not on the full row count. For a pattern with a gap, this means less history "
      "than its total row count suggests - see the per-pattern segment sizes in `metrics.json` "
      "(`gap_handling.per_pattern_segments`).")
    a("")
    a("## Method")
    a(f"- **Validation:** expanding-window walk-forward within each pattern's own series. The first backtest "
      f"origin needs {MIN_TRAIN} training points; from there, train on everything up to time t, predict "
      f"t+1..t+{HORIZON}, then slide forward one step and repeat. No test point is ever used to fit a model "
      f"that predicts it, and no pattern's data is used to forecast another pattern.")
    a("- **Models:** naive (persistence), drift (linear extrapolation), Holt (trend-only exponential "
      "smoothing), ARIMA (order search over a small fixed candidate list, falling back to naive if none "
      "converge). Periodic additionally gets seasonal-naive and Holt-Winters "
      f"(seasonal period {SEASONAL_PERIOD} samples = {SEASONAL_PERIOD * STEP_SECONDS}s, matching the load "
      "shape's own cycle).")
    a(f"- **Selection rule (fixed before running):** lowest mean absolute error (MAE) across all backtest "
      f"origins and horizon steps; if models are within {int(TIE_MARGIN*100)}% of the best, the simplest one "
      f"is kept ({' > '.join(SIMPLICITY_ORDER)}). Because the same backtest is used to select and to report, "
      "the reported error for the selected model is somewhat optimistic.")
    a("")
    a("## Results")
    a("")
    a("| Pattern | Segments (used/total) | Backtest origins | Selected model | MAE (millicores) | RMSE (millicores) | Naive MAE (reference) |")
    a("|---|---|---|---|---|---|---|")
    for p in CLASS_ORDER:
        r = ctx["per_pattern"][p]
        a(f"| {p} | {r['segments_used_in_backtest']}/{r['n_segments']} | {r['n_origins']} | {r['selected']} | "
          f"{r['mae']:.2f} | {r['rmse']:.2f} | {r['naive_mae']:.2f} |")
    a("")
    a("MAE/RMSE are averaged over all horizon steps 1-4; `metrics.json` also has the per-step breakdown, "
      "which typically grows with horizon step.")
    a("")
    a("## Limitations")
    a("- **Single run per pattern.** These numbers show how well a model fits held-out segments of the same "
      "run, not generalisation to a new run, a different service, or production traffic.")
    a(f"- **Short horizon.** {HORIZON * STEP_SECONDS}s, not the 15-20 minute target - see the scope note above.")
    a(f"- **Few backtest origins for Stable ({ctx['per_pattern']['stable']['n_origins']}) and Spike "
      f"({ctx['per_pattern']['spike']['n_origins']}).** Their error estimates are the least reliable.")
    a("- **Selected-model error is somewhat optimistic** (selection and reporting share the same backtest).")
    a("- **Per-pattern models assume the pattern is already known** at forecast time (e.g. from the "
      "classifier). A single model that forecasts without knowing the pattern first is a further step.")
    a("")
    a("## Files")
    for f, d in ctx["file_descriptions"].items():
        a(f"- `{f}` - {d}")
    a("")
    a("## Using a saved forecaster")
    a("```python")
    a("import joblib")
    a("bundle = joblib.load('forecasters.joblib')")
    a("entry = bundle['periodic']   # any pattern name")
    a("model = entry['model']       # already fit on the full series for that pattern")
    a(f"forecast = model.predict({HORIZON})   # next {HORIZON} steps, {STEP_SECONDS}s apart")
    a("```")
    a("")
    a(f"## Reproducibility")
    a(f"Seed {SEED}; Python {platform.python_version()}, statsmodels {statsmodels.__version__}, "
      f"pandas {pd.__version__}, numpy {np.__version__}.")
    Path(path).write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default=str(Path.home() / "k8sight-locust"))
    ap.add_argument("--input-csv", default="combined_all_patterns_ml.csv",
                    help="Filename (relative to --project-dir) or full path of the input CSV. "
                         "Default: combined_all_patterns_ml.csv (the corrected, ML-ready dataset).")
    args = ap.parse_args()
    project = Path(args.project_dir)
    input_arg = Path(args.input_csv)
    input_csv = input_arg if input_arg.is_absolute() else project / input_arg
    out_dir = project / "forecast_output"
    if not project.is_dir():
        fail(f"Project directory not found: {project}")
    out_dir.mkdir(exist_ok=True)
    csv_before = {p.name: sha256(p) for p in project.glob("*.csv")}

    stage("1/6 Loading and validating data")
    df = load_and_validate(input_csv)
    input_sha = sha256(input_csv)
    print(f"    {len(df)} rows across {df['pattern'].nunique()} patterns")

    stage("2/6 Building per-pattern gap-aware segments")
    segments, seg_notes = build_segments(df)
    for n in seg_notes:
        print(f"    note: {n}")
    for p in CLASS_ORDER:
        sizes = [s["n"] for s in segments[p]]
        tag = f" (split into {len(sizes)} segments: {sizes})" if len(sizes) > 1 else ""
        print(f"    {p:<14}{sum(sizes):>4} points total{tag}")

    stage("3/6 Walk-forward backtest per pattern (never across a gap)")
    per_pattern, bt_frames = {}, []
    for p in CLASS_ORDER:
        bt, n_origins, usable_segs, skipped_segs = walk_forward_backtest(p, segments[p])
        bt_frames.append(bt)
        selected, agg = select_model(bt)
        naive_mae = agg["naive"]["mae"]
        print(f"    {p:<14}origins={n_origins:<4}segs_used={usable_segs:<3}", end="")
        for m in sorted(agg):
            print(f"{m}={agg[m]['mae']:.2f}  ", end="")
        print(f"-> selected: {selected}")
        if skipped_segs:
            print(f"        skipped (too short for MIN_TRAIN+HORIZON={MIN_TRAIN + HORIZON}): {skipped_segs}")
        per_pattern[p] = {"n_origins": n_origins, "n_segments": len(segments[p]),
                          "segments_used_in_backtest": usable_segs, "segments_skipped": skipped_segs,
                          "segment_sizes": [s["n"] for s in segments[p]],
                          "selected": selected, "mae": agg[selected]["mae"], "rmse": agg[selected]["rmse"],
                          "naive_mae": naive_mae, "all_models": agg}
    backtest_all = pd.concat(bt_frames, ignore_index=True)
    backtest_all.to_csv(out_dir / "walkforward_results.csv", index=False)

    stage("4/6 Per-horizon-step error for each selected model")
    per_step = {}
    for p in CLASS_ORDER:
        sel = per_pattern[p]["selected"]
        sub = backtest_all[(backtest_all["pattern"] == p) & (backtest_all["model"] == sel)]
        step_mae = sub.groupby("horizon_step")["abs_error"].mean().to_dict()
        per_step[p] = {int(k): float(v) for k, v in step_mae.items()}
        print(f"    {p:<14}" + "  ".join(f"h{h}={step_mae[h]:.2f}" for h in sorted(step_mae)))

    stage("5/6 Fitting final models on the LATEST segment and saving artefacts")
    bundle = {}
    for p in CLASS_ORDER:
        latest = segments[p][-1]          # chronologically last continuous segment
        y_latest = latest["y"]
        registry = model_registry_for(p)
        mcls = registry[per_pattern[p]["selected"]]
        model = mcls().fit(y_latest)
        bundle[p] = {"model": model, "model_type": per_pattern[p]["selected"],
                    "n_points": len(y_latest), "n_segments_total": len(segments[p]),
                    "used_segment_index": len(segments[p]) - 1,
                    "fit_note": "fitted on the latest continuous (gap-free) segment only",
                    "target": "cpu_millicores", "horizon": HORIZON, "step_seconds": STEP_SECONDS}
        print(f"    {p:<14}fit on latest segment ({len(y_latest)} points"
              f"{', of ' + str(len(segments[p])) + ' segments' if len(segments[p]) > 1 else ''})")
    joblib.dump(bundle, out_dir / "forecasters.joblib")

    if HAVE_MPL:
        fig, axes = plt.subplots(len(CLASS_ORDER), 1, figsize=(8, 3 * len(CLASS_ORDER)))
        for ax, p in zip(axes, CLASS_ORDER):
            y = segments[p][-1]["y"]      # plot the same latest segment the final model used
            x = np.arange(len(y))
            ax.plot(x, y, label="observed (latest segment)", color="tab:blue")
            fut = bundle[p]["model"].predict(HORIZON)
            xf = np.arange(len(y), len(y) + HORIZON)
            ax.plot(xf, fut, "o--", label=f"{bundle[p]['model_type']} forecast", color="tab:orange")
            title = f"{p} (selected: {bundle[p]['model_type']})"
            if len(segments[p]) > 1:
                title += f" - latest of {len(segments[p])} segments (gap in earlier data)"
            ax.set_title(title)
            ax.set_xlabel("15s samples")
            ax.set_ylabel("CPU (millicores)")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "forecast_vs_actual.png", dpi=150)
        plt.close(fig)
    else:
        print("    [note] matplotlib not installed; skipped forecast_vs_actual.png")

    metrics = {
        "target": "cpu_millicores", "horizon_steps": HORIZON, "horizon_seconds": HORIZON * STEP_SECONDS,
        "min_train_points": MIN_TRAIN, "seasonal_period_steps": SEASONAL_PERIOD,
        "selection_rule": f"lowest MAE; within {TIE_MARGIN*100:.0f}% -> simplest model "
                          f"({' > '.join(SIMPLICITY_ORDER)})",
        "per_pattern": {p: {k: v for k, v in d.items() if k != "all_models"} | {"all_models": d["all_models"]}
                        for p, d in per_pattern.items()},
        "per_horizon_step_mae_selected_model": per_step,
        "gap_handling": {
            "method": "Each pattern's series is split into continuous, gap-free segments wherever the "
                     "gap between consecutive timestamps deviates from the 15s step by more than "
                     f"{STEP_TOLERANCE}s. Walk-forward training and forecasting never cross a segment "
                     "boundary - a gap is never bridged, and is never treated as if it were a normal 15s step.",
            "segments_too_short_are_skipped": f"Any segment with fewer than MIN_TRAIN+HORIZON="
                                              f"{MIN_TRAIN + HORIZON} points is excluded from the backtest.",
            "final_model_uses_latest_segment_only": "The model saved to forecasters.joblib is fit only on "
                                                    "each pattern's most recent continuous segment - the "
                                                    "freshest usable history - not on earlier segments before a gap.",
            "notes": seg_notes if seg_notes else ["No timestamp gaps detected in this input."],
            "per_pattern_segments": {p: {"n_segments": per_pattern[p]["n_segments"],
                                        "segment_sizes": per_pattern[p]["segment_sizes"],
                                        "segments_skipped_in_backtest": per_pattern[p]["segments_skipped"],
                                        "final_model_segment_size": bundle[p]["n_points"]}
                                     for p in CLASS_ORDER},
        },
        "limitations": [
            "One run per pattern: backtest shows within-run generalisation only.",
            f"Horizon is {HORIZON * STEP_SECONDS}s, not the original 15-20 minute target "
            "(runs are under 10 minutes long).",
            "Selected-model error is somewhat optimistic (selection and reporting share the same backtest).",
            "Per-pattern models assume the pattern is already known at forecast time.",
            "random_spike (and any other pattern with a real Prometheus gap) has less usable history for "
            "the final model than its raw row count suggests, because only the latest gap-free segment is used.",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=2))
    manifest = {"input_csv": str(input_csv), "input_csv_name": input_csv.name, "input_sha256": input_sha,
                "python": platform.python_version(), "pandas": pd.__version__,
                "numpy": np.__version__, "statsmodels": statsmodels.__version__,
                "joblib": joblib.__version__, "seed": SEED}
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    file_descriptions = {
        "forecasters.joblib": "dict of pattern -> fitted forecasting model (predict(h) -> next h CPU values)",
        "walkforward_results.csv": "every backtest prediction: pattern, origin, horizon step, model, actual, predicted",
        "metrics.json": "MAE/RMSE per pattern per model, selected model, per-horizon-step error, limitations",
        "forecast_vs_actual.png": "observed series + selected model's forecast, one panel per pattern",
        "run_manifest.json": "input hash and library versions",
        "README.md": "this document",
    }
    stage("6/6 Writing README and verifying outputs")
    write_readme(out_dir / "README.md", {"input_sha": input_sha, "input_name": input_csv.name,
                                         "per_pattern": per_pattern, "seg_notes": seg_notes,
                                         "file_descriptions": file_descriptions})
    expected = list(file_descriptions)
    if not HAVE_MPL:
        expected.remove("forecast_vs_actual.png")
    ok = True
    for name in expected:
        p = out_dir / name
        good = p.exists() and p.stat().st_size > 0
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} {name} ({p.stat().st_size if p.exists() else 0} bytes)")
    reloaded = joblib.load(out_dir / "forecasters.joblib")
    demo = reloaded["periodic"]["model"].predict(HORIZON)
    demo_ok = len(demo) == HORIZON and np.isfinite(demo).all()
    print(f"    reloaded model predicts {HORIZON} finite values   : {'OK' if demo_ok else 'BAD'}")
    csv_after = {p.name: sha256(p) for p in project.glob("*.csv")}
    unchanged = csv_before == csv_after
    print(f"    {len(csv_after)} CSV files in project unchanged (SHA-256): {'OK' if unchanged else 'BAD'}")
    ok &= demo_ok and unchanged

    print("\n" + "=" * 64)
    for p in CLASS_ORDER:
        r = per_pattern[p]
        print(f"{p:<14} selected={r['selected']:<15} MAE={r['mae']:.2f} mc  "
              f"(naive reference MAE={r['naive_mae']:.2f} mc)")
    print(f"\nOutputs: {out_dir}")
    print("RESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
