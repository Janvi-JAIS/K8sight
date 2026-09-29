#!/usr/bin/env python3
"""
K8sight - right_sizing_engine.py

Implements:
  #7 Static Allocation Baseline - the current, fixed Kubernetes config
  #6 Right-Sizing Decision Engine - per-pattern CPU/memory request & limit
     recommendations, combining a short-horizon CPU forecast with
     historical percentiles, then compared against the static baseline.

Inputs (READ-ONLY, hashed before and after; script fails if either changes):
  combined_all_patterns_ml.csv   - corrected, gap-aware dataset
  forecast_output/forecasters.joblib - per-pattern fitted CPU forecasters

No Docker / Kubernetes / Prometheus needed.

Run:  python right_sizing_engine.py
      (optional) --project-dir /path/to/dir

Methodology in one paragraph
  Static baseline is not computed - it is the actual deployed config
  (200m/500m CPU, 64Mi/128Mi memory), the same for every pattern. The
  Right-Sizing Engine computes two independent CPU estimates per pattern -
  a 60s-ahead forecast (forecast_mean, forecast_max) from forecasters.joblib,
  and historical percentiles (P90/P95/P99/max) from the SAME latest gap-free
  segment the forecaster itself was trained on (for consistency with the
  forecasting stage's gap handling) - and combines them with max(), never an
  average, so neither estimate can be silently diluted by the other. Memory
  has NO forecaster (forecasting was scoped to CPU only), so memory
  recommendations are historical-percentile only; this is stated in every
  output file, not just here. request = margin over the higher of
  (P90, forecast_mean); limit = margin over the higher of (P99, forecast_max,
  request). Static vs recommended is then compared on throttle risk (samples
  above the static CPU limit), OOM risk (samples above the static memory
  limit - this connects directly to the real OOMKilled incident on the
  untuned Random Spike run), and waste (samples far below the static request).
"""

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    import joblib
except ImportError as e:
    sys.exit(f"Missing dependency ({e}). Run: pip install pandas joblib")

try:
    # Required (even though unused directly) so joblib can unpickle the model
    # classes saved inside forecasters.joblib - they live in this shared module,
    # not inside forecast_workload.py's own namespace. Must sit in the same
    # directory as this script (or be importable) for the load below to work.
    import forecasting_models  # noqa: F401
except ImportError:
    sys.exit("forecasting_models.py not found next to right_sizing_engine.py. "
             "Copy it into the same folder (it ships alongside forecast_workload.py) and re-run.")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
CLASS_ORDER = ["stable", "increasing", "periodic", "spike", "random_spike"]
SCHEMA = ["timestamp", "pod", "cpu_cores", "cpu_millicores",
          "memory_bytes", "memory_mib", "pattern"]
STEP_SECONDS = 15
STEP_TOLERANCE = 1.5
HORIZON = 4                     # must match forecast_workload.py

# --- Static Allocation Baseline (#7) - the actual deployed config ---
STATIC_CPU_REQUEST_MC = 200
STATIC_CPU_LIMIT_MC = 500
STATIC_MEM_REQUEST_MIB = 64
STATIC_MEM_LIMIT_MIB = 128

# --- Right-Sizing Engine (#6) safety margins ---
REQUEST_MARGIN = 0.15           # 15% headroom over the higher of (P90, forecast mean)
LIMIT_MARGIN = 0.30             # 30% headroom over the higher of (P99, forecast max, request)
MIN_CPU_MC = 10
MIN_MEM_MIB = 16
CPU_ROUND_TO = 10               # round CPU recommendations up to the nearest 10m
MEM_ROUND_TO = 5                # round memory recommendations up to the nearest 5Mi
WASTE_UTIL_THRESHOLD = 0.20     # below 20% of the static request counts as "wasted" for that sample

FILE_DESCRIPTIONS = {
    "static_baseline.json": "the fixed, currently-deployed Kubernetes CPU/memory request and limit",
    "recommendations.json": "per-pattern recommended request/limit, the two estimates behind each, and the formula",
    "recommendations.csv": "the same recommendations in a flat table",
    "comparison.csv": "static vs recommended per pattern, % change, throttle/OOM risk, waste, verdict",
    "metrics.json": "everything above in one file, plus methodology and limitations",
    "README.md": "this document",
    "run_manifest.json": "input hashes and library versions",
}


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


def round_up(x, step):
    return int(math.ceil(x / step) * step)


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
    df["ts"] = parse_timestamps(df["timestamp"])
    if problems:
        fail("Data validation failed: " + "; ".join(problems))
    return df


def latest_segment(df, pattern):
    """Same gap-splitting rule as forecast_workload.py: return the chronologically
    last continuous (gap-free) segment's CPU and memory arrays for this pattern."""
    g = df[df["pattern"] == pattern].sort_values("ts").reset_index(drop=True)
    ts = g["ts"].to_numpy()
    dts = np.diff(ts)
    breaks = np.where(np.abs(dts - STEP_SECONDS) > STEP_TOLERANCE)[0]
    bounds = [0] + [b + 1 for b in breaks] + [len(g)]
    segs = [(s, e) for s, e in zip(bounds[:-1], bounds[1:])]
    s, e = segs[-1]
    seg = g.iloc[s:e]
    return seg["cpu_millicores"].to_numpy(dtype=float), seg["memory_mib"].to_numpy(dtype=float), len(segs), e - s


def percentiles(x):
    return {"p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)),
            "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
            "max": float(np.max(x)), "mean": float(np.mean(x))}


# ----------------------------------------------------------------------
# core computation
# ----------------------------------------------------------------------
def recommend_for_pattern(pattern, cpu_hist, mem_hist, forecaster_bundle):
    cpu_pct = percentiles(cpu_hist)
    mem_pct = percentiles(mem_hist)

    model_entry = forecaster_bundle[pattern]
    forecast = np.asarray(model_entry["model"].predict(HORIZON), dtype=float)
    forecast = np.clip(forecast, 0, None)  # a forecast cannot recommend negative CPU
    forecast_mean = float(forecast.mean())
    forecast_max = float(forecast.max())

    cpu_request_raw = max(cpu_pct["p90"], forecast_mean) * (1 + REQUEST_MARGIN)
    cpu_request = max(MIN_CPU_MC, round_up(cpu_request_raw, CPU_ROUND_TO))
    cpu_limit_raw = max(cpu_pct["p99"], forecast_max, cpu_request) * (1 + LIMIT_MARGIN)
    cpu_limit = max(cpu_request, round_up(cpu_limit_raw, CPU_ROUND_TO))

    mem_request_raw = mem_pct["p90"] * (1 + REQUEST_MARGIN)
    mem_request = max(MIN_MEM_MIB, round_up(mem_request_raw, MEM_ROUND_TO))
    mem_limit_raw = max(mem_pct["p99"], mem_pct["max"]) * (1 + LIMIT_MARGIN)
    mem_limit = max(mem_request, round_up(mem_limit_raw, MEM_ROUND_TO))

    return {
        "pattern": pattern,
        "n_historical_samples": int(len(cpu_hist)),
        "cpu_historical_percentiles_mc": cpu_pct,
        "cpu_forecast_mc": {"mean_next_60s": forecast_mean, "max_next_60s": forecast_max,
                            "values": forecast.tolist()},
        "cpu_request_mc": cpu_request, "cpu_limit_mc": cpu_limit,
        "cpu_request_driver": "forecast_mean" if forecast_mean > cpu_pct["p90"] else "historical_p90",
        "cpu_limit_driver": ("forecast_max" if forecast_max >= cpu_pct["p99"]
                             else "historical_p99") if cpu_limit_raw > cpu_request else "cpu_request_floor",
        "memory_historical_percentiles_mib": mem_pct,
        "memory_forecast_used": False,
        "mem_request_mib": mem_request, "mem_limit_mib": mem_limit,
    }


def compare_to_static(rec, cpu_hist, mem_hist):
    n = len(cpu_hist)
    throttle_frac = float((cpu_hist > STATIC_CPU_LIMIT_MC).mean())
    oom_frac = float((mem_hist > STATIC_MEM_LIMIT_MIB).mean())
    waste_frac = float((cpu_hist < STATIC_CPU_REQUEST_MC * WASTE_UTIL_THRESHOLD).mean())

    if throttle_frac > 0 or oom_frac > 0:
        verdict = "static under-provisioned"
    elif waste_frac > 0.5:
        verdict = "static over-provisioned"
    else:
        verdict = "static roughly adequate"

    def pct_change(new, old):
        return float((new - old) / old * 100.0)

    return {
        "pattern": rec["pattern"],
        "static_cpu_request_mc": STATIC_CPU_REQUEST_MC, "static_cpu_limit_mc": STATIC_CPU_LIMIT_MC,
        "static_mem_request_mib": STATIC_MEM_REQUEST_MIB, "static_mem_limit_mib": STATIC_MEM_LIMIT_MIB,
        "recommended_cpu_request_mc": rec["cpu_request_mc"], "recommended_cpu_limit_mc": rec["cpu_limit_mc"],
        "recommended_mem_request_mib": rec["mem_request_mib"], "recommended_mem_limit_mib": rec["mem_limit_mib"],
        "cpu_request_change_pct": pct_change(rec["cpu_request_mc"], STATIC_CPU_REQUEST_MC),
        "cpu_limit_change_pct": pct_change(rec["cpu_limit_mc"], STATIC_CPU_LIMIT_MC),
        "mem_request_change_pct": pct_change(rec["mem_request_mib"], STATIC_MEM_REQUEST_MIB),
        "mem_limit_change_pct": pct_change(rec["mem_limit_mib"], STATIC_MEM_LIMIT_MIB),
        "throttle_risk_fraction_of_samples_over_static_limit": throttle_frac,
        "oom_risk_fraction_of_samples_over_static_mem_limit": oom_frac,
        "waste_fraction_of_samples_under_20pct_of_static_request": waste_frac,
        "n_samples_evaluated": n,
        "verdict": verdict,
        "verdict_rule": "throttle or OOM risk > 0 -> under-provisioned; "
                       "else waste > 50% of samples -> over-provisioned; else adequate",
    }


# ----------------------------------------------------------------------
# README
# ----------------------------------------------------------------------
def write_readme(path, ctx):
    L = []
    a = L.append
    a("# K8sight - Right-Sizing Decision Engine & Static Allocation Baseline")
    a("")
    a("Produced by `right_sizing_engine.py`. Implements Functionality #7 (Static Allocation Baseline: "
      "the currently-deployed, fixed Kubernetes config) and #6 (Right-Sizing Decision Engine: per-pattern "
      "CPU/memory request and limit recommendations), then compares the two.")
    a("")
    a("## Static Allocation Baseline (#7)")
    a("Not computed - this is the actual current deployment, the same for every pattern:")
    a(f"- CPU: request `{STATIC_CPU_REQUEST_MC}m`, limit `{STATIC_CPU_LIMIT_MC}m`")
    a(f"- Memory: request `{STATIC_MEM_REQUEST_MIB}Mi`, limit `{STATIC_MEM_LIMIT_MIB}Mi`")
    a("")
    a("## Right-Sizing Decision Engine (#6)")
    a("")
    a("**Inputs (read-only):**")
    a(f"- `{ctx['input_name']}` - CPU and memory history, per pattern")
    a("- `forecast_output/forecasters.joblib` - each pattern's fitted CPU forecaster, already trained on "
      "that pattern's latest gap-free segment (see the forecasting stage's README)")
    a("")
    a("**Per pattern, two independent estimates, combined with `max()` (never averaged):**")
    a("1. *Historical* - P50/P90/P95/P99/max of CPU (and memory) from the SAME latest gap-free segment the "
      "forecaster was trained on, for consistency with the forecasting stage's gap handling.")
    a(f"2. *Forecast* - the saved model's {HORIZON}-step ({HORIZON * STEP_SECONDS}s) CPU forecast; "
      "mean and max of those steps. **CPU only** - forecasting was scoped to CPU, so memory recommendations "
      "below are historical-percentile only, with no forecast component.")
    a("")
    a("**Formulas:**")
    a("```")
    a(f"CPU request  = max(MIN_CPU_MC={MIN_CPU_MC}, round_up_to_{CPU_ROUND_TO}m(")
    a(f"                 max(P90_cpu_hist, forecast_mean_next_{HORIZON*STEP_SECONDS}s) * (1 + {REQUEST_MARGIN})))")
    a(f"CPU limit    = max(CPU_request, round_up_to_{CPU_ROUND_TO}m(")
    a(f"                 max(P99_cpu_hist, forecast_max_next_{HORIZON*STEP_SECONDS}s, CPU_request) * (1 + {LIMIT_MARGIN})))")
    a("")
    a(f"Memory request = max(MIN_MEM_MIB={MIN_MEM_MIB}, round_up_to_{MEM_ROUND_TO}Mi(P90_mem_hist * (1 + {REQUEST_MARGIN})))")
    a(f"Memory limit   = max(Memory_request, round_up_to_{MEM_ROUND_TO}Mi(max(P99_mem_hist, max_mem_hist) * (1 + {LIMIT_MARGIN})))")
    a("```")
    a(f"- Request margin: {REQUEST_MARGIN*100:.0f}%. Limit margin: {LIMIT_MARGIN*100:.0f}%. "
      "Both fixed before computing any recommendation, not tuned per pattern after seeing results.")
    a("- CPU values cannot go negative even if a forecast model extrapolates below zero (clipped at 0 before use).")
    a("")
    a("## Comparison methodology (static vs recommended)")
    a("For each pattern, using the same historical sample window as the recommendation:")
    a(f"- **Throttle risk** - fraction of samples where CPU exceeded the *static* limit ({STATIC_CPU_LIMIT_MC}m). "
      "Kubernetes throttles a container's CPU once it hits its limit; any nonzero fraction here means the "
      "static config would have throttled this workload.")
    a(f"- **OOM risk** - fraction of samples where memory exceeded the *static* limit ({STATIC_MEM_LIMIT_MIB}Mi). "
      "This is the same failure mode that OOMKilled the untuned Random Spike run earlier in the project - "
      "a direct, concrete link between that incident and this stage's numbers.")
    a(f"- **Waste** - fraction of samples where CPU was under {WASTE_UTIL_THRESHOLD*100:.0f}% of the *static* "
      f"request ({STATIC_CPU_REQUEST_MC}m), i.e. the request reserved far more than was typically used.")
    a("- **Verdict** (a fixed rule, not a per-pattern judgment call): throttle or OOM risk above zero -> "
      "*static under-provisioned*; else waste above 50% of samples -> *static over-provisioned*; "
      "otherwise -> *static roughly adequate*.")
    a("")
    a("## Results")
    a("")
    a("| Pattern | Static CPU req/limit | Recommended CPU req/limit | Throttle risk | OOM risk | Waste | Verdict |")
    a("|---|---|---|---|---|---|---|")
    for p in CLASS_ORDER:
        c = ctx["comparison"][p]
        a(f"| {p} | {c['static_cpu_request_mc']}m / {c['static_cpu_limit_mc']}m | "
          f"{c['recommended_cpu_request_mc']}m / {c['recommended_cpu_limit_mc']}m | "
          f"{c['throttle_risk_fraction_of_samples_over_static_limit']*100:.0f}% | "
          f"{c['oom_risk_fraction_of_samples_over_static_mem_limit']*100:.0f}% | "
          f"{c['waste_fraction_of_samples_under_20pct_of_static_request']*100:.0f}% | {c['verdict']} |")
    a("")
    a("| Pattern | Static memory req/limit | Recommended memory req/limit |")
    a("|---|---|---|")
    for p in CLASS_ORDER:
        c = ctx["comparison"][p]
        a(f"| {p} | {c['static_mem_request_mib']}Mi / {c['static_mem_limit_mib']}Mi | "
          f"{c['recommended_mem_request_mib']}Mi / {c['recommended_mem_limit_mib']}Mi |")
    a("")
    a("## Limitations")
    a("- **Memory is historical-percentile only** - no forecast component, because forecasting was scoped to CPU.")
    a("- **One run per pattern** (same limitation as the classifier and forecaster stages): these percentiles "
      "and risk fractions describe the one collected run per pattern, not a general claim about the workload.")
    a("- **Recommendations depend on the forecaster's own limitations** - the forecast component inherits "
      "the forecasting stage's 60s horizon and within-run-only validation.")
    a("- **Margins (15%/30%) are fixed, reasonable defaults, not tuned or validated against an SLO.** "
      "A production system would set these from an actual latency/error-budget target.")
    a("- **This is a proof of concept comparing one static config against five short synthetic runs**, not "
      "a production sizing recommendation.")
    a("")
    a("## Files")
    for f, d in ctx["file_descriptions"].items():
        a(f"- `{f}` - {d}")
    a("")
    a("## Reproducibility")
    a(f"Python {platform.python_version()}, pandas {pd.__version__}, numpy {np.__version__}, "
      f"joblib {joblib.__version__}. Deterministic given the same inputs (no randomness in this stage).")
    Path(path).write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default=str(Path.home() / "k8sight-locust"))
    ap.add_argument("--input-csv", default="combined_all_patterns_ml.csv")
    args = ap.parse_args()
    project = Path(args.project_dir)
    input_arg = Path(args.input_csv)
    input_csv = input_arg if input_arg.is_absolute() else project / input_arg
    forecaster_path = project / "forecast_output" / "forecasters.joblib"
    out_dir = project / "right_sizing_output"
    if not project.is_dir():
        fail(f"Project directory not found: {project}")
    if not forecaster_path.exists():
        fail(f"Forecaster not found: {forecaster_path}. Run forecast_workload.py first.")
    out_dir.mkdir(exist_ok=True)

    protected = [p for p in project.rglob("*") if p.is_file() and out_dir not in p.parents]
    hashes_before = {str(p.relative_to(project)): sha256(p) for p in protected}

    stage("1/6 Loading and validating data + forecasters")
    df = load_and_validate(input_csv)
    input_sha = sha256(input_csv)
    bundle = joblib.load(forecaster_path)
    missing = [p for p in CLASS_ORDER if p not in bundle]
    if missing:
        fail(f"forecasters.joblib is missing patterns: {missing}")
    print(f"    {len(df)} rows, {len(bundle)} forecasters loaded")

    stage("2/6 Static Allocation Baseline (#7)")
    static_baseline = {"cpu_request_mc": STATIC_CPU_REQUEST_MC, "cpu_limit_mc": STATIC_CPU_LIMIT_MC,
                       "mem_request_mib": STATIC_MEM_REQUEST_MIB, "mem_limit_mib": STATIC_MEM_LIMIT_MIB,
                       "note": "fixed, currently-deployed config; identical for every pattern"}
    (out_dir / "static_baseline.json").write_text(json.dumps(static_baseline, indent=2))
    print(f"    CPU {STATIC_CPU_REQUEST_MC}m/{STATIC_CPU_LIMIT_MC}m, "
          f"memory {STATIC_MEM_REQUEST_MIB}Mi/{STATIC_MEM_LIMIT_MIB}Mi")

    stage("3/6 Computing recommendations (#6) per pattern")
    recs, comparisons, hist_info = {}, {}, {}
    for p in CLASS_ORDER:
        cpu_hist, mem_hist, n_segs, seg_len = latest_segment(df, p)
        if len(cpu_hist) < 4:
            fail(f"{p}: latest segment has only {len(cpu_hist)} points; too few for percentiles.")
        hist_info[p] = {"n_segments_total": n_segs, "latest_segment_length": seg_len}
        rec = recommend_for_pattern(p, cpu_hist, mem_hist, bundle)
        recs[p] = rec
        comparisons[p] = compare_to_static(rec, cpu_hist, mem_hist)
        seg_note = f" (of {n_segs} segments)" if n_segs > 1 else ""
        print(f"    {p:<14}n={seg_len:<4}{seg_note:<16}CPU req/limit={rec['cpu_request_mc']}m/"
              f"{rec['cpu_limit_mc']}m  mem req/limit={rec['mem_request_mib']}Mi/{rec['mem_limit_mib']}Mi")

    stage("4/6 Comparing recommendations to the static baseline")
    print(f"    {'pattern':<14}{'throttle%':>10}{'oom%':>7}{'waste%':>8}  verdict")
    for p in CLASS_ORDER:
        c = comparisons[p]
        print(f"    {p:<14}{c['throttle_risk_fraction_of_samples_over_static_limit']*100:>9.1f}%"
              f"{c['oom_risk_fraction_of_samples_over_static_mem_limit']*100:>6.1f}%"
              f"{c['waste_fraction_of_samples_under_20pct_of_static_request']*100:>7.1f}%  {c['verdict']}")

    stage("5/6 Writing outputs")
    (out_dir / "recommendations.json").write_text(json.dumps(jsonable(recs), indent=2))
    pd.DataFrame([{
        "pattern": p, "n_historical_samples": recs[p]["n_historical_samples"],
        "cpu_request_mc": recs[p]["cpu_request_mc"], "cpu_limit_mc": recs[p]["cpu_limit_mc"],
        "cpu_request_driver": recs[p]["cpu_request_driver"], "cpu_limit_driver": recs[p]["cpu_limit_driver"],
        "mem_request_mib": recs[p]["mem_request_mib"], "mem_limit_mib": recs[p]["mem_limit_mib"],
        "cpu_forecast_mean_mc": recs[p]["cpu_forecast_mc"]["mean_next_60s"],
        "cpu_forecast_max_mc": recs[p]["cpu_forecast_mc"]["max_next_60s"],
        "cpu_p90_hist_mc": recs[p]["cpu_historical_percentiles_mc"]["p90"],
        "cpu_p99_hist_mc": recs[p]["cpu_historical_percentiles_mc"]["p99"],
    } for p in CLASS_ORDER]).to_csv(out_dir / "recommendations.csv", index=False)
    pd.DataFrame([comparisons[p] for p in CLASS_ORDER]).to_csv(out_dir / "comparison.csv", index=False)

    metrics = {
        "static_baseline": static_baseline,
        "margins": {"request_margin": REQUEST_MARGIN, "limit_margin": LIMIT_MARGIN,
                   "min_cpu_mc": MIN_CPU_MC, "min_mem_mib": MIN_MEM_MIB,
                   "cpu_round_to_mc": CPU_ROUND_TO, "mem_round_to_mib": MEM_ROUND_TO,
                   "waste_util_threshold": WASTE_UTIL_THRESHOLD},
        "recommendations": jsonable(recs),
        "comparison": jsonable(comparisons),
        "historical_segment_info": hist_info,
        "limitations": [
            "Memory is historical-percentile only; no memory forecaster exists (forecasting scoped to CPU).",
            "One run per pattern: percentiles and risk fractions describe that single run, not a general claim.",
            "Recommendations inherit the forecasting stage's 60s horizon and within-run-only validation.",
            "Margins (15% request / 30% limit) are fixed reasonable defaults, not SLO-derived.",
            "Proof of concept: one static config compared against five short synthetic runs.",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=2))
    manifest = {"input_csv": str(input_csv), "input_csv_sha256": input_sha,
               "forecaster_path": str(forecaster_path), "forecaster_sha256": sha256(forecaster_path),
               "python": platform.python_version(), "pandas": pd.__version__,
               "numpy": np.__version__, "joblib": joblib.__version__}
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_readme(out_dir / "README.md", {"input_name": input_csv.name, "comparison": comparisons,
                                         "file_descriptions": FILE_DESCRIPTIONS})

    stage("6/6 Verifying outputs and read-only inputs")
    ok = True
    for name in FILE_DESCRIPTIONS:
        fp = out_dir / name
        good = fp.exists() and fp.stat().st_size > 0
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} {name} ({fp.stat().st_size if fp.exists() else 0} bytes)")
    reloaded = json.loads((out_dir / "recommendations.json").read_text())
    consistent = all(reloaded[p]["cpu_limit_mc"] >= reloaded[p]["cpu_request_mc"]
                     and reloaded[p]["mem_limit_mib"] >= reloaded[p]["mem_request_mib"] for p in CLASS_ORDER)
    print(f"    limit >= request for every pattern           : {'OK' if consistent else 'BAD'}")
    ok &= consistent
    protected_after = [p for p in project.rglob("*") if p.is_file() and out_dir not in p.parents]
    hashes_after = {str(p.relative_to(project)): sha256(p) for p in protected_after}
    unchanged = hashes_before == hashes_after
    print(f"    {len(hashes_after)} project files unchanged (SHA-256)   : {'OK' if unchanged else 'BAD'}")
    ok &= unchanged

    print("\n" + "=" * 70)
    for p in CLASS_ORDER:
        c = comparisons[p]
        print(f"{p:<14} static={c['static_cpu_request_mc']}m/{c['static_cpu_limit_mc']}m  "
              f"recommended={c['recommended_cpu_request_mc']}m/{c['recommended_cpu_limit_mc']}m  "
              f"-> {c['verdict']}")
    print(f"\nOutputs: {out_dir}")
    print("RESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
