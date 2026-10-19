"""Read-only integration of designated runs, eligibility evidence, and windows.

This module does not write a dataset or acceptance record. Callers must pass
an explicit accepted-run set, expected metadata (including the designated
final output path), and per-run evidence shaped like ``eligibility_audit``'s
``audit_run`` result. A fresh audit is run and compared before observations
reach Stage 2. Schema validity, acceptance, training eligibility, row
selection, and window eligibility remain separate results.

Future writer contract (not implemented): require a new unique destination and
refuse an existing one; record accepted IDs and exact source mapping, schema,
feature units, selection/exclusion rules, source hashes, and build provenance;
keep outputs separate from legacy CSVs and classifier artifacts.
"""
from __future__ import annotations

import collections
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from dataset_schema_v1.eligibility_audit import RunSpec, audit_run
from dataset_schema_v1.phase_aware_windows import (
    FEATURE_UNITS,
    RowDecision,
    WindowDecision,
    build_feature_windows,
    select_eligible_observations,
)
from dataset_schema_v1.validate_run import COLS


@dataclass(frozen=True)
class RunBuildResult:
    """In-memory verification, selection, and window results for one run."""

    run_id: str
    pattern: str | None
    source_output_dir: str | None
    status: str
    reasons: tuple[str, ...]
    source_verification: Mapping[str, Any]
    acceptance_evidence: Mapping[str, Any]
    eligibility_evidence: Mapping[str, Any]
    input_observation_count: int
    selected_observation_count: int
    excluded_observation_counts: Mapping[str, int]
    candidate_window_count: int
    eligible_window_count: int
    excluded_window_counts: Mapping[str, int]
    row_decisions: tuple[RowDecision, ...]
    windows: tuple[WindowDecision, ...]


@dataclass(frozen=True)
class DatasetBuildResult:
    """Deterministic collection of per-run results; no disk output is made."""

    runs: tuple[RunBuildResult, ...]


def _blocked(
    run_id: str,
    expected: Mapping[str, Any] | None,
    reason: str,
    *,
    output: str | None = None,
    input_count: int = 0,
    source_verification: Mapping[str, Any] | None = None,
    acceptance: Mapping[str, Any] | None = None,
    eligibility: Mapping[str, Any] | None = None,
) -> RunBuildResult:
    return RunBuildResult(
        run_id=run_id,
        pattern=expected.get("pattern") if expected else None,
        source_output_dir=output,
        status="blocked",
        reasons=(reason,),
        source_verification=dict(source_verification or {"status": "blocked", "reasons": [reason]}),
        acceptance_evidence=dict(acceptance or {"decision": "not_evaluated"}),
        eligibility_evidence=dict(eligibility or {"status": "unavailable"}),
        input_observation_count=input_count,
        selected_observation_count=0,
        excluded_observation_counts={reason: input_count} if input_count else {},
        candidate_window_count=0,
        eligible_window_count=0,
        excluded_window_counts={},
        row_decisions=(),
        windows=(),
    )


def _resolve_under_root(path_value: Any, root: Path, *, must_exist: bool) -> Path:
    if not isinstance(path_value, (str, Path)) or not str(path_value):
        raise ValueError("path_missing_or_invalid")
    path = Path(path_value)
    resolved = (root / path).resolve(strict=must_exist) if not path.is_absolute() else path.resolve(strict=must_exist)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("path_outside_project_root") from exc
    return resolved


def _evidence_problem(evidence: Any, run_id: str, expected_pattern: str) -> str | None:
    if not isinstance(evidence, Mapping):
        return "eligibility_evidence_missing_or_malformed"
    if evidence.get("run_id") != run_id:
        return "eligibility_evidence_run_id_mismatch"
    if evidence.get("expected_pattern") != expected_pattern:
        return "eligibility_evidence_pattern_mismatch"
    if evidence.get("status") not in {"pass", "review_required"}:
        return "eligibility_evidence_blocked_or_invalid_status"
    if evidence.get("schema_validity") != "pass":
        return "schema_validation_not_pass"
    training = evidence.get("training_eligibility")
    if not isinstance(training, Mapping) or training.get("status") != "observed" or not isinstance(training.get("observed"), bool):
        return "training_eligibility_evidence_missing_or_malformed"
    if evidence.get("acceptance_decision") != "not_made":
        return "eligibility_audit_acceptance_field_invalid"
    if evidence.get("source_integrity") != "verified":
        return "source_integrity_not_verified"
    return None


def _same_audit_evidence(supplied: Mapping[str, Any], fresh: Mapping[str, Any]) -> bool:
    # Stage 1's public audit result is a JSON-shaped mapping. Exact comparison
    # catches stale caller evidence whenever its reported counts, caveats, or
    # source reference differ from the fresh read-only audit.
    return dict(supplied) == dict(fresh)


def _read_source(output: Path, run_id: str, pattern: str) -> tuple[dict, list[dict[str, str]]]:
    manifest_path = output / "manifest.json"
    observations_path = output / "observations.csv"
    if not manifest_path.is_file() or not observations_path.is_file():
        raise ValueError("required_manifest_or_observations_missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest_not_object")
        with observations_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != COLS:
                raise ValueError("observations_header_mismatch")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
        raise ValueError("source_parse_failed: " + str(exc)) from exc
    if manifest.get("run_id") != run_id:
        raise ValueError("manifest_run_id_mismatch")
    if manifest.get("pattern") != pattern:
        raise ValueError("manifest_pattern_mismatch")
    dq = manifest.get("data_quality")
    if not isinstance(dq, Mapping):
        raise ValueError("manifest_data_quality_missing_or_malformed")
    for key in ("expected_samples", "observed_samples"):
        value = dq.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != len(rows):
            raise ValueError("manifest_" + key + "_row_count_mismatch")
    if dq.get("source_integrity") != "verified":
        raise ValueError("manifest_source_integrity_not_verified")
    return manifest, rows


def build_in_memory(
    source_dirs: Mapping[str, str | Path],
    accepted_run_ids: Iterable[str] | None,
    eligibility_evidence: Mapping[str, Mapping[str, Any]],
    expected_run_metadata: Mapping[str, Mapping[str, Any]],
    *,
    project_root: str | Path,
    window_size: int = 8,
    cadence_seconds: float = 15.0,
    tolerance_seconds: float = 0.001,
) -> DatasetBuildResult:
    """Verify source runs and return selection/windows entirely in memory.

    ``source_dirs`` maps each run ID to its designated output. The matching
    ``expected_run_metadata`` entry must provide ``pattern``,
    ``designated_output_dir`` (relative to ``project_root``),
    ``experiment_metadata_path`` (relative to ``project_root``), ``replicate``,
    ``seed``, and ``observation_units``. ``eligibility_evidence`` entries must
    be Stage 1 ``audit_run`` result mappings. The function performs a fresh
    audit and requires the supplied result's identity and gate evidence to
    match. A missing acceptance set is distinct from an explicit empty set;
    neither admits a run.
    """
    if not isinstance(source_dirs, Mapping):
        raise ValueError("source_dirs_must_be_mapping")
    if not isinstance(expected_run_metadata, Mapping) or not isinstance(eligibility_evidence, Mapping):
        raise ValueError("metadata_and_eligibility_evidence_must_be_mappings")
    if any(not isinstance(run_id, str) or not run_id for run_id in source_dirs):
        raise ValueError("source_mapping_run_id_missing_or_invalid")
    source_ids = set(source_dirs)
    if set(expected_run_metadata) - source_ids or set(eligibility_evidence) - source_ids:
        raise ValueError("unmapped_expected_metadata_or_eligibility_evidence")
    if isinstance(accepted_run_ids, str):
        acceptance = None
        acceptance_invalid = True
    else:
        try:
            acceptance = None if accepted_run_ids is None else frozenset(accepted_run_ids)
            acceptance_invalid = acceptance is not None and any(not isinstance(value, str) or not value for value in acceptance)
        except TypeError:
            acceptance = None
            acceptance_invalid = True

    root = Path(project_root).resolve(strict=True)
    run_ids = sorted(source_dirs)
    resolved_paths: dict[str, Path] = {}
    path_owners: dict[Path, list[str]] = collections.defaultdict(list)
    path_errors: dict[str, str] = {}
    for run_id in run_ids:
        try:
            if not isinstance(run_id, str) or not run_id:
                raise ValueError("run_id_missing_or_invalid")
            expected = expected_run_metadata.get(run_id)
            if not isinstance(expected, Mapping):
                raise ValueError("expected_run_metadata_missing_or_malformed")
            if expected.get("run_id") != run_id:
                raise ValueError("expected_run_metadata_run_id_mismatch")
            if not isinstance(expected.get("pattern"), str) or not expected.get("pattern"):
                raise ValueError("expected_pattern_missing_or_invalid")
            output = _resolve_under_root(source_dirs[run_id], root, must_exist=True)
            designated = _resolve_under_root(expected.get("designated_output_dir"), root, must_exist=True)
            if output != designated:
                raise ValueError("source_not_designated_final_output")
            resolved_paths[run_id] = output
            path_owners[output].append(run_id)
        except (OSError, RuntimeError, ValueError) as exc:
            path_errors[run_id] = str(exc)

    duplicate_owners = {run_id for owners in path_owners.values() if len(owners) > 1 for run_id in owners}
    results: list[RunBuildResult] = []
    for run_id in run_ids:
        expected = expected_run_metadata.get(run_id)
        if run_id in path_errors:
            results.append(_blocked(run_id, expected if isinstance(expected, Mapping) else None, path_errors[run_id], output=str(source_dirs[run_id])))
            continue
        output = resolved_paths[run_id]
        output_ref = str(output)
        if run_id in duplicate_owners:
            results.append(_blocked(run_id, expected, "duplicate_or_ambiguous_source_mapping", output=output_ref))
            continue
        pattern = expected["pattern"]

        try:
            relative_output = output.relative_to(root).as_posix()
            metadata_path = _resolve_under_root(expected.get("experiment_metadata_path"), root, must_exist=True)
            relative_metadata = metadata_path.relative_to(root).as_posix()
            if (not isinstance(expected.get("replicate"), int) or isinstance(expected.get("replicate"), bool)
                    or expected.get("replicate") < 1):
                raise ValueError("expected_replicate_missing_or_invalid")
            if not isinstance(expected.get("seed"), int) or isinstance(expected.get("seed"), bool):
                raise ValueError("expected_seed_missing_or_invalid")
            units = expected.get("observation_units")
            if not isinstance(units, Mapping) or any(units.get(key) != unit for key, unit in FEATURE_UNITS.items()):
                raise ValueError("expected_feature_units_missing_or_invalid")
        except (OSError, RuntimeError, ValueError) as exc:
            results.append(_blocked(run_id, expected, str(exc), output=output_ref))
            continue

        spec = RunSpec(run_id, pattern, (relative_output,), relative_metadata)
        fresh = audit_run(spec, root)
        supplied = eligibility_evidence.get(run_id)
        evidence_problem = _evidence_problem(supplied, run_id, pattern)
        if evidence_problem is None and isinstance(supplied, Mapping) and supplied.get("output_dir") != output_ref:
            evidence_problem = "eligibility_evidence_source_path_mismatch"
        if evidence_problem:
            results.append(_blocked(run_id, expected, evidence_problem, output=output_ref,
                                    source_verification={"status": fresh.get("status"), "schema_validity": fresh.get("schema_validity"), "reasons": fresh.get("reasons", [])},
                                    eligibility=supplied if isinstance(supplied, Mapping) else None))
            continue
        assert isinstance(supplied, Mapping)
        fresh_problem = _evidence_problem(fresh, run_id, pattern)
        if fresh_problem or fresh.get("status") == "blocked" or not _same_audit_evidence(supplied, fresh):
            reason = fresh_problem or "fresh_audit_blocked_or_evidence_mismatch"
            results.append(_blocked(run_id, expected, reason, output=output_ref,
                                    source_verification={"status": fresh.get("status"), "schema_validity": fresh.get("schema_validity"), "reasons": fresh.get("reasons", [])},
                                    eligibility=supplied))
            continue

        try:
            manifest, rows = _read_source(output, run_id, pattern)
        except (OSError, UnicodeError, csv.Error, json.JSONDecodeError, ValueError) as exc:
            results.append(_blocked(run_id, expected, str(exc), output=output_ref,
                                    source_verification={"status": "blocked", "schema_validity": fresh.get("schema_validity"), "reasons": [str(exc)]},
                                    acceptance={"decision": "accepted" if acceptance is not None and run_id in acceptance else "not_accepted"},
                                    eligibility=supplied))
            continue

        if acceptance_invalid:
            results.append(_blocked(run_id, expected, "acceptance_set_invalid", output=output_ref, input_count=len(rows), eligibility=supplied))
            continue
        if acceptance is None:
            results.append(RunBuildResult(run_id, pattern, output_ref, "not_accepted", ("acceptance_set_missing",),
                {"status": "verified", "schema_validity": fresh["schema_validity"], "source_integrity": fresh["source_integrity"]},
                {"decision": "missing"}, dict(supplied), len(rows), 0,
                {"acceptance_set_missing": len(rows)}, 0, 0, {}, (), ()))
            continue
        if run_id not in acceptance:
            results.append(RunBuildResult(run_id, pattern, output_ref, "not_accepted", ("run_not_explicitly_accepted",),
                {"status": "verified", "schema_validity": fresh["schema_validity"], "source_integrity": fresh["source_integrity"]},
                {"decision": "not_accepted"}, dict(supplied), len(rows), 0,
                {"run_not_explicitly_accepted": len(rows)}, 0, 0, {}, (), ()))
            continue
        training = supplied["training_eligibility"]
        if training["observed"] is not True:
            results.append(_blocked(run_id, expected, "training_ineligible", output=output_ref, input_count=len(rows),
                                    source_verification={"status": "verified", "schema_validity": fresh["schema_validity"], "source_integrity": fresh["source_integrity"]},
                                    acceptance={"decision": "accepted"}, eligibility=supplied))
            continue

        metadata_for_windows = {
            run_id: {
                "schema_validity": fresh["schema_validity"],
                "training_eligible": training["observed"],
                "pattern": pattern,
                "replicate": expected["replicate"],
                "seed": expected["seed"],
                "observation_units": dict(units),
            }
        }
        selection = select_eligible_observations(rows, metadata_for_windows, {run_id})
        window_result = build_feature_windows(selection, metadata_for_windows,
            window_size=window_size, cadence_seconds=cadence_seconds, tolerance_seconds=tolerance_seconds)
        status = "eligible_for_windowing" if fresh["status"] == "pass" else "review_required"
        reasons = tuple(fresh.get("reasons", []))
        results.append(RunBuildResult(
            run_id=run_id,
            pattern=pattern,
            source_output_dir=output_ref,
            status=status,
            reasons=reasons,
            source_verification={"status": "verified", "schema_validity": fresh["schema_validity"], "source_integrity": fresh["source_integrity"]},
            acceptance_evidence={"decision": "accepted", "source": "caller_supplied_accepted_run_ids"},
            eligibility_evidence=dict(supplied),
            input_observation_count=len(rows),
            selected_observation_count=selection.included_count,
            excluded_observation_counts=selection.exclusion_counts,
            candidate_window_count=window_result.candidate_count,
            eligible_window_count=len(window_result.eligible_windows),
            excluded_window_counts=window_result.exclusion_counts,
            row_decisions=selection.decisions,
            windows=window_result.decisions,
        ))
    return DatasetBuildResult(tuple(results))
