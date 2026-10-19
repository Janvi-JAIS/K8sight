"""Purpose-authorized, read-only integration for the in-memory dataset builder."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from dataset_schema_v1.acceptance_record_store import (
    AcceptanceRecordStoreError,
    load_validated_acceptance_history,
)
from dataset_schema_v1.acceptance_records import authorize_purpose
from dataset_schema_v1.acceptance_records import PURPOSES
from dataset_schema_v1.dataset_builder import (
    DatasetBuildResult,
    RunBuildResult,
    _blocked,
    build_in_memory,
)


def build_from_acceptance_store(
    source_dirs: Mapping[str, str | Path],
    eligibility_evidence: Mapping[str, Mapping[str, Any]],
    expected_run_metadata: Mapping[str, Mapping[str, Any]],
    *,
    project_root: str | Path,
    purpose: str,
    window_size: int = 8,
    cadence_seconds: float = 15.0,
    tolerance_seconds: float = 0.001,
) -> DatasetBuildResult:
    """Build in memory using only runs authorized by the complete record store."""
    run_ids = sorted(source_dirs)

    if not isinstance(purpose, str) or purpose not in PURPOSES:
        reason = "purpose_invalid"
        return _blocked_runs(source_dirs, expected_run_metadata, purpose, reason)
    try:
        root = Path(project_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("project_root_not_directory")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        reason = "project_root_invalid:" + (str(exc) or type(exc).__name__)
        return _blocked_runs(source_dirs, expected_run_metadata, purpose, reason)

    try:
        records = load_validated_acceptance_history(root)
    except (AcceptanceRecordStoreError, OSError, ValueError) as exc:
        reason = "acceptance_store_invalid:" + str(exc)
        return DatasetBuildResult(tuple(
            RunBuildResult(
                run_id=run_id,
                pattern=(
                    expected_run_metadata.get(run_id, {}).get("pattern")
                    if isinstance(expected_run_metadata.get(run_id), Mapping)
                    else None
                ),
                source_output_dir=str(source_dirs[run_id]),
                status="blocked",
                reasons=(reason,),
                source_verification={"status": "not_verified", "reasons": [reason]},
                acceptance_evidence={"decision": "not_authorized", "purpose": purpose},
                eligibility_evidence={"status": "not_evaluated"},
                input_observation_count=0,
                selected_observation_count=0,
                excluded_observation_counts={},
                candidate_window_count=0,
                eligible_window_count=0,
                excluded_window_counts={},
                row_decisions=(),
                windows=(),
            )
            for run_id in run_ids
        ))

    authorized_ids: set[str] = set()
    authorization_by_run: dict[str, dict[str, Any]] = {}
    for run_id in run_ids:
        expected = expected_run_metadata.get(run_id)
        if not isinstance(expected, Mapping):
            authorization_by_run[run_id] = {
                "authorized": False,
                "errors": ["expected_run_metadata_missing_or_malformed"],
                "record_id": None,
            }
            continue

        pattern = expected.get("pattern")
        output_path = expected.get("designated_output_dir")
        matching = [
            record for record in records
            if record.get("run_id") == run_id
            and record.get("pattern") == pattern
        ]

        history = [
            record for record in matching
            if record.get("designated_output_path") == output_path
        ]
        current_candidates = []
        for record in history:
            result = authorize_purpose(
                record,
                history,
                purpose,
                root,
                expected_run_id=run_id,
                expected_pattern=pattern,
                expected_output_path=output_path,
            )
            if result.authorized:
                current_candidates.append(record)

        if len(current_candidates) == 1:
            record = current_candidates[0]
            authorized_ids.add(run_id)
            authorization_by_run[run_id] = {
                "authorized": True,
                "record_id": record["acceptance_record_id"],
                "purpose": purpose,
            }
        else:
            errors = []
            for record in matching:
                result = authorize_purpose(
                    record,
                    matching,
                    purpose,
                    root,
                    expected_run_id=run_id,
                    expected_pattern=pattern,
                    expected_output_path=output_path,
                )
                errors.extend(result.errors)
            if not errors:
                errors = ["no_unique_current_record_authorizes_purpose"]
            authorization_by_run[run_id] = {
                "authorized": False,
                "record_id": None,
                "errors": sorted(set(errors)),
            }

    authorized_sources = {run_id: source_dirs[run_id] for run_id in sorted(authorized_ids)}
    result = build_in_memory(
        authorized_sources,
        authorized_ids,
        eligibility_evidence,
        expected_run_metadata,
        project_root=root,
        window_size=window_size,
        cadence_seconds=cadence_seconds,
        tolerance_seconds=tolerance_seconds,
    )

    runs_by_id = {run.run_id: run for run in result.runs}
    replaced = []
    for run_id in run_ids:
        evidence = authorization_by_run.get(run_id, {
            "authorized": False,
            "record_id": None,
            "errors": ["run_not_authorized"],
        })
        run = runs_by_id.get(run_id)
        if not evidence["authorized"]:
            reason = "acceptance_not_authorized:" + ",".join(evidence["errors"])
            expected = expected_run_metadata.get(run_id)
            replaced.append(_blocked(
                run_id,
                expected if isinstance(expected, Mapping) else None,
                reason,
                output=str(source_dirs[run_id]),
                acceptance={"decision": "not_authorized", "purpose": purpose,
                            "errors": evidence["errors"]},
            ))
        else:
            # Authorized IDs are the only inputs to the builder. A missing
            # result here indicates a broken builder contract and fails closed.
            if run is None:
                replaced.append(_blocked(run_id, expected_run_metadata.get(run_id),
                    "authorized_run_missing_from_builder_result", output=str(source_dirs[run_id])))
                continue
            replaced.append(RunBuildResult(
                run_id=run_id,
                pattern=run.pattern,
                source_output_dir=run.source_output_dir,
                status=run.status,
                reasons=run.reasons,
                source_verification=dict(run.source_verification),
                acceptance_evidence={
                    "decision": "accepted",
                    "source": "acceptance_record_store",
                    "record_id": evidence["record_id"],
                    "purpose": purpose,
                },
                eligibility_evidence=dict(run.eligibility_evidence),
                input_observation_count=run.input_observation_count,
                selected_observation_count=run.selected_observation_count,
                excluded_observation_counts=dict(run.excluded_observation_counts),
                candidate_window_count=run.candidate_window_count,
                eligible_window_count=run.eligible_window_count,
                excluded_window_counts=dict(run.excluded_window_counts),
                row_decisions=run.row_decisions,
                windows=run.windows,
            ))

    return DatasetBuildResult(tuple(replaced))


def _blocked_runs(source_dirs, expected_run_metadata, purpose, reason):
    return DatasetBuildResult(tuple(
        _blocked(
            run_id,
            expected_run_metadata.get(run_id)
            if isinstance(expected_run_metadata, Mapping)
            and isinstance(expected_run_metadata.get(run_id), Mapping)
            else None,
            reason,
            output=str(source_dirs[run_id]),
            acceptance={"decision": "not_authorized", "purpose": purpose},
        )
        for run_id in sorted(source_dirs)
    ))
