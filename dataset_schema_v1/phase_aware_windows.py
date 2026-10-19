"""Select accepted active observations and form provenance-rich feature windows.

Pure in-memory helpers only: this module does not read or write files, run a
validator, accept runs, or fit models. Callers must provide both explicit
acceptance IDs and separate schema/training-eligibility metadata. Downstream
evaluation must group by the preserved run_id, never by Pod or pattern alone.
"""
from __future__ import annotations

import collections
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence


IDENTITY_FIELDS = ("namespace", "pod", "pod_uid", "container", "container_id")
FEATURE_UNITS = {"cpu_usage_cores": "cores", "memory_working_set_bytes": "bytes"}
DEFAULT_WINDOW_SIZE = 8
DEFAULT_CADENCE_SECONDS = 15.0
DEFAULT_TOLERANCE_SECONDS = 0.001


@dataclass(frozen=True)
class RowDecision:
    """Selection result for one input row; ``source_index`` is its stable input ordinal."""

    source_index: int
    row_id: str
    run_id: str | None
    included: bool
    reasons: tuple[str, ...]
    row: Mapping[str, Any]
    timestamp: datetime | None


@dataclass(frozen=True)
class SelectionResult:
    """All row decisions, including exclusions, in original input order."""

    decisions: tuple[RowDecision, ...]

    @property
    def included_count(self) -> int:
        return sum(item.included for item in self.decisions)

    @property
    def excluded_count(self) -> int:
        return sum(not item.included for item in self.decisions)

    @property
    def eligible_rows(self) -> tuple[RowDecision, ...]:
        return tuple(item for item in self.decisions if item.included)

    @property
    def exclusion_counts(self) -> dict[str, int]:
        counts: collections.Counter[str] = collections.Counter()
        for item in self.decisions:
            if not item.included:
                counts.update(item.reasons)
        return dict(sorted(counts.items()))


@dataclass(frozen=True)
class WindowDecision:
    """An included feature vector or a rejected candidate window."""

    included: bool
    reasons: tuple[str, ...]
    run_id: str | None
    pattern: str | None
    replicate: int | None
    seed: int | None
    start_timestamp_utc: str | None
    end_timestamp_utc: str | None
    source_row_ids: tuple[str, ...]
    source_indices: tuple[int, ...]
    source_timestamps_utc: tuple[str, ...]
    source_observation_count: int
    features: Mapping[str, tuple[int | float, ...]]
    feature_units: Mapping[str, str]


@dataclass(frozen=True)
class WindowResult:
    """Window decisions and deterministic count summaries."""

    decisions: tuple[WindowDecision, ...]
    run_exclusions: Mapping[str, tuple[str, ...]]

    @property
    def candidate_count(self) -> int:
        return len(self.decisions)

    @property
    def eligible_windows(self) -> tuple[WindowDecision, ...]:
        return tuple(item for item in self.decisions if item.included)

    @property
    def exclusion_counts(self) -> dict[str, int]:
        counts: collections.Counter[str] = collections.Counter()
        for item in self.decisions:
            if not item.included:
                counts.update(item.reasons)
        for reasons in self.run_exclusions.values():
            counts.update(reasons)
        return dict(sorted(counts.items()))


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None
    if result.utcoffset() != timezone.utc.utcoffset(result):
        return None
    return result.astimezone(timezone.utc)


def _parse_json_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value or "[]")
        if isinstance(parsed, list):
            return parsed
    raise ValueError("quality_flags must be a list or JSON list")


def _parse_json_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        parsed = json.loads(value or "{}")
        if isinstance(parsed, Mapping):
            return parsed
    raise ValueError("missing_reasons must be a mapping or JSON object")


def _format_timestamp(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value else None


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _memory_bytes(value: Any) -> int | None:
    """Parse whole-byte values without converting large integers through float."""
    if isinstance(value, bool) or value is None or value == "":
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    if isinstance(value, float) and math.isfinite(value) and value >= 0 and value.is_integer():
        return int(value)
    return None


def _metadata_for(run_metadata: Mapping[str, Mapping[str, Any]], run_id: str | None):
    if run_id is None:
        return None
    return run_metadata.get(run_id)


def select_eligible_observations(
    observations: Iterable[Mapping[str, Any]],
    run_metadata: Mapping[str, Mapping[str, Any]],
    accepted_run_ids: Iterable[str] | None,
) -> SelectionResult:
    """Select active CPU/memory rows using explicit acceptance and metadata.

    ``run_metadata`` is keyed by actual run ID. Each entry must explicitly
    supply ``schema_validity='pass'``, ``training_eligible=True``, ``pattern``,
    positive integer ``replicate``, integer ``seed``, and ``observation_units``
    with CPU in cores and memory in bytes. ``accepted_run_ids`` is an
    independent explicit approval set; ``None`` means no acceptance evidence.
    The function never infers run identity or acceptance.
    """
    if isinstance(accepted_run_ids, str):
        accepted = None
        acceptance_input_invalid = True
    else:
        try:
            accepted = None if accepted_run_ids is None else frozenset(accepted_run_ids)
            acceptance_input_invalid = accepted is not None and any(not isinstance(item, str) or not item for item in accepted)
        except TypeError:
            accepted = None
            acceptance_input_invalid = True
    copied = [dict(row) for row in observations]
    parsed_times = [_timestamp(row.get("timestamp_utc")) for row in copied]
    timestamp_counts: collections.Counter[tuple[str, datetime]] = collections.Counter(
        (row["run_id"], parsed_times[index])
        for index, row in enumerate(copied)
        if isinstance(row.get("run_id"), str) and row.get("run_id") and parsed_times[index] is not None
    )

    decisions = []
    for index, row in enumerate(copied):
        run_id = row.get("run_id") if isinstance(row.get("run_id"), str) and row.get("run_id") else None
        reasons: list[str] = []
        metadata = _metadata_for(run_metadata, run_id)
        if run_id is None:
            reasons.append("run_id_missing")
        elif metadata is None:
            reasons.append("run_metadata_missing")
        if acceptance_input_invalid:
            reasons.append("acceptance_set_invalid")
        elif accepted is None:
            reasons.append("acceptance_set_missing")
        elif run_id is not None and run_id not in accepted:
            reasons.append("run_not_explicitly_accepted")
        if metadata is not None:
            if metadata.get("schema_validity") != "pass":
                reasons.append("schema_validity_not_pass")
            if metadata.get("training_eligible") is False:
                reasons.append("training_ineligible")
            elif metadata.get("training_eligible") is not True:
                reasons.append("training_eligibility_unknown")
            if not isinstance(metadata.get("pattern"), str) or not metadata.get("pattern"):
                reasons.append("run_pattern_missing")
            elif row.get("pattern") != metadata["pattern"]:
                reasons.append("row_pattern_mismatch")
            replicate = metadata.get("replicate")
            if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 1:
                reasons.append("replicate_missing_or_invalid")
            seed = metadata.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                reasons.append("seed_missing_or_invalid")
            units = metadata.get("observation_units")
            if not isinstance(units, Mapping) or any(units.get(key) != unit for key, unit in FEATURE_UNITS.items()):
                reasons.append("feature_units_missing_or_invalid")

        stamp = parsed_times[index]
        if stamp is None:
            reasons.append("timestamp_missing_or_invalid")
        if run_id is not None and stamp is not None and timestamp_counts[(run_id, stamp)] > 1:
            reasons.append("duplicate_timestamp")
        if row.get("sample_phase") != "active":
            reasons.append("sample_phase_not_active")
        if row.get("cpu_rate_window_status") != "fully_within_active":
            reasons.append("cpu_rate_window_not_fully_active")
        for field in IDENTITY_FIELDS:
            if not isinstance(row.get(field), str) or not row.get(field).strip():
                reasons.append("target_identity_missing")
                break

        for column, code in (("cpu_usage_cores", "cpu_missing_or_invalid"),
                             ("memory_working_set_bytes", "memory_missing_or_invalid")):
            value = _number(row.get(column)) if column == "cpu_usage_cores" else _memory_bytes(row.get(column))
            if value is None:
                reasons.append(code)

        try:
            flags = _parse_json_list(row.get("quality_flags"))
        except (TypeError, ValueError, json.JSONDecodeError):
            reasons.append("quality_flags_invalid")
        else:
            if flags:
                reasons.append("quality_flags_present")

        try:
            _parse_json_mapping(row.get("missing_reasons"))
        except (TypeError, ValueError, json.JSONDecodeError):
            reasons.append("missing_reasons_invalid")

        row_id = f"{run_id or 'unknown'}:{row.get('timestamp_utc') or 'unknown'}:{index}"
        decisions.append(RowDecision(
            source_index=index,
            row_id=row_id,
            run_id=run_id,
            included=not reasons,
            reasons=tuple(dict.fromkeys(reasons)),
            row=row,
            timestamp=stamp,
        ))
    return SelectionResult(tuple(decisions))


def _excluded_window(
    reason: str | Iterable[str],
    run_id: str | None,
    source: Sequence[RowDecision] = (),
    metadata: Mapping[str, Any] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> WindowDecision:
    source_times = tuple(item.row.get("timestamp_utc", "") for item in source)
    reasons = (reason,) if isinstance(reason, str) else tuple(sorted(set(reason)))
    return WindowDecision(
        included=False,
        reasons=reasons,
        run_id=run_id,
        pattern=(metadata or {}).get("pattern"),
        replicate=(metadata or {}).get("replicate"),
        seed=(metadata or {}).get("seed"),
        start_timestamp_utc=_format_timestamp(start),
        end_timestamp_utc=_format_timestamp(end),
        source_row_ids=tuple(item.row_id for item in source),
        source_indices=tuple(item.source_index for item in source),
        source_timestamps_utc=source_times,
        source_observation_count=len(source),
        features={},
        feature_units=dict(FEATURE_UNITS),
    )


def build_feature_windows(
    selection: SelectionResult,
    run_metadata: Mapping[str, Mapping[str, Any]],
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    cadence_seconds: float = DEFAULT_CADENCE_SECONDS,
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
) -> WindowResult:
    """Build sliding raw CPU/memory feature vectors without crossing boundaries.

    Rows are ordered by parsed UTC timestamp within actual run ID. Every
    timestamped input row remains a timeline barrier, including excluded rows,
    so a filtered row cannot be silently skipped to fill a window. A run with
    an unplaceable timestamp is excluded from window construction. ``run_id``
    remains the grouping key for future train/test splitting.
    """
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 2:
        raise ValueError("window_size must be an integer of at least 2")
    if isinstance(cadence_seconds, bool) or not isinstance(cadence_seconds, (int, float)) or not math.isfinite(cadence_seconds) or cadence_seconds <= 0:
        raise ValueError("cadence_seconds must be a positive finite number")
    if isinstance(tolerance_seconds, bool) or not isinstance(tolerance_seconds, (int, float)) or not math.isfinite(tolerance_seconds) or tolerance_seconds < 0 or tolerance_seconds >= cadence_seconds:
        raise ValueError("tolerance_seconds must be finite, nonnegative, and less than cadence_seconds")

    if any(item.run_id is None for item in selection.decisions):
        return WindowResult((), {"*": ("run_id_missing",)})

    grouped: dict[str, list[RowDecision]] = collections.defaultdict(list)
    for item in selection.decisions:
        if item.run_id is not None:
            grouped[item.run_id].append(item)

    result: list[WindowDecision] = []
    run_exclusions: dict[str, tuple[str, ...]] = {}
    for run_id, items in sorted(grouped.items()):
        metadata = run_metadata.get(run_id)
        if metadata is None:
            run_exclusions[run_id] = ("run_metadata_missing",)
            continue
        if any(item.timestamp is None for item in items):
            run_exclusions[run_id] = ("run_contains_unplaceable_timestamp",)
            continue
        ordered = sorted(items, key=lambda item: (item.timestamp, item.source_index))
        cadence = float(cadence_seconds)
        tolerance = float(tolerance_seconds)
        for first in ordered:
            start = first.timestamp
            assert start is not None
            end = start + timedelta(seconds=(window_size - 1) * cadence)
            within = [item for item in ordered if start - timedelta(seconds=tolerance) <= item.timestamp <= end + timedelta(seconds=tolerance)]
            if any("duplicate_timestamp" in item.reasons for item in within):
                result.append(_excluded_window("duplicate_timestamp", run_id, within, metadata, start, end))
                continue
            expected = [start + timedelta(seconds=i * cadence) for i in range(window_size)]
            matched: list[RowDecision] = []
            unmatched = list(within)
            cadence_gap = False
            duplicate_or_extra = False
            for target in expected:
                matches = [item for item in unmatched if abs((item.timestamp - target).total_seconds()) <= tolerance]
                if len(matches) != 1:
                    cadence_gap = True
                    continue
                matched.append(matches[0])
                unmatched.remove(matches[0])
            if unmatched:
                duplicate_or_extra = True
            if cadence_gap:
                remaining_rows = sum(item.timestamp >= start for item in ordered)
                reason = "insufficient_window_rows" if remaining_rows < window_size else "cadence_gap"
                result.append(_excluded_window(reason, run_id, within, metadata, start, end))
                continue
            if duplicate_or_extra:
                result.append(_excluded_window("unexpected_or_duplicate_timestamp", run_id, within, metadata, start, end))
                continue
            if any(not item.included for item in matched):
                reasons = {reason for item in matched if not item.included for reason in item.reasons}
                reasons.add("candidate_contains_excluded_row")
                result.append(_excluded_window(reasons, run_id, matched, metadata, start, end))
                continue

            row_reasons: set[str] = set()
            patterns = {item.row.get("pattern") for item in matched}
            phases = {item.row.get("sample_phase") for item in matched}
            identities = {tuple(item.row.get(key) for key in IDENTITY_FIELDS) for item in matched}
            if len(patterns) != 1 or next(iter(patterns)) != metadata.get("pattern"):
                row_reasons.add("pattern_changed_or_metadata_mismatch")
            if phases != {"active"}:
                row_reasons.add("phase_changed")
            if len(identities) != 1 or any(any(value is None or value == "" for value in identity) for identity in identities):
                row_reasons.add("container_identity_changed_or_missing")
            if any(item.row.get("cpu_rate_window_status") != "fully_within_active" for item in matched):
                row_reasons.add("cpu_rate_window_not_fully_active")
            if any(_number(item.row.get("cpu_usage_cores")) is None for item in matched):
                row_reasons.add("cpu_missing_or_invalid")
            if any(_memory_bytes(item.row.get("memory_working_set_bytes")) is None for item in matched):
                row_reasons.add("memory_missing_or_invalid")
            if row_reasons:
                result.append(_excluded_window(row_reasons, run_id, matched, metadata, start, end))
                continue

            cpu = tuple(float(_number(item.row["cpu_usage_cores"])) for item in matched)
            memory = tuple(_memory_bytes(item.row["memory_working_set_bytes"]) for item in matched)
            result.append(WindowDecision(
                included=True,
                reasons=(),
                run_id=run_id,
                pattern=metadata["pattern"],
                replicate=metadata["replicate"],
                seed=metadata["seed"],
                start_timestamp_utc=_format_timestamp(start),
                end_timestamp_utc=_format_timestamp(end),
                source_row_ids=tuple(item.row_id for item in matched),
                source_indices=tuple(item.source_index for item in matched),
                source_timestamps_utc=tuple(item.row["timestamp_utc"] for item in matched),
                source_observation_count=len(matched),
                features={"cpu_usage_cores": cpu, "memory_working_set_bytes": memory},
                feature_units=dict(FEATURE_UNITS),
            ))
    return WindowResult(tuple(result), run_exclusions)
