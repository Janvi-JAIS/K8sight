# K8sight dataset schema v1

This directory contains a versioned run manifest, observation schema, read-only validator, and synthetic fixture. No project data is migrated or combined.

Evidence states are observed, unknown, not_collected, and not_applicable. Only observed carries a value; unavailable states use null and a reason. Metric provenance records exact query, labels, interval, step, lookback, units, scrape-interval evidence, and raw response paths/SHA-256. Historical audit findings are separate from archived responses.

Throttling states: evidenced, not_evidenced_under_observed_coverage, unknown. Evidence requires archived target CFS counters with a positive valid increment. A negative finding requires complete aligned throttled-period and total-period counters across the stated coverage, no gaps or resets, advancing total periods, and zero throttled-period increments. A missing series is unknown, never zero. The ratio of throttled periods is not lost CPU demand.

Observations retain UTC timestamps, run identity, target identity, phase (pre_start/active/cooldown/post_shutdown/unknown), CPU rate lookback/window state, memory working set, raw CFS counters, derived ratio, quality flags, and missing reasons. Active-only CPU summaries must exclude lookback windows crossing workload boundaries. Null measurements need reasons and are never replaced with zero.

Synthetic tests: python -B -m unittest discover -s dataset_schema_v1/tests -v
Read-only run validation: python -B dataset_schema_v1/validate_run.py --run-dir PATH
Existing legacy datasets and Run 2 record remain unchanged. Future model evaluation should split by run_id.

## Validator verification rules

The validator requires a run directory. Every declared raw source must resolve
inside that directory and its SHA-256 must match the file. For a throttling
finding, the CFS counter response must be a successful Prometheus range-query
matrix with exactly one series for the metric and the recorded namespace, Pod,
and container labels. Each raw counter timestamp is matched one-to-one to an
observation timestamp within 1 millisecond; the response and observations must
have the same timestamp set, and counter values must match with a relative and
absolute tolerance of 1e-12. The CSV counter fields are therefore checked
against the archived response, not merely against the manifest hash.

Both input documents are checked against their Draft 2020-12 schemas before
semantic validation. The historical_audits records use their own strict record
shape and cannot contain metric raw_sources. Unknown resource/evidence states
remain valid when their reasons are recorded.

Rolling-rate status is derived from the UTC sample time, lookback, and recorded
boundaries. A window crossing more than one boundary is classified by the
earliest crossed boundary (run start, cooldown start, then shutdown). A window
fully within active time must fit between run start and the active end
(cooldown start, or workload end when no cooldown is recorded). unknown is
valid only when a required boundary or lookback is unavailable, or the window
falls outside the defined classes. Phase checks require their relevant boundaries. Cooldown observations begin at cooldown_start_utc and may continue after Locust shutdown until collection_end_utc; post_shutdown observations must not overlap that cooldown interval.
