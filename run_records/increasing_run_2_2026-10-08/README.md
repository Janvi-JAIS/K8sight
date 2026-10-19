# Increasing Run 2 — 2026-10-08

- Run ID: `increasing_run_2_2026-10-08`
- Kubernetes context: `docker-desktop`
- Job: `locust-increasing`
- Target: namespace `default`, Pod `php-apache-5b5d776b6-llmk5`, container `hpa-example`
- Prometheus Service: `prometheus-kube-prometheus-prometheus` in `default`
- Query range: `2026-10-08T19:21:00Z` through `2026-10-08T19:31:15Z`, inclusive, at a `15s` step
- Samples: 42 CPU and 42 memory observations; timestamps align, with no missing points
- Timestamp representation: original UTC instants as ISO 8601 values ending in `Z`

## Queries

CPU (1-minute rate; raw values are CPU cores):

```promql
rate(container_cpu_usage_seconds_total{namespace="default",pod="php-apache-5b5d776b6-llmk5",container="hpa-example"}[1m])
```

Memory (container working-set bytes):

```promql
container_memory_working_set_bytes{namespace="default",pod="php-apache-5b5d776b6-llmk5",container="hpa-example"}
```

The two JSON files are the separate full Prometheus API responses, retaining their status, labels, timestamps, and raw values.

## CSV columns and units

`timestamp_utc,pod,cpu_cores,cpu_millicores,memory_bytes,memory_mib,run_id`

- `cpu_cores`: raw CPU rate value in cores.
- `cpu_millicores`: `cpu_cores * 1000`.
- `memory_bytes`: raw working-set value in bytes.
- `memory_mib`: `memory_bytes / 1048576` (MiB).
- Rows contain only the target Pod. No interpolation, smoothing, or resampling was applied.

## Summary

Statistics are computed from the 42 exported observations; CPU is reported in mCPU and memory in MiB.

| Metric | Minimum | Mean | Maximum | Count |
|---|---:|---:|---:|---:|
| CPU | 0.0563 mCPU | 215.3314 mCPU | 400.0596 mCPU | 42 |
| Memory working set | 19.7109 MiB | 20.1388 MiB | 20.5938 MiB | 42 |

CPU observations may be affected by container CPU throttling; the CPU series is a 1-minute rate of the cumulative usage counter.