import copy,csv,hashlib,json,sys,tempfile,unittest
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve();sys.path.insert(0,str(HERE.parents[1]))
import shutil,subprocess
from validate_run import COLS,validate_run,load_run

class SyntheticValidatorTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  d=HERE.parent/"fixtures"/"valid_run"
  cls.manifest=json.loads((d/"manifest.json").read_text(encoding="utf-8"))
  with (d/"observations.csv").open(encoding="utf-8",newline="") as f:cls.rows=list(csv.DictReader(f))
 def clone(self):return copy.deepcopy(self.manifest),copy.deepcopy(self.rows)
 def assertError(self,m,r,text,run_dir=None):
  errors=validate_run(m,r,run_dir or HERE.parent/"fixtures"/"valid_run");self.assertTrue(any(text in x for x in errors),errors)
 def test_schema_json_and_fixture(self):
  self.assertEqual([],validate_run(*self.clone(),HERE.parent/"fixtures"/"valid_run"))
  for name in ("run_manifest.schema.json","observation.schema.json"):json.loads((HERE.parents[1]/name).read_text(encoding="utf-8"))
  self.assertEqual(COLS,list(self.rows[0]))
 def test_invalid_timestamp(self):
  m,r=self.clone();r[0]["timestamp_utc"]="2026-01-01 00:02:00";self.assertError(m,r,"timestamp_utc:")
 def test_duplicate_observation(self):
  m,r=self.clone();r[1]["timestamp_utc"]=r[0]["timestamp_utc"];self.assertError(m,r,"duplicate observation")
 def test_missing_value_reason(self):
  m,r=self.clone();r[0]["memory_working_set_bytes"]="";r[0]["missing_reasons"]="{}";self.assertError(m,r,"missing value requires a reason")
 def test_invalid_throttling_status(self):
  m,r=self.clone();m["throttling"]["status"]="zero";self.assertError(m,r,"throttling.status:")
 def test_unsupported_negative_finding(self):
  m,r=self.clone();m["throttling"]["status"]="not_evidenced_under_observed_coverage";m["throttling"]["reason"]=None
  self.assertError(m,r,"missing archived raw source");self.assertError(m,r,"needs complete counters")
 def test_pod_identity_mismatch(self):
  m,r=self.clone();r[0]["pod"]="other";self.assertError(m,r,"inconsistent target identity")
 def test_container_identity_change(self):
  m,r=self.clone();r[1]["container"]="other";self.assertError(m,r,"inconsistent target identity container")
 def test_missing_required_manifest_field(self):
  m,r=self.clone();del m["run_id"];self.assertError(m,r,"required property")
 def test_invalid_hash_format(self):
  m,r=self.clone();m["metrics"][0]["raw_sources"]=[{"path":"raw/cpu.json","sha256":"bad","media_type":"application/json"}]
  self.assertError(m,r,"sha256:")
 def test_non_finite_measurement(self):
  m,r=self.clone();r[0]["cpu_usage_cores"]="NaN";self.assertError(m,r,"non-finite/negative")
 def test_negative_measurement(self):
  m,r=self.clone();r[0]["memory_working_set_bytes"]="-1";self.assertError(m,r,"minimum of 0")

 def throttling_run(self,root,status="evidenced",mutation=None):
  m,r=self.clone();timestamps=[datetime.fromisoformat(x["timestamp_utc"].replace("Z","+00:00")).timestamp() for x in r]
  throttled=[1,2] if status=="evidenced" else [1,1];periods=[100,110]
  for row,a,b in zip(r,throttled,periods):
   reasons=json.loads(row["missing_reasons"])
   for key,value in (("cfs_throttled_periods_total",a),("cfs_periods_total",b)):
    row[key]=str(value);reasons.pop(key,None)
   row["missing_reasons"]=json.dumps(reasons)
  m["throttling"]={"status":status,"reason":"synthetic test","coverage_start_utc":r[0]["timestamp_utc"],"coverage_end_utc":r[-1]["timestamp_utc"],"covered_sample_count":len(r),"evidence_metrics":["container_cpu_cfs_throttled_periods_total","container_cpu_cfs_periods_total"]}
  m["data_quality"].update(expected_samples=len(r),observed_samples=len(r),gaps=[],source_integrity="verified")
  specs=[
   ("container_cpu_cfs_throttled_periods_total","cfs_throttled_periods_total",throttled),
   ("container_cpu_cfs_periods_total","cfs_periods_total",periods)]
  for name,column,values in specs:
   labels={k:m["environment"][k]["value"] for k in ("namespace","pod","container")}
   rec={"name":name,"availability":"available","promql":name+"{namespace=\"default\",pod=\"target-pod\",container=\"app\"}","unit":"periods","query_start_utc":r[0]["timestamp_utc"],"query_end_utc":r[-1]["timestamp_utc"],"step_seconds":15,"lookback_seconds":None,"labels":labels,"raw_sources":[]}
   doc={"status":"success","data":{"resultType":"matrix","result":[{"metric":{"__name__":name,**labels},"values":[[ts,str(val)] for ts,val in zip(timestamps,values)]}]}}
   if mutation and mutation[0]==name:
    kind=mutation[1]
    if kind=="unrelated":doc["data"]["result"][0]["metric"]["__name__"]="container_cpu_usage_seconds_total"
    if kind=="wrong_labels":doc["data"]["result"][0]["metric"]["pod"]="companion-pod"
    if kind=="timestamp":doc["data"]["result"][0]["values"][0][0]+=1
    if kind=="value":doc["data"]["result"][0]["values"][0][1]=str(float(doc["data"]["result"][0]["values"][0][1])+5)
   rel=Path("raw")/(name+".json");target=root/rel;target.parent.mkdir(parents=True,exist_ok=True)
   payload=json.dumps(doc,separators=(",",":")).encode();target.write_bytes(payload)
   rec["raw_sources"]=[{"path":rel.as_posix(),"sha256":hashlib.sha256(payload).hexdigest(),"media_type":"application/json"}]
   m["metrics"].append(rec)
  return m,r,root
 def test_validator_requires_run_directory(self):
  m,r=self.clone()
  with self.assertRaises(TypeError):validate_run(m,r)
 def test_missing_raw_source_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);m,r=self.clone();m["metrics"][0]["raw_sources"]=[{"path":"raw/missing.json","sha256":"a"*64,"media_type":"application/json"}]
   self.assertError(m,r,"source missing/outside run directory",root)
 def test_wrong_raw_source_hash_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);raw=root/"raw.json";raw.write_text("{}",encoding="utf-8");m,r=self.clone()
   m["metrics"][0]["raw_sources"]=[{"path":"raw.json","sha256":"b"*64,"media_type":"application/json"}]
   self.assertError(m,r,"content hash mismatch",root)
 def test_escaping_raw_source_path_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);m,r=self.clone();m["metrics"][0]["raw_sources"]=[{"path":"../outside.json","sha256":"a"*64,"media_type":"application/json"}]
   self.assertError(m,r,"unsafe path",root)
 def test_valid_synthetic_prometheus_throttling_response(self):
  with tempfile.TemporaryDirectory() as temp:
   m,r,root=self.throttling_run(Path(temp),"evidenced")
   self.assertEqual([],validate_run(m,r,root))
 def test_unrelated_raw_response_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   m,r,root=self.throttling_run(Path(temp),"evidenced",("container_cpu_cfs_throttled_periods_total","unrelated"))
   self.assertError(m,r,"metric/target labels mismatch",root)
 def test_wrong_raw_target_labels_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   m,r,root=self.throttling_run(Path(temp),"evidenced",("container_cpu_cfs_throttled_periods_total","wrong_labels"))
   self.assertError(m,r,"metric/target labels mismatch",root)
 def test_raw_timestamp_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   m,r,root=self.throttling_run(Path(temp),"evidenced",("container_cpu_cfs_throttled_periods_total","timestamp"))
   self.assertError(m,r,"timestamps do not exactly align",root)
 def test_raw_counter_value_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   m,r,root=self.throttling_run(Path(temp),"evidenced",("container_cpu_cfs_throttled_periods_total","value"))
   self.assertError(m,r,"counter value mismatch",root)
 def test_valid_negative_finding_has_complete_raw_coverage(self):
  with tempfile.TemporaryDirectory() as temp:
   m,r,root=self.throttling_run(Path(temp),"not_evidenced_under_observed_coverage")
   self.assertEqual([],validate_run(m,r,root))
 def test_historical_audit_cannot_contain_raw_sources(self):
  m,r=self.clone();m["historical_audits"]=[{"source_type":"audit_report","reference":"synthetic","finding":"example","raw_responses_archived":True,"raw_sources":[]}]
  self.assertError(m,r,"manifest schema",HERE.parent/"fixtures"/"valid_run")
 def test_audit_record_cannot_be_a_metric_raw_source(self):
  m,r=self.clone();m["metrics"][0]["raw_sources"]=[{"path":"audit.json","sha256":"a"*64,"media_type":"audit_report"}]
  self.assertError(m,r,"media_type",HERE.parent/"fixtures"/"valid_run")
 def test_incorrect_sample_phase_rejected(self):
  m,r=self.clone();r[0]["sample_phase"]="pre_start";self.assertError(m,r,"pre_start boundary mismatch")
 def test_incorrect_rolling_rate_status_rejected(self):
  m,r=self.clone();r[0]["cpu_rate_window_status"]="crosses_pre_start";self.assertError(m,r,"rolling-rate status")
 def test_active_phase_rejected_after_earliest_workload_end_boundary(self):
  m,r=self.clone()
  m["boundaries"].update(
   workload_end_utc="2026-01-01T00:02:00Z",
   locust_shutdown_utc="2026-01-01T00:02:00Z",
   cooldown_start_utc="2026-01-01T00:04:00Z",
   collection_end_utc="2026-01-01T00:04:00Z")
  for i,row in enumerate(r):
   seconds=15+15*i
   row["timestamp_utc"]=f"2026-01-01T00:03:{seconds:02d}Z"
   row["sample_phase"]="active"
   row["cpu_rate_lookback_seconds"]="30"
   row["cpu_rate_window_status"]="unknown"
  self.assertError(m,r,"active boundary mismatch")
 def test_active_phase_requires_boundaries(self):
  m,r=self.clone();m["boundaries"]["run_start_utc"]=None
  self.assertError(m,r,"active phase requires run_start_utc")
 def test_crosses_pre_start_status_matches_window(self):
  m,r=self.clone();r[0]["timestamp_utc"]="2026-01-01T00:00:30Z";r[0]["cpu_rate_window_status"]="crosses_pre_start"
  self.assertEqual([],validate_run(m,r,HERE.parent/"fixtures"/"valid_run"))
 def test_crosses_cooldown_status_matches_window(self):
  m,r=self.clone();m["boundaries"].update(workload_end_utc="2026-01-01T00:02:00Z",cooldown_start_utc="2026-01-01T00:02:00Z")
  for row in r:row["sample_phase"]="cooldown";row["cpu_rate_window_status"]="crosses_cooldown"
  self.assertEqual([],validate_run(m,r,HERE.parent/"fixtures"/"valid_run"))
 def test_crosses_shutdown_status_matches_window(self):
  m,r=self.clone();m["boundaries"].update(workload_end_utc="2026-01-01T00:02:00Z",cooldown_start_utc="2026-01-01T00:02:00Z",collection_end_utc="2026-01-01T00:04:00Z")
  for i,row in enumerate(r):row["timestamp_utc"]="2026-01-01T00:03:15Z" if i==0 else "2026-01-01T00:03:30Z";row["sample_phase"]="post_shutdown";row["cpu_rate_window_status"]="crosses_shutdown"
  self.assertEqual([],validate_run(m,r,HERE.parent/"fixtures"/"valid_run"))
 def test_cooldown_continues_after_locust_shutdown(self):
  m,r=self.clone()
  m["boundaries"].update(workload_end_utc="2026-01-01T00:02:00Z",cooldown_start_utc="2026-01-01T00:02:00Z",locust_shutdown_utc="2026-01-01T00:02:00Z",collection_end_utc="2026-01-01T00:04:00Z")
  for i,row in enumerate(r):
   row["timestamp_utc"]="2026-01-01T00:03:15Z" if i==0 else "2026-01-01T00:03:30Z"
   row["sample_phase"]="cooldown"
   row["cpu_rate_window_status"]="unknown"
  self.assertEqual([],validate_run(m,r,HERE.parent/"fixtures"/"valid_run"))
 def test_cooldown_wins_tied_rate_boundary(self):
  m,r=self.clone()
  m["boundaries"].update(workload_end_utc="2026-01-01T00:02:00Z",cooldown_start_utc="2026-01-01T00:02:00Z",locust_shutdown_utc="2026-01-01T00:02:00Z",collection_end_utc="2026-01-01T00:04:00Z")
  for i,row in enumerate(r):
   row["timestamp_utc"]="2026-01-01T00:02:30Z" if i==0 else "2026-01-01T00:02:45Z"
   row["sample_phase"]="cooldown"
   row["cpu_rate_lookback_seconds"]="60"
   row["cpu_rate_window_status"]="crosses_cooldown"
  self.assertEqual([],validate_run(m,r,HERE.parent/"fixtures"/"valid_run"))
 def test_rate_after_shutdown_before_later_cooldown_is_unknown(self):
  m,r=self.clone()
  m["boundaries"].update(
   workload_end_utc="2026-01-01T00:02:00Z",
   locust_shutdown_utc="2026-01-01T00:02:00Z",
   cooldown_start_utc="2026-01-01T00:04:00Z",
   collection_end_utc="2026-01-01T00:04:00Z")
  for i,row in enumerate(r):
   row["timestamp_utc"]="2026-01-01T00:03:15Z" if i==0 else "2026-01-01T00:03:30Z"
   row["sample_phase"]="post_shutdown"
   row["cpu_rate_lookback_seconds"]="30"
   row["cpu_rate_window_status"]="unknown"
  self.assertEqual([],validate_run(m,r,HERE.parent/"fixtures"/"valid_run"))
 def test_unknown_rate_status_only_when_unclassifiable(self):
  m,r=self.clone();m["boundaries"]["collection_end_utc"]="2026-01-01T00:04:00Z"
  for i,row in enumerate(r):row["timestamp_utc"]="2026-01-01T00:03:45Z" if i==0 else "2026-01-01T00:04:00Z";row["sample_phase"]="post_shutdown";row["cpu_rate_lookback_seconds"]="30";row["cpu_rate_window_status"]="unknown"
  self.assertEqual([],validate_run(m,r,HERE.parent/"fixtures"/"valid_run"))

class FileLoadingAndCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = HERE.parent / "fixtures" / "valid_run"

    def test_valid_fixture_load_run(self):
        errors = load_run(self.fixture_dir)
        self.assertEqual([], errors)

    def test_valid_fixture_cli(self):
        res = subprocess.run(
            [sys.executable, str(HERE.parents[1] / "validate_run.py"), "--run-dir", str(self.fixture_dir)],
            capture_output=True, text=True
        )
        self.assertEqual(0, res.returncode, f"CLI failed with: {res.stderr}")
        self.assertIn("Valid run:", res.stdout)

    def test_nonexistent_run_directory(self):
        nonexistent = self.fixture_dir.parent / "nonexistent_run_dir"
        errors = load_run(nonexistent)
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("cannot read run inputs:" in e for e in errors))

    def test_nonexistent_run_directory_cli(self):
        res = subprocess.run(
            [sys.executable, str(HERE.parents[1] / "validate_run.py"), "--run-dir", "nonexistent_dir"],
            capture_output=True, text=True
        )
        self.assertEqual(1, res.returncode)
        self.assertIn("ERROR: cannot read run inputs:", res.stdout)

    def test_incorrect_csv_headers(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            shutil.copy(self.fixture_dir / "manifest.json", t / "manifest.json")
            (t / "observations.csv").write_text("col_a,col_b\n1,2\n", encoding="utf-8")
            errors = load_run(t)
            self.assertEqual(["observations.csv header mismatch"], errors)

    def test_missing_csv_columns(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            shutil.copy(self.fixture_dir / "manifest.json", t / "manifest.json")
            (t / "observations.csv").write_text(",".join(COLS[:-1]) + "\n", encoding="utf-8")
            errors = load_run(t)
            self.assertEqual(["observations.csv header mismatch"], errors)

    def test_extra_csv_columns(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            shutil.copy(self.fixture_dir / "manifest.json", t / "manifest.json")
            (t / "observations.csv").write_text(",".join(COLS + ["extra_col"]) + "\n", encoding="utf-8")
            errors = load_run(t)
            self.assertEqual(["observations.csv header mismatch"], errors)

    def test_empty_zero_byte_csv(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            shutil.copy(self.fixture_dir / "manifest.json", t / "manifest.json")
            (t / "observations.csv").write_text("", encoding="utf-8")
            errors = load_run(t)
            self.assertEqual(["observations.csv header mismatch"], errors)

    def test_header_only_csv(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            shutil.copy(self.fixture_dir / "manifest.json", t / "manifest.json")
            (t / "observations.csv").write_text(",".join(COLS) + "\n", encoding="utf-8")
            errors = load_run(t)
            self.assertTrue(len(errors) > 0)
            self.assertTrue(any("observations empty" in e for e in errors))

if __name__ == "__main__":
    unittest.main()
