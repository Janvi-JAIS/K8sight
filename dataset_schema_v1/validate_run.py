#!/usr/bin/env python3
"""Read-only K8sight run validator."""
import argparse,csv,datetime as dt,hashlib,json,math,re
from pathlib import Path,PurePosixPath
from jsonschema import Draft202012Validator,FormatChecker
PAT={"stable","increasing","periodic","spike","random_spike"}
PH={"pre_start","active","cooldown","post_shutdown","unknown"}
RATE={"fully_within_active","crosses_pre_start","crosses_cooldown","crosses_shutdown","unknown","not_applicable"}
EV={"observed","unknown","not_collected","not_applicable"}
TH={"evidenced","not_evidenced_under_observed_coverage","unknown"}
TR=re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
HR=re.compile(r"^[0-9a-fA-F]{64}$")
COLUMNS=COLS="timestamp_utc run_id pattern namespace pod pod_uid container container_id sample_phase cpu_usage_cores cpu_rate_lookback_seconds cpu_rate_window_status memory_working_set_bytes cfs_throttled_periods_total cfs_periods_total cfs_throttled_seconds_total throttled_period_ratio_1m quality_flags missing_reasons".split()
MEAS={"cpu_usage_cores":"cores","memory_working_set_bytes":"bytes","cfs_throttled_periods_total":"periods","cfs_periods_total":"periods","cfs_throttled_seconds_total":"seconds","throttled_period_ratio_1m":"ratio"}
CTR=("cfs_throttled_periods_total","cfs_periods_total","cfs_throttled_seconds_total")
IDS=("namespace","pod","pod_uid","container","container_id")
MU={"container_cpu_usage_seconds_total":"cores","container_memory_working_set_bytes":"bytes","container_cpu_cfs_throttled_periods_total":"periods","container_cpu_cfs_periods_total":"periods","container_cpu_cfs_throttled_seconds_total":"seconds"}
def stamp(v):
 if not isinstance(v,str) or not TR.fullmatch(v):return None
 try:return dt.datetime.fromisoformat(v[:-1]+"+00:00")
 except ValueError:return None
def number(v):
 if v is None or v=="":return None
 try:return float(v)
 except (ValueError,TypeError):return math.nan
def known(m,g,k):
 x=m.get(g,{}).get(k);return x.get("value") if isinstance(x,dict) and x.get("status")=="observed" else None
def evidence(x,n):
 e=[]
 if not isinstance(x,dict):return [n+": evidence object missing"]
 if x.get("status") not in EV:return [n+".status invalid"]
 if x["status"]=="observed" and x.get("value") is None:e.append(n+": observed value required")
 if x["status"]!="observed" and x.get("value") is not None:e.append(n+": unavailable value must be null")
 if x["status"]!="observed" and not x.get("reason"):e.append(n+": unavailable reason required")
 if x.get("observed_at_utc") is not None and stamp(x["observed_at_utc"]) is None:e.append(n+": invalid observed_at_utc")
 return e
def schema_errors(schema_name,instance,label):
 schema_path=Path(__file__).resolve().parent/schema_name
 schema=json.loads(schema_path.read_text(encoding="utf-8"))
 Draft202012Validator.check_schema(schema)
 validator=Draft202012Validator(schema,format_checker=FormatChecker())
 return [f"{label} schema: {'.'.join(map(str,x.absolute_path)) or '<root>'}: {x.message}" for x in validator.iter_errors(instance)]
def observation_for_schema(row):
 out=dict(row)
 for k in IDS:
  if out.get(k)=="":out[k]=None
 for k in MEAS:
  out[k]=number(out.get(k))
 out["cpu_rate_lookback_seconds"]=number(out.get("cpu_rate_lookback_seconds"))
 for k,default in (("quality_flags",[]),("missing_reasons",{})):
  out[k]=json.loads(out.get(k) or json.dumps(default))
 return out
def expected_rate_state(time,look,bd):
 if time is None or look is None:return "unknown"
 window_start=time-dt.timedelta(seconds=look)
 crossed=[]
 for name,key in (("crosses_pre_start","run_start_utc"),("crosses_cooldown","cooldown_start_utc"),("crosses_shutdown","locust_shutdown_utc")):
  boundary=bd.get(key)
  if boundary and window_start<boundary<=time:crossed.append((boundary,name))
 if crossed:return min(crossed,key=lambda x:x[0])[1]
 start=bd.get("run_start_utc")
 active_ends=[bd[k] for k in ("workload_end_utc","cooldown_start_utc","locust_shutdown_utc") if bd.get(k) is not None]
 active_end=min(active_ends) if active_ends else None
 if start and active_end and window_start>=start and time<=active_end:return "fully_within_active"
 return "unknown"
def verify_prometheus_counter(metric_name,record,documents,rows,parsed,identity):
 errors=[]
 if not documents:return ["throttling missing archived raw source "+metric_name]
 if record.get("availability")!="available":errors.append("throttling counter is not available: "+metric_name)
 if any(x.get("media_type")!="application/json" for x in record.get("raw_sources",[])):errors.append("throttling source must be application/json: "+metric_name)
 expected_labels={k:identity.get(k) for k in ("namespace","pod","container")}
 if any(not v for v in expected_labels.values()):return errors+["cannot verify target labels for "+metric_name]
 if any(record.get("labels",{}).get(k)!=v for k,v in expected_labels.items()):errors.append("manifest selector labels mismatch for "+metric_name)
 samples={}
 column={"container_cpu_cfs_throttled_periods_total":"cfs_throttled_periods_total","container_cpu_cfs_periods_total":"cfs_periods_total","container_cpu_cfs_throttled_seconds_total":"cfs_throttled_seconds_total"}[metric_name]
 for doc in documents:
  if not isinstance(doc,dict) or doc.get("status")!="success":errors.append("raw Prometheus response status invalid for "+metric_name);continue
  data=doc.get("data")
  if not isinstance(data,dict) or data.get("resultType")!="matrix" or not isinstance(data.get("result"),list) or len(data["result"])!=1:
   errors.append("raw Prometheus response must contain one matrix series for "+metric_name);continue
  series=data["result"][0];labels=series.get("metric",{})
  expected={"__name__":metric_name,**expected_labels}
  if any(labels.get(k)!=v for k,v in expected.items()):errors.append("raw Prometheus metric/target labels mismatch for "+metric_name)
  for pair in series.get("values",[]):
   if not isinstance(pair,list) or len(pair)!=2:errors.append("raw Prometheus sample malformed for "+metric_name);continue
   try:
    at=dt.datetime.fromtimestamp(float(pair[0]),tz=dt.timezone.utc);value=float(pair[1])
   except (ValueError,TypeError,OverflowError,OSError):errors.append("raw Prometheus sample non-numeric for "+metric_name);continue
   if not math.isfinite(value) or value<0:errors.append("raw Prometheus counter invalid for "+metric_name)
   if at in samples:errors.append("duplicate raw Prometheus timestamp for "+metric_name)
   samples[at]=value
 targets={t for t,_ in parsed if t is not None};matched={}
 for at,value in samples.items():
  closest=min(targets,key=lambda target:abs((target-at).total_seconds())) if targets else None
  if closest is None or abs((closest-at).total_seconds())>0.001:
   errors.append("raw Prometheus timestamp is not aligned within 1 ms for "+metric_name);continue
  if closest in matched:errors.append("multiple raw Prometheus samples align to one observation for "+metric_name)
  matched[closest]=value
 if set(matched)!=targets or len(samples)!=len(targets):errors.append("raw Prometheus timestamps do not exactly align with observations for "+metric_name)
 for at,values in parsed:
  if at is None or at not in matched:continue
  observed=values.get(column)
  if observed is None or not math.isclose(observed,matched[at],rel_tol=1e-12,abs_tol=1e-12):errors.append("raw Prometheus counter value mismatch for "+metric_name)
 return errors
def validate_run(m,rows,run_dir):
 """Return errors without mutating any input or file."""
 e=[]
 try:
  root=Path(run_dir).resolve(strict=True)
  if not root.is_dir():return ["run directory is missing or is not a directory"]
 except (OSError,TypeError,ValueError):return ["run directory is missing or is not a directory"]
 if not isinstance(m,dict):return ["manifest must be an object"]
 if not isinstance(rows,list):return ["observations must be an array"]
 try:
  e+=schema_errors("run_manifest.schema.json",m,"manifest")
  for i,row in enumerate(rows,2):e+=schema_errors("observation.schema.json",observation_for_schema(row),f"row {i}")
 except (OSError,json.JSONDecodeError,ValueError,TypeError) as exc:return [f"schema validation failed: {exc}"]
 if e:return e
 required={"schema_version","run_id","pattern","replicate","timezone","run_outcome","workload","environment","boundaries","observation_units","metrics","throttling","data_quality","historical_audits"}
 for k in sorted(required-set(m)):e.append("manifest missing required field "+k)
 if m.get("schema_version")!="1.0.0":e.append("schema_version invalid")
 if not isinstance(m.get("run_id"),str) or not m.get("run_id"):e.append("run_id invalid")
 if m.get("pattern") not in PAT:e.append("pattern invalid")
 if m.get("timezone")!="UTC":e.append("timezone invalid")
 if m.get("run_outcome") not in {"completed","partial","failed","stopped","unknown"}:e.append("run_outcome invalid")
 e+=evidence(m.get("replicate"),"replicate")
 w=m.get("workload",{})
 for k in ("generator","generator_version","generator_sha256","random_seed"):e+=evidence(w.get(k),"workload."+k)
 env=m.get("environment",{})
 for k in ("kubernetes_context","namespace","job","pod","pod_uid","container","container_id","image","node"):e+=evidence(env.get(k),"environment."+k)
 for k in ("cpu_request","memory_request","cpu_limit","memory_limit"):e+=evidence(env.get("resources",{}).get(k),"resources."+k)
 for k,u in MEAS.items():
  if m.get("observation_units",{}).get(k)!=u:e.append("unit mismatch "+k)
 b=m.get("boundaries",{});bd={}
 for k in ("run_start_utc","workload_end_utc","cooldown_start_utc","locust_shutdown_utc","collection_start_utc","collection_end_utc"):
  if b.get(k) is not None:
   bd[k]=stamp(b[k])
   if bd[k] is None:e.append("invalid boundary "+k)
 # Validate the workload timeline separately from cooldown collection.
 for earlier,later in (
  ("run_start_utc","workload_end_utc"),
  ("workload_end_utc","cooldown_start_utc"),
  ("run_start_utc","locust_shutdown_utc"),
  ("locust_shutdown_utc","collection_end_utc"),
 ):
  if bd.get(earlier) and bd.get(later) and bd[earlier]>bd[later]:
   e.append("boundary ordering invalid at "+later)
 if bd.get("collection_start_utc") and bd.get("collection_end_utc") and bd["collection_start_utc"]>bd["collection_end_utc"]:e.append("collection interval inverted")
 source_documents={};metrics={}
 for i,x in enumerate(m.get("metrics",[])):
  if not isinstance(x,dict) or not x.get("name"):e.append(f"metrics[{i}] invalid");continue
  n=x["name"]
  if n in metrics:e.append("duplicate metric "+n)
  metrics[n]=x
  if x.get("availability") not in {"available","empty","unknown","not_collected","not_applicable"}:e.append(f"metrics[{i}] availability invalid")
  if n in MU and x.get("unit")!=MU[n]:e.append(f"metrics[{i}] unit invalid for {n}")
  for k in ("query_start_utc","query_end_utc"):
   if x.get(k) is not None and stamp(x[k]) is None:e.append(f"metrics[{i}] invalid {k}")
  start,end=stamp(x.get("query_start_utc")),stamp(x.get("query_end_utc"))
  if start and end and start>end:e.append(f"metrics[{i}] inverted query interval")
  for k,pos in (("step_seconds",True),("lookback_seconds",False)):
   v=x.get(k)
   if v is not None and (not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 or (pos and v==0)):e.append(f"metrics[{i}] invalid {k}")
  for k in ("namespace","pod","container"):
   v=known(m,"environment",k)
   if v is not None and x.get("labels",{}).get(k)!=v:e.append(f"metrics[{i}] label {k} mismatch")
  sources=x.get("raw_sources",[])
  if not isinstance(sources,list):e.append(f"metrics[{i}] raw_sources must be array");continue
  for j,s in enumerate(sources):
   label=f"metrics[{i}].raw_sources[{j}]"
   if not isinstance(s,dict) or not HR.fullmatch(str(s.get("sha256",""))):e.append(label+" invalid SHA-256");continue
   path=PurePosixPath(str(s.get("path","")))
   if path.is_absolute() or ".." in path.parts or not path.parts:e.append(label+" unsafe path");continue
   try:
    f=root.joinpath(*path.parts).resolve(strict=True);f.relative_to(root)
    content=f.read_bytes()
    if hashlib.sha256(content).hexdigest().lower()!=s["sha256"].lower():e.append(label+" content hash mismatch")
    if s.get("media_type")=="application/json":
     try:source_documents.setdefault(n,[]).append(json.loads(content.decode("utf-8")))
     except (UnicodeDecodeError,json.JSONDecodeError):e.append(label+" invalid JSON source")
   except (OSError,ValueError):e.append(label+" source missing/outside run directory")
 t=m.get("throttling",{});status=t.get("status")
 if status not in TH:e.append("throttling.status invalid")
 if status=="unknown" and not t.get("reason"):e.append("unknown throttling requires reason")
 if status in {"evidenced","not_evidenced_under_observed_coverage"}:
  needed=["container_cpu_cfs_throttled_periods_total"]
  if status.startswith("not_evidenced"):needed.append("container_cpu_cfs_periods_total")
  for n in needed:
   if n not in metrics or not metrics[n].get("raw_sources"):e.append("throttling missing archived raw source "+n)
 q=m.get("data_quality",{})
 if q.get("observed_samples")!=len(rows):e.append("observed sample count mismatch")
 if q.get("source_integrity")=="failed":e.append("source integrity failed")
 seen=set();parsed=[];idvals={k:set() for k in IDS}
 if not rows:e.append("observations empty")
 for i,r in enumerate(rows,2):
  p=f"row {i}"
  if set(r)!=set(COLS):e.append(p+" columns mismatch")
  if r.get("run_id")!=m.get("run_id"):e.append(p+" inconsistent run_id")
  if r.get("pattern")!=m.get("pattern"):e.append(p+" inconsistent pattern")
  time=stamp(r.get("timestamp_utc"))
  if time is None:e.append(p+" invalid ISO 8601 UTC timestamp")
  identity=tuple(r.get(k) or None for k in IDS)
  for k in IDS:
   if r.get(k):idvals[k].add(r[k])
   v=known(m,"environment",k)
   if v is not None and r.get(k)!=v:e.append(p+" inconsistent target identity "+k)
  if r.get("sample_phase") not in PH:e.append(p+" invalid phase")
  key=(time,identity)
  if time is not None and key in seen:e.append(p+" duplicate observation")
  seen.add(key)
  try:
   missing=json.loads(r.get("missing_reasons") or "{}");flags=json.loads(r.get("quality_flags") or "[]")
   if not isinstance(missing,dict) or not isinstance(flags,list):raise ValueError()
  except (ValueError,TypeError,json.JSONDecodeError):missing={};e.append(p+" invalid quality JSON")
  values={}
  for k in MEAS:
   v=number(r.get(k));values[k]=v
   if v is not None and (not math.isfinite(v) or v<0):e.append(p+" non-finite/negative "+k)
   if v is None and not (isinstance(missing.get(k),str) and missing[k].strip()):e.append(p+" missing value requires a reason "+k)
   if v is not None and k in missing:e.append(p+" reason on populated value "+k)
  ratio=values["throttled_period_ratio_1m"]
  if ratio is not None and ratio>1:e.append(p+" throttling ratio exceeds one")
  state=r.get("cpu_rate_window_status")
  if values["cpu_usage_cores"] is None and state!="not_applicable":e.append(p+" missing CPU needs not_applicable rate window")
  if values["cpu_usage_cores"] is not None and state not in RATE-{"not_applicable"}:e.append(p+" invalid CPU rate window status")
  look=number(r.get("cpu_rate_lookback_seconds"))
  if look is not None and (not math.isfinite(look) or look<0):e.append(p+" invalid lookback")
  if values["cpu_usage_cores"] is not None:
   expected=expected_rate_state(time,look,bd)
   if state!=expected:e.append(p+f" rolling-rate status {state!r} inconsistent with boundaries (expected {expected!r})")
  if time:
   phase=r.get("sample_phase");begin=bd.get("run_start_utc");shutdown=bd.get("locust_shutdown_utc");cool=bd.get("cooldown_start_utc")
   active_ends=[bd[k] for k in ("workload_end_utc","locust_shutdown_utc","cooldown_start_utc") if bd.get(k) is not None]
   active_end=min(active_ends) if active_ends else None
   if phase=="pre_start":
    if begin is None:e.append(p+" pre_start phase requires run_start_utc")
    elif time>=begin:e.append(p+" pre_start boundary mismatch")
   if phase=="active":
    if begin is None or active_end is None:e.append(p+" active phase requires run_start_utc and active end boundary")
    elif time<begin or time>=active_end:e.append(p+" active boundary mismatch")
   if phase=="cooldown":
    if cool is None:e.append(p+" cooldown phase requires cooldown_start_utc")
    elif time<cool:e.append(p+" cooldown boundary mismatch")
    elif bd.get("collection_end_utc") and time>bd["collection_end_utc"]:e.append(p+" cooldown boundary mismatch")
   if phase=="post_shutdown":
    if shutdown is None:e.append(p+" post_shutdown phase requires locust_shutdown_utc")
    elif time<=shutdown:e.append(p+" post_shutdown boundary mismatch")
   if bd.get("collection_start_utc") and time<bd["collection_start_utc"]:e.append(p+" before collection interval")
   if bd.get("collection_end_utc") and time>bd["collection_end_utc"]:e.append(p+" after collection interval")
  parsed.append((time,values))
 for k,vals in idvals.items():
  if len(vals)>1:e.append("target identity changes: "+k)
 parsed.sort(key=lambda x:x[0] or dt.datetime.min.replace(tzinfo=dt.timezone.utc))
 deltas={k:[] for k in CTR};reset=False
 for k in CTR:
  old=None
  for _,vals in parsed:
   cur=vals[k]
   if cur is not None and old is not None:
    delta=cur-old
    if delta<0:reset=True
    else:deltas[k].append(delta)
   old=cur
 if status in {"evidenced","not_evidenced_under_observed_coverage"}:
  identity={}
  for key in ("namespace","pod","container"):
   vals=[known(m,"environment",key)]
   vals.extend(idvals[key])
   identity[key]=next((v for v in vals if v),None)
  raw_needed=["container_cpu_cfs_throttled_periods_total"]
  if status=="not_evidenced_under_observed_coverage":raw_needed.append("container_cpu_cfs_periods_total")
  for name in raw_needed:
   if name not in metrics:e.append("throttling missing metric record "+name)
   else:e+=verify_prometheus_counter(name,metrics[name],source_documents.get(name,[]),rows,parsed,identity)
 if status=="evidenced" and (reset or not any(d>0 for d in deltas[CTR[0]])):e.append("evidenced needs positive throttled-period increment without reset")
 if status=="not_evidenced_under_observed_coverage":
  pair=(CTR[0],CTR[1])
  if reset or any(vals[k] is None for _,vals in parsed for k in pair):e.append("negative throttling finding needs complete counters without resets")
  if any(d>0 for d in deltas[CTR[0]]):e.append("negative throttling finding contradicts positive increments")
  if not any(d>0 for d in deltas[CTR[1]]):e.append("negative throttling finding needs advancing total periods")
  if q.get("gaps") or q.get("expected_samples")!=len(rows):e.append("negative throttling finding needs gap-free complete coverage")
  if t.get("covered_sample_count")!=len(rows):e.append("negative throttling coverage count mismatch")
  times=[x for x,_ in parsed if x is not None]
  if times and (stamp(t.get("coverage_start_utc"))!=times[0] or stamp(t.get("coverage_end_utc"))!=times[-1]):e.append("negative throttling coverage bounds mismatch")
  metric_names={CTR[0]:"container_cpu_cfs_throttled_periods_total",CTR[1]:"container_cpu_cfs_periods_total"};steps=[]
  for counter,name in metric_names.items():
   record=metrics.get(name);step=record.get("step_seconds") if record else None
   if not record or record.get("availability")!="available" or step is None:e.append("negative throttling finding requires available counters and query step: "+name)
   else:
    steps.append(step);qstart=stamp(record.get("query_start_utc"));qend=stamp(record.get("query_end_utc"))
    if times and (qstart is None or qend is None or qstart>times[0] or qend<times[-1]):e.append("negative throttling metric range does not cover observations: "+name)
    if any(abs((b-a).total_seconds()-step)>0.001 for a,b in zip(times,times[1:])):e.append("negative throttling samples do not match query step: "+name)
   if len(steps)==2 and steps[0]!=steps[1]:e.append("negative throttling counter query steps are misaligned")
 return e

def load_run(directory):
 d=Path(directory)
 try:
  manifest=json.loads((d/"manifest.json").read_text(encoding="utf-8"))
  with (d/"observations.csv").open(encoding="utf-8-sig",newline="") as stream:
   reader=csv.DictReader(stream)
   if reader.fieldnames!=COLS:return ["observations.csv header mismatch"]
   rows=list(reader)
 except (OSError,json.JSONDecodeError,csv.Error) as exc:return [f"cannot read run inputs: {exc}"]
 return validate_run(manifest,rows,d)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run-dir",required=True,type=Path);args=parser.parse_args()
 errors=load_run(args.run_dir)
 if errors:
  for error in errors:print("ERROR:",error)
  return 1
 print("Valid run:",args.run_dir);return 0
if __name__=="__main__":raise SystemExit(main())
