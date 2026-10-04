#!/usr/bin/env python3
"""Completes a CmdStan run whose sampling finished but whose post-processing
crashed (the first seed-1 runs looked for CSV columns named beta[1] instead
of CmdStan's beta.1). Sampling had completed and CmdStan's CSV files were
intact in .tmp/NAME.

It post-processes those CSV files with the fixed code (run_cmdstan.postprocess)
and completes results/runs/NAME.json:
- compile and sampling seconds from cmdstanpy's own log lines in the run's
  log (one-second resolution): "compiling" to "compiled model", and
  "CmdStan start processing" to the last "Chain [k] done processing";
- the post-processing is timed here, and its wall and CPU seconds are added
  to the measured run's total wall and CPU seconds; its peak RSS is compared
  with the measured peak (the larger is kept);
- the record is marked "recovered" with these details.

usage: .venv/bin/python bench/shootout/recover_cmdstan.py NAME [--reduce-sum]
"""
import datetime
import json
import os
import re
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import run_cmdstan  # noqa: E402

name = sys.argv[1]
reduce_sum = "--reduce-sum" in sys.argv
rec_path = os.path.join(common.RESULTS, "runs", name + ".json")
rec = json.load(open(rec_path))
assert rec["exit_status"] != 0 and not rec.get("recovered"), "only for a run whose post-processing failed"
log = open(os.path.join(common.BUILD, "logs", name + ".log")).read()


def ts(pattern, last=False):
    hits = re.findall(r"^(\d\d:\d\d:\d\d) - cmdstanpy - INFO - " + pattern, log, flags=re.M)
    t = hits[-1] if last else hits[0]
    return datetime.datetime.strptime(t, "%H:%M:%S")


compile_s = (ts("compiled model") - ts("compiling stan file")).total_seconds()
sampling_s = (ts(r"(?:Chain \[\d\]|CmdStan) done processing", last=True) - ts("CmdStan start processing")).total_seconds()
out_dir = os.path.join(common.ROOT, ".tmp", name)
csvs = sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".csv"))
assert len(csvs) == common.CHAINS, csvs
r0 = resource.getrusage(resource.RUSAGE_SELF)
t0 = time.perf_counter()
facts = run_cmdstan.postprocess(csvs, name, out_dir)
post_wall = time.perf_counter() - t0
r1 = resource.getrusage(resource.RUSAGE_SELF)
post_cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
seed = int(re.search(r"_s(\d+)$", name).group(1))
rec["phases"] = {"compile": compile_s, "sampling": sampling_s, "postprocess": post_wall}
rec["extra"] = {"seed": seed, **facts, **run_cmdstan.describe(reduce_sum)}
rec["recovered"] = {
    "why": "post-processing crashed on CSV column names (beta[1] instead of CmdStan's beta.1) after sampling "
           "finished; the CSV files were post-processed afterwards by recover_cmdstan.py",
    "measured_total_wall_seconds": rec["total_wall_seconds"], "measured_cpu_seconds": rec["cpu_seconds"],
    "postprocess_wall_seconds": post_wall, "postprocess_cpu_seconds": post_cpu,
    "postprocess_peak_rss_bytes": r1.ru_maxrss * 1024,
    "compile_and_sampling_from": "cmdstanpy log timestamps (1 s resolution)",
}
rec["total_wall_seconds"] += post_wall
rec["cpu_seconds"] += post_cpu
rec["cpu_user_seconds"] += r1.ru_utime - r0.ru_utime
rec["cpu_system_seconds"] += r1.ru_stime - r0.ru_stime
rec["peak_memory_bytes"] = max(rec["peak_memory_bytes"], r1.ru_maxrss * 1024)
rec["exit_status"] = 0
json.dump(rec, open(rec_path, "w"), indent=1)
print(json.dumps({k: rec[k] for k in ("phases", "recovered", "total_wall_seconds", "cpu_seconds")}, indent=1))
