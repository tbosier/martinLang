#!/usr/bin/env python3
"""Runs one benchmark command and measures it from outside, identically for
every framework.

    measure.py NAME [--timeout S] [--cpus LIST] -- COMMAND...

- Waits until the 1-minute load average is below 3 and fewer than 2
  logical CPUs are busy (other work may share the machine; at most 30
  minutes), and records the load average before and after and how busy the
  CPUs outside the pinned set were during the run.
- Runs COMMAND under `taskset -c LIST` (default 0-11: one hardware thread on
  each of the machine's 12 physical cores, so every framework has the same
  12-core budget and none can use SMT siblings).
- Total wall time: from launch to exit of the whole process tree.
- CPU time: user + system of the process and every descendant it waited
  for (os.wait4's rusage), which covers CmdStan's chain processes.
- Peak memory: the process tree is polled every 0.2 s and the resident set
  sizes of all its processes are summed (shared pages count once per
  process, so this overstates multi-process runs slightly); the
  proportional set size (PSS, shared pages divided between the processes)
  is summed too, every 1 s. ru_maxrss (largest single process, exact) is
  recorded as a cross-check; the reported peak is max(polled RSS sum,
  ru_maxrss).
- Thread count: the peak number of threads in the tree.
- The command's own phase timings (compile, sampling, ...) are read from the
  JSON file it writes to $SHOOTOUT_PHASES.

Writes bench/shootout/results/runs/NAME.json and build/shootout/logs/NAME.log.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

PAGE = os.sysconf("SC_PAGE_SIZE")


def loadavg():
    return [float(x) for x in open("/proc/loadavg").read().split()[:3]]


def children_map():
    kids = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            st = open(f"/proc/{p}/stat").read()
        except OSError:
            continue
        rest = st[st.rindex(")") + 2:].split()
        kids.setdefault(int(rest[1]), []).append(int(p))
    return kids


def tree(pid):
    kids = children_map()
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo.extend(kids.get(p, []))
    return out


def rss_threads(pids):
    rss = thr = 0
    for p in pids:
        try:
            rss += int(open(f"/proc/{p}/statm").read().split()[1]) * PAGE
            st = open(f"/proc/{p}/stat").read()
            thr += int(st[st.rindex(")") + 2:].split()[17])
        except (OSError, IndexError, ValueError):
            pass
    return rss, thr


def pss(pids):
    tot = 0
    for p in pids:
        try:
            for line in open(f"/proc/{p}/smaps_rollup"):
                if line.startswith("Pss:"):
                    tot += int(line.split()[1]) * 1024
                    break
        except OSError:
            pass
    return tot


def cpu_times():
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            v = [int(x) for x in f[1:]]
            out[int(f[0][3:])] = (sum(v), v[3] + v[4])
    return out


def cpu_list(spec):
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit(__doc__)
    k = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--cpus", default="0-11")
    ap.add_argument("--max-load", type=float, default=3.0)
    a = ap.parse_args(argv[:k])
    cmd = argv[k + 1:]

    waited = 0.0
    while loadavg()[0] >= a.max_load:
        if waited == 0.0:
            print(f"waiting for load < {a.max_load} (now {loadavg()[0]})", flush=True)
        time.sleep(15)
        waited += 15

    # The load average lags; also require that fewer than 2 logical CPUs are busy
    # right now (a 5 s sample), waiting up to 30 minutes for other work to finish.
    def busy_now():
        c0 = cpu_times()
        time.sleep(5)
        c1 = cpu_times()
        return sum((c1[c][0] - c0[c][0] - (c1[c][1] - c0[c][1])) / max(1, c1[c][0] - c0[c][0]) for c in c1)

    busy_before = busy_now()
    while busy_before >= 2.0 and waited < 1800:
        print(f"waiting for other work to stop ({busy_before:.1f} CPUs busy)", flush=True)
        time.sleep(25)
        waited += 30
        busy_before = busy_now()
    before = loadavg()

    logs = os.path.join(common.BUILD, "logs")
    os.makedirs(logs, exist_ok=True)
    os.makedirs(os.path.join(common.RESULTS, "runs"), exist_ok=True)
    phases_path = os.path.join(logs, a.name + ".phases.json")
    if os.path.exists(phases_path):
        os.remove(phases_path)
    env = dict(os.environ, **common.cache_env(), SHOOTOUT_PHASES=phases_path, SHOOTOUT_RUN=a.name)
    log = open(os.path.join(logs, a.name + ".log"), "w")
    full = ["taskset", "-c", a.cpus] + cmd if a.cpus else cmd
    ct0 = cpu_times()
    t0 = time.perf_counter()
    proc = subprocess.Popen(full, cwd=common.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    peak = {"rss": 0, "pss": 0, "threads": 0}
    stop = threading.Event()

    def poll():
        n = 0
        while not stop.is_set():
            pids = tree(proc.pid)
            r, t = rss_threads(pids)
            peak["rss"] = max(peak["rss"], r)
            peak["threads"] = max(peak["threads"], t)
            if n % 5 == 0:
                peak["pss"] = max(peak["pss"], pss(pids))
            n += 1
            stop.wait(0.2)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    timed_out = False
    status = ru = None
    deadline = t0 + a.timeout
    while True:
        pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
        if pid:
            break
        if time.perf_counter() > deadline:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(10)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            pid, status, ru = os.wait4(proc.pid, 0)
            break
        time.sleep(0.1)
    wall = time.perf_counter() - t0
    stop.set()
    th.join()
    log.close()
    after = loadavg()
    ct1 = cpu_times()
    pinned = cpu_list(a.cpus) if a.cpus else set(ct1)
    busy = {c: (ct1[c][0] - ct0[c][0] - (ct1[c][1] - ct0[c][1])) / max(1, ct1[c][0] - ct0[c][0]) for c in ct1}
    outside = sorted(set(ct1) - pinned)

    phases = json.load(open(phases_path)) if os.path.exists(phases_path) else {"phases": {}, "extra": {}}
    rec = {
        "name": a.name,
        "command": cmd,
        "cpus": a.cpus,
        "loadavg_before": before,
        "loadavg_after": after,
        "waited_for_load_seconds": waited,
        "busy_cpus_before": busy_before,
        "exit_status": os.waitstatus_to_exitcode(status),
        "timed_out": timed_out,
        "timeout_seconds": a.timeout,
        "total_wall_seconds": wall,
        "cpu_user_seconds": ru.ru_utime,
        "cpu_system_seconds": ru.ru_stime,
        "cpu_seconds": ru.ru_utime + ru.ru_stime,
        "maxrss_largest_process_bytes": ru.ru_maxrss * 1024,
        "peak_rss_tree_sum_bytes": peak["rss"],
        "peak_pss_tree_sum_bytes": peak["pss"],
        "peak_memory_bytes": max(peak["rss"], ru.ru_maxrss * 1024),
        "peak_threads": peak["threads"],
        "busy_cpus_inside_set": sum(busy[c] for c in pinned),
        "busy_cpus_outside_set": sum(busy[c] for c in outside),
        "busy_cpus_note": "average number of busy logical CPUs during the run, inside and outside the taskset "
                          "list; work outside it is other work on the machine (CPUs 12-23 are the SMT siblings "
                          "of 0-11, so it competes for the same cores)",
        **phases,
    }
    out = os.path.join(common.RESULTS, "runs", a.name + ".json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"{a.name}: exit {rec['exit_status']}{' TIMED OUT' if timed_out else ''}, wall {wall:.1f} s, "
          f"cpu {rec['cpu_seconds']:.1f} s, peak mem {rec['peak_memory_bytes'] / 2**30:.2f} GiB, "
          f"threads {peak['threads']}, load before {before[0]:.2f}, other work {rec['busy_cpus_outside_set']:.2f} CPUs -> {out}", flush=True)
    sys.exit(0 if rec["exit_status"] == 0 else 1)


if __name__ == "__main__":
    main()
