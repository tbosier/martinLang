"""Shared definitions of the same-sampler harness: the problems, the
implementations of each, how to run one with given sampler settings, and how
to read the runtime's report. Run the scripts from the repository root."""
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "build", "same_sampler")
RESULTS = os.path.join(ROOT, "bench", "same_sampler", "results")
MINTC = os.path.join(ROOT, "compiler", "target", "release", "mintc")

# problem -> (Mint source, data-path rewrite, Stan model, Stan data, Rust baselines)
PROBLEMS = {
    "dynpois_small": dict(mint="examples/dynamic_poisson.mint", mint_data=("data_small", "data_small"),
                          stan={"stan": "dynpois"}, json="dynpois_small",
                          rust={"rust_max": ["rs_dynpois_max", "bench/dynpois/data_small/y.f64"],
                                "rust_par": ["rs_dynpois_par", "bench/dynpois/data_small/y.f64"]}),
    "dynpois_large": dict(mint="examples/dynamic_poisson.mint", mint_data=("data_small", "data_large"),
                          stan={"stan": "dynpois"}, json="dynpois_large",
                          rust={"rust_max": ["rs_dynpois_max", "bench/dynpois/data_large/y.f64"],
                                "rust_par": ["rs_dynpois_par", "bench/dynpois/data_large/y.f64"]}),
    "logistic": dict(mint="examples/logistic_bayes.mint", mint_data=None,
                     stan={"stan": "logistic", "stan_glm": "logistic_glm"}, json="logistic",
                     rust={"rust_max": ["rs_logistic_bayes_max"]}),
    "eight_schools": dict(mint="examples/eight_schools.mint", mint_data=None,
                          stan={"stan": "eight_schools"}, json="eight_schools",
                          rust={"rust": ["rs_eight_schools"]}),
}


def implementations(problem):
    p = PROBLEMS[problem]
    return ["mint"] + list(p["rust"]) + list(p["stan"])


def loadavg():
    return [float(x) for x in open("/proc/loadavg").read().split()[:3]]


def _cpu_times():
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            v = [int(x) for x in f[1:]]
            out[int(f[0][3:])] = (sum(v), v[3] + v[4])
    return out


def _read_list(path):
    out = []
    for part in open(path).read().strip().split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def topology():
    """[(L3 group: [(physical core: [its hardware threads])])] from sysfs."""
    cpus = sorted(_cpu_times())
    cores, l3 = {}, {}
    for c in cpus:
        sib = tuple(_read_list(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"))
        l3c = tuple(_read_list(f"/sys/devices/system/cpu/cpu{c}/cache/index3/shared_cpu_list"))
        cores[sib] = l3c
    for sib, g in cores.items():
        l3.setdefault(g, []).append(list(sib))
    return [sorted(v) for _, v in sorted(l3.items())]


def quiet_cpus(n, busy):
    """n physical cores in one L3 whose hardware threads were least busy;
    returns the first hardware thread of each (a taskset list) and their score."""
    best = None
    for group in topology():
        scored = sorted(group, key=lambda sib: sum(busy[c] for c in sib))
        if len(scored) < n:
            continue
        pick = scored[:n]
        score = sum(busy[c] for sib in pick for c in sib)
        if best is None or score < best[1]:
            best = (pick, score)
    return ",".join(str(sib[0]) for sib in best[0]), best[1]


def cpu_busy(seconds=1.0):
    """Percent busy of each CPU over the next `seconds` (all processes)."""
    a = _cpu_times()
    time.sleep(seconds)
    b = _cpu_times()
    return {c: round(100 * (1 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]))) for c in sorted(a)}


def mint_program(problem, draws, warmup, chains, seed):
    """Builds (once) the Mint program for these sampler settings; returns its path."""
    p = PROBLEMS[problem]
    src = open(os.path.join(ROOT, p["mint"])).read()
    if p["mint_data"]:
        a, b = p["mint_data"]
        src = src.replace(f"bench/dynpois/{a}/y.f64", f"bench/dynpois/{b}/y.f64")
    new, n = re.subn(r"draws = \d+, warmup = \d+, chains = \d+, seed = \d+",
                     f"draws = {draws}, warmup = {warmup}, chains = {chains}, seed = {seed}", src)
    assert n == 1, f"cannot set the sampler settings in {p['mint']}"
    d = os.path.join(OUT, "mint")
    os.makedirs(d, exist_ok=True)
    prog = os.path.join(d, f"{problem}_d{draws}_w{warmup}_c{chains}_s{seed}")
    if not os.path.exists(prog) or open(prog + ".mint").read() != new:
        open(prog + ".mint", "w").write(new)
        for attempt in range(3):  # (rustc and gcc have crashed spuriously on this machine)
            r = subprocess.run([MINTC, "build", prog + ".mint", "-o", prog], cwd=ROOT, capture_output=True, text=True)
            if r.returncode == 0:
                break
        else:
            raise RuntimeError(r.stderr)
    return prog


def command(problem, impl, draws=1000, warmup=1000, chains=4, seed=1):
    """(argv, extra environment) running `impl` on `problem` with these settings."""
    p = PROBLEMS[problem]
    if impl == "mint":
        return [mint_program(problem, draws, warmup, chains, seed)], {}
    if impl in p["rust"]:
        exe, *args = p["rust"][impl]
        if exe.startswith("rs_dynpois"):
            args = args + [str(seed)]
        env = {"MINT_BASELINE_DRAWS": str(draws), "MINT_BASELINE_WARMUP": str(warmup),
               "MINT_BASELINE_CHAINS": str(chains), "MINT_BASELINE_SEED": str(seed)}
        return [os.path.join(OUT, exe)] + args, env
    if impl in p["stan"]:
        so = os.path.join(OUT, "stan", p["stan"][impl] + "_model.so")
        js = os.path.join(OUT, "data", p["json"] + ".json")
        return [os.path.join(OUT, "bs_driver"), so, js, str(draws), str(warmup), str(chains), str(seed)], {}
    raise KeyError(impl)


def run(argv, env_extra, pin=None, timeout=None):
    """Runs a program from the repository root; returns (stdout + stderr, process wall seconds)."""
    env = dict(os.environ, **env_extra)
    if pin is not None:
        argv = ["taskset", "-c", pin] + argv
    t = time.perf_counter()
    r = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    wall = time.perf_counter() - t
    out = r.stdout + r.stderr
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)} failed ({r.returncode}):\n{out[-3000:]}")
    return out, wall


def parse_grad_bench(out):
    m = re.search(r"grad-bench: reps=(\d+) ns_per_eval=(\S+) logp=(\S+) grad_norm=(\S+)", out)
    return {"reps": int(m.group(1)), "ns": float(m.group(2)), "logp": float(m.group(3)), "grad_norm": float(m.group(4))}


def parse_printed_grad(out):
    lp = float(re.search(r"exact log density: (\S+)", out).group(1))
    g = [float(x) for x in out.split("grad:")[1].split("\n")[0].split()]
    return lp, g


def parse_run(out):
    """The runtime's report of a whole run (mint_print_posterior)."""
    num = r"([-\d.eE+]+|nan|inf)"
    m = re.search(r"chains=(\d+) draws/chain=(\d+) gradients=(\d+) divergences=(\d+) step size=(\S+) leapfrog/draw=(\S+)", out)
    s = re.search(r"all \d+ parameters: highest rhat " + num + r" \((\S+)\), lowest ess " + num + r" \((\S+)\)", out)
    w = re.search(r"gradients: warmup=(\d+) sampling=(\d+) \(warmup=(\w+), (\d+) iterations\)", out)
    t = re.search(r"sampler: threads per chain=(\d+) \(smallest team that ran=(\d+)\)", out)
    return {
        "sampling_seconds": float(re.search(r"sampling took (\S+) s", out).group(1)),
        "prep_seconds": float(re.search(r"preparation took (\S+) s", out).group(1)),
        "chains": int(m.group(1)), "draws": int(m.group(2)), "gradients": int(m.group(3)),
        "divergences": int(m.group(4)),
        "step_size": [float(x) for x in m.group(5).split(",")],
        "leapfrog_per_draw": [float(x) for x in m.group(6).split(",")],
        "max_rhat": float(s.group(1)), "max_rhat_param": s.group(2),
        "min_ess": float(s.group(3)), "min_ess_param": s.group(4),
        "warmup_gradients": int(w.group(1)), "warmup_kind": w.group(3), "warmup_iters": int(w.group(4)),
        "threads_per_chain": int(t.group(1)), "smallest_team": int(t.group(2)),
    }
