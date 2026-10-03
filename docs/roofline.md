# Roofline: how close the kernels are to the hardware

A quick estimate, made on 2026-10-02 at commit d697cc1 on one core of a
Ryzen 9 5900X (Zen 3, AVX2 and FMA, no AVX-512), to see how much
same-algorithm speed is left before any low-level work. Single-threaded
gradients only; the sampler and multi-threaded runs were not measured.

## Peaks

`bench/roofline/peak.c` (`clang -O2 -march=native`, run as
`taskset -c 2 ./peak`):

| resource | measured |
|---|---|
| FMA arithmetic (12 independent 4-wide FMA chains) | 76 GFLOP/s, about 16 operations per cycle at about 4.75 GHz |
| reading a 16 KB buffer (L1) | 302 GB/s |
| 256 KB (L2) | 141 GB/s |
| 4 and 16 MB (L3) | 115 to 117 GB/s |
| 512 MB (memory) | 15.1 GB/s |

These are simple read loops, not tuned to the limit; treat them as
approximate ceilings.

## The kernels

Retired floating-point operations (`fp_ret_sse_avx_ops.all`, an FMA counted
as two) and cycles per gradient from the hardware counters
(`bench/roofline/count.py`; user-mode counts; each gradient program at K
and 2K evaluations, difference divided by K). The memory column is an estimate from the sizes of
the data and parameter arrays each gradient reads and writes, not a
measurement.

| kernel | Martin, operations per cycle | share of FMA peak | estimated traffic | Rust, operations per cycle | Martin's operations vs Rust's |
|---|---|---|---|---|---|
| Newton (X 200,000 x 50, 10 iterations) | 11.9 | 74% | about 7 GB/s; memory gives 14 | 5.5 | 11% more |
| logistic gradient (n = 5000, p = 20) | 6.3 | 40% | about 35 GB/s from L3; L3 gives 115 | 3.3 | 12% more |
| time-series gradient, 37,901 parameters | 6.0 | 38% | about 22 GB/s from L3 | 4.0 | 21% more |
| time-series gradient, 3,171 parameters | 6.4 | 40% | fits in L2 | 4.7 | 23% more |

Newton's counts are for the whole program (user mode, so reading the data
file, mostly kernel time, barely enters); by the fit's own timing, 6.3 GFLOP
in 0.114 s is about 55 GFLOP/s, so 72 to 74% of the FMA peak either way. "Rust" is the max-effort baseline in
each case (`baselines/*_max.rs`).

## Reading

- **Newton is near the arithmetic limit.** About 1.3x is left relative to the
  FMA peak (Zen 3's separate add pipes could in principle let mixed code go
  a little above it), and the last part is the hardest to get.
- **The logistic and time-series gradients are limited by neither arithmetic
  nor bandwidth.** They use about 40% of the FMA peak and 20 to 30% of the
  estimated L3 bandwidth. The likely limit is latency: the running sum makes
  each column of a group wait for the previous one (3 cycles per add on Zen 3,
  with two independent chains per group of 8 series), and Martin's `exp`,
  `log` and `log1p` look up tables with gathers, which have long latency.
  This is a reading of the code and the counts, not a measured breakdown.
- **So there is plausibly room left on those two at the same algorithm**
  (they would need to roughly double their operations per cycle to reach
  80% of the FMA peak, which latency-bound code rarely does), from giving the core more independent work at once (more
  groups, or several chains' gradients interleaved in one kernel, item 5 of
  the [roadmap](roadmap.md)) and from cheaper table lookups. The dependency
  chains come from the algorithm and its tiling, not from instruction
  selection.
- **Martin does more arithmetic than the Rust but uses each cycle better**:
  11 to 23% more operations (its polynomial `exp`, padded tiles, extra sums)
  at 1.4 to 2.2 times the operations per cycle.
