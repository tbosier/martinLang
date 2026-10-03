# Merged build (59dca39) against the frozen Rust reference

`python3 bench/mint_vs_ref.py 7`, 2026-10-02, Ryzen 9 5900X, pinned to core 2. The Rust medians are the frozen values in bench/rust_reference.json (measured once, interleaved, 2026-09-30), not re-measured in this run.

load average 0.71; 7 runs each
| problem | Martin median (range) | Rust reference median | Rust / Martin |
|---|---|---|---|
| dynpois_small | 3.52 µs (3.50 µs to 3.67 µs) | 4.07 µs | 1.156x |
| dynpois_large | 43.71 µs (43.34 µs to 44.04 µs) | 52.16 µs | 1.193x |
| logistic | 23.56 µs (22.90 µs to 23.63 µs) | 37.87 µs | 1.607x |
| linear | 0.10 µs (0.10 µs to 0.10 µs) | 0.26 µs | 2.601x |
| newton | 0.114 s (0.113 s to 0.121 s) | 0.209 s | 1.831x |
