# Whole runs at commit d51f859 (wave 1 merged)

`python3 bench/dynpois/seed_runs.py SIZE SEEDS`, 4 chains, 1000 warmup +
1000 draws, Ryzen 9 5900X, 2026-09-30. Both programs were built against the
runtime of d51f859 (the Rust baseline links the same sampler; its gradient is
the hand-written AVX2 code in baselines/dynpois_max.rs and runs on one thread
per chain, Mint's on the chain's threads). Lowest ESS and highest R-hat are
the runtime's own estimates over all parameters. One run per seed; wall times
of the same binary vary by about 20% between runs on this machine.

Large model (37,901 parameters):

| program | seed | sampling s | gradients | µs per gradient per chain | lowest ESS | highest R-hat |
|---|---|---|---|---|---|---|
| Mint | 11 | 64.68 | 3696457 | 70.0 | 332 | 1.007 |
| Mint | 5 | 61.42 | 3653061 | 67.2 | 424 | 1.007 |
| Mint | 11 | 62.81 | 3696457 | 68.0 | 332 | 1.007 |
| max-effort Rust | 11 | 92.81 | 3671785 | 101.1 | 386 | 1.005 |

Small model (3,171 parameters):

| program | seed | sampling s | gradients | µs per gradient per chain | lowest ESS | highest R-hat |
|---|---|---|---|---|---|---|
| Mint | 1 | 3.67 | 2038971 | 7.2 | 2041 | 1.001 |
| max-effort Rust | 1 | 3.73 | 2018606 | 7.4 | 1702 | 1.002 |
| Mint | 2 | 3.68 | 2022472 | 7.3 | 2028 | 1.003 |
| max-effort Rust | 2 | 3.79 | 2023143 | 7.5 | 2185 | 1.002 |
| Mint | 3 | 3.75 | 2023364 | 7.4 | 2081 | 1.002 |
| max-effort Rust | 3 | 3.80 | 2027395 | 7.5 | 2144 | 1.001 |
