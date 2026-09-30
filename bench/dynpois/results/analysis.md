## Diagnostics (pop, beta[g], terminal[g]; ArviZ rank R-hat, bulk/tail ESS)

'mixed' = max R-hat <= 1.01 and min bulk and tail ESS >= 400. ESS/s for a run that has not mixed is not a usable efficiency figure (its ESS estimate is unreliable).

| run | size | warmup | draws x thin | wall s | mixed | max R-hat (at) | share R-hat>1.01 | min bulk ESS (at) | min tail ESS | min bulk ESS/s | gradients |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mint_large | large | 1000 | 1000 x 1 | 277.7 | yes | 1.009 (pop) | 0.000 | 463 (pop) | 919 | 1.67 | 3662634 |
| mint_small | small | 1000 | 1000 x 1 | 8.3 | yes | 1.002 (terminal[15]) | 0.000 | 2301 (pop) | 2648 | 276 | 1997671 |
| rust_max_large | large | 1000 | 1000 x 1 | 220.0 | yes | 1.009 (pop) | 0.000 | 449 (pop) | 984 | 2.04 | 3670608 |
| rust_max_small | small | 1000 | 1000 x 1 | 6.6 | yes | 1.003 (beta[0]) | 0.000 | 1983 (pop) | 2382 | 299 | 2023407 |
| rustmc_large | large | 1000 | 125 x 8 | 109.1 | NO | 2.565 (pop) | 0.996 | 5 (pop) | 11 | 0.0468 (not mixed) | n/a |
| rustmc_small | small | 1000 | 1000 x 1 | 7.0 | NO | 1.953 (terminal[5]) | 1.000 | 6 (terminal[5]) | 15 | 0.793 (not mixed) | n/a |
| stan_large | large | 300 | 300 x 1 | 3322.0 | NO | 1.023 (pop) | 0.016 | 125 (pop) | 261 | 0.0375 (not mixed) | 1189242 |
| stan_small | small | 1000 | 1000 x 1 | 67.7 | yes | 1.002 (beta[4]) | 0.000 | 1860 (pop) | 2486 | 27.5 | 2010131 |
| stan_small_o1 | small | 1000 | 1000 x 1 | 48.5 | yes | 1.002 (beta[4]) | 0.000 | 1860 (pop) | 2486 | 38.3 | 2010131 |

## Truth recovery (descriptive: z = (posterior mean - truth) / posterior sd, one data set)

| run | pop mean (sd) | pop z | mean z | max abs z | share abs z > 2 |
|---|---|---|---|---|---|
| mint_large | 1.476 (0.059) | -0.42 | -0.00 | 3.17 | 0.042 |
| mint_small | 1.448 (0.124) | -0.42 | -0.16 | 2.70 | 0.024 |
| rust_max_large | 1.478 (0.061) | -0.36 | -0.00 | 3.28 | 0.042 |
| rust_max_small | 1.451 (0.122) | -0.40 | -0.16 | 2.74 | 0.024 |
| rustmc_large | 1.447 (0.093) | -0.57 | -0.11 | 2.89 | 0.032 |
| rustmc_small | 1.383 (0.094) | -1.24 | -0.26 | 2.26 | 0.073 |
| stan_large | 1.483 (0.066) | -0.26 | -0.00 | 3.15 | 0.040 |
| stan_small | 1.446 (0.121) | -0.45 | -0.17 | 2.74 | 0.024 |
| stan_small_o1 | 1.446 (0.121) | -0.45 | -0.17 | 2.74 | 0.024 |

## Posterior-mean agreement between runs on the same data

MCSE-scaled differences are shown only when both runs mixed and are independent (same implementation with the same seed shares chain trajectories). Means only; agreement of means does not establish agreement of variances or tails.

| size | run a | run b | both mixed | independent | max abs diff / sd (at) | median | max abs diff / MCSE | share > 3 MCSE |
|---|---|---|---|---|---|---|---|---|
| large | mint_large | rust_max_large | yes | yes | 0.054 (beta[199]) | 0.014 | 2.50 | 0.000 |
| large | mint_large | rustmc_large | NO | yes | 0.881 (beta[43]) | 0.131 | n/a | n/a |
| large | mint_large | stan_large | NO | yes | 0.112 (pop) | 0.030 | n/a | n/a |
| large | rust_max_large | rustmc_large | NO | yes | 0.909 (beta[43]) | 0.135 | n/a | n/a |
| large | rust_max_large | stan_large | NO | yes | 0.091 (beta[226]) | 0.024 | n/a | n/a |
| large | rustmc_large | stan_large | NO | yes | 0.928 (beta[43]) | 0.145 | n/a | n/a |
| small | mint_small | rust_max_small | yes | yes | 0.048 (beta[17]) | 0.011 | 2.27 | 0.000 |
| small | mint_small | rustmc_small | NO | yes | 0.842 (beta[9]) | 0.212 | n/a | n/a |
| small | mint_small | stan_small | yes | yes | 0.028 (terminal[18]) | 0.012 | 1.45 | 0.000 |
| small | mint_small | stan_small_o1 | yes | yes | 0.028 (terminal[18]) | 0.012 | 1.45 | 0.000 |
| small | rust_max_small | rustmc_small | NO | yes | 0.842 (beta[9]) | 0.233 | n/a | n/a |
| small | rust_max_small | stan_small | yes | yes | 0.056 (beta[17]) | 0.019 | 2.51 | 0.000 |
| small | rust_max_small | stan_small_o1 | yes | yes | 0.056 (beta[17]) | 0.019 | 2.51 | 0.000 |
| small | rustmc_small | stan_small | NO | yes | 0.861 (beta[9]) | 0.203 | n/a | n/a |
| small | rustmc_small | stan_small_o1 | NO | yes | 0.861 (beta[9]) | 0.203 | n/a | n/a |
| small | stan_small | stan_small_o1 | yes | NO | 0.000 (pop) | 0.000 | n/a | n/a |
