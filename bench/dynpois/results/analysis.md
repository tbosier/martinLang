## Diagnostics (pop, beta[g], terminal[g]; ArviZ rank R-hat, bulk/tail ESS)

'mixed' = max R-hat <= 1.01 and min bulk and tail ESS >= 400. ESS/s for a run that has not mixed is not a usable efficiency figure (its ESS estimate is unreliable).

| run | size | warmup | draws x thin | wall s | mixed | max R-hat (at) | share R-hat>1.01 | min bulk ESS (at) | min tail ESS | min bulk ESS/s | gradients |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mint_large | large | 1000 | 1000 x 1 | 1448.3 | NO | 1.016 (pop) | 0.002 | 363 (pop) | 674 | 0.25 (not mixed) | 3668578 |
| mint_small | small | 1000 | 1000 x 1 | 15.3 | yes | 1.002 (terminal[15]) | 0.000 | 2301 (pop) | 2648 | 150 | 1997671 |
| rust_max_large | large | 1000 | 1000 x 1 | 1291.5 | NO | 1.008 (pop) | 0.000 | 370 (pop) | 737 | 0.286 (not mixed) | 3650160 |
| rust_max_small | small | 1000 | 1000 x 1 | 13.0 | yes | 1.003 (beta[0]) | 0.000 | 1983 (pop) | 2382 | 152 | 2023407 |
| rustmc_large | large | 1000 | 125 x 8 | 109.1 | NO | 2.565 (pop) | 0.996 | 5 (pop) | 11 | 0.0468 (not mixed) | n/a |
| rustmc_small | small | 1000 | 1000 x 1 | 7.0 | NO | 1.953 (terminal[5]) | 1.000 | 6 (terminal[5]) | 15 | 0.793 (not mixed) | n/a |
| stan_small | small | 1000 | 1000 x 1 | 67.7 | yes | 1.002 (beta[4]) | 0.000 | 1860 (pop) | 2486 | 27.5 | 2010131 |

## Truth recovery (descriptive: z = (posterior mean - truth) / posterior sd, one data set)

| run | pop mean (sd) | pop z | mean z | max abs z | share abs z > 2 |
|---|---|---|---|---|---|
| mint_large | 1.474 (0.062) | -0.42 | -0.01 | 3.29 | 0.038 |
| mint_small | 1.448 (0.124) | -0.42 | -0.16 | 2.70 | 0.024 |
| rust_max_large | 1.471 (0.063) | -0.46 | -0.01 | 3.30 | 0.038 |
| rust_max_small | 1.451 (0.122) | -0.40 | -0.16 | 2.74 | 0.024 |
| rustmc_large | 1.447 (0.093) | -0.57 | -0.11 | 2.89 | 0.032 |
| rustmc_small | 1.383 (0.094) | -1.24 | -0.26 | 2.26 | 0.073 |
| stan_small | 1.446 (0.121) | -0.45 | -0.17 | 2.74 | 0.024 |

## Posterior-mean agreement between runs on the same data

MCSE-scaled differences are shown only when both runs mixed and are independent (same implementation with the same seed shares chain trajectories). Means only; agreement of means does not establish agreement of variances or tails.

| size | run a | run b | both mixed | independent | max abs diff / sd (at) | median | max abs diff / MCSE | share > 3 MCSE |
|---|---|---|---|---|---|---|---|---|
| large | mint_large | rust_max_large | NO | yes | 0.075 (terminal[114]) | 0.015 | n/a | n/a |
| large | mint_large | rustmc_large | NO | yes | 0.876 (beta[43]) | 0.130 | n/a | n/a |
| large | rust_max_large | rustmc_large | NO | yes | 0.843 (beta[43]) | 0.123 | n/a | n/a |
| small | mint_small | rust_max_small | yes | yes | 0.048 (beta[17]) | 0.011 | 2.27 | 0.000 |
| small | mint_small | rustmc_small | NO | yes | 0.842 (beta[9]) | 0.212 | n/a | n/a |
| small | mint_small | stan_small | yes | yes | 0.028 (terminal[18]) | 0.012 | 1.45 | 0.000 |
| small | rust_max_small | rustmc_small | NO | yes | 0.842 (beta[9]) | 0.233 | n/a | n/a |
| small | rust_max_small | stan_small | yes | yes | 0.056 (beta[17]) | 0.019 | 2.51 | 0.000 |
| small | rustmc_small | stan_small | NO | yes | 0.861 (beta[9]) | 0.203 | n/a | n/a |
