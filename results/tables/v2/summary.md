## test_id

| method | seeds | scenarios | weighted_score | value_fraction | makespan | completion_rate |
|---|---|---|---|---|---|---|
| random | – | 500 | 0.104 ± 0.074 [0.098, 0.111] | 0.104 ± 0.074 [0.098, 0.111] | 1703.805 ± 221.206 [1684.265, 1723.756] | 0.504 ± 0.188 [0.487, 0.520] |
| greedy_value_rate | – | 500 | 0.364 ± 0.113 [0.355, 0.374] | 0.364 ± 0.113 [0.355, 0.374] | 1541.208 ± 236.351 [1520.289, 1561.875] | 0.812 ± 0.147 [0.798, 0.824] |
| greedy_nearest | – | 500 | 0.327 ± 0.114 [0.318, 0.338] | 0.327 ± 0.114 [0.318, 0.338] | 1448.015 ± 223.107 [1428.526, 1467.456] | 0.820 ± 0.119 [0.810, 0.830] |
| rh_rk | – | 500 | 0.362 ± 0.119 [0.352, 0.373] | 0.362 ± 0.119 [0.352, 0.373] | 1511.892 ± 231.799 [1491.463, 1532.556] | 0.817 ± 0.156 [0.803, 0.830] |
| ppo_entity_hard | 3 | 500 | 0.365 ± 0.002 [0.355, 0.374] | 0.365 ± 0.002 [0.355, 0.374] | 1509.930 ± 3.605 [1491.355, 1528.713] | 0.825 ± 0.000 [0.813, 0.837] |
| ppo_entity_hard_lookahead | 3 | 500 | 0.386 ± 0.000 [0.376, 0.396] | 0.386 ± 0.000 [0.376, 0.396] | 1489.583 ± 5.066 [1470.539, 1508.923] | 0.840 ± 0.002 [0.828, 0.851] |

| method | seeds | scenarios | energy_used_total | risk_exposure | mean_response_delay | latency_median_ms | failed_mission | n_invalid_actions | critical_on_time_rate |
|---|---|---|---|---|---|---|---|---|---|
| random | – | 500 | 4.240 ± 1.311 [4.124, 4.355] | 0.519 ± 0.087 [0.511, 0.526] | 29.374 ± 13.007 [28.263, 30.520] | 0.011 ± 0.007 [0.010, 0.012] | 0.000 ± 0.000 [0.000, 0.000] | 0.000 ± 0.000 [0.000, 0.000] | 0.171 ± 0.164 [0.157, 0.185] |
| greedy_value_rate | – | 500 | 4.030 ± 1.224 [3.923, 4.137] | 0.453 ± 0.070 [0.446, 0.459] | 36.000 ± 17.735 [34.466, 37.615] | 0.059 ± 0.024 [0.057, 0.061] | 0.000 ± 0.000 [0.000, 0.000] | 0.000 ± 0.000 [0.000, 0.000] | 0.679 ± 0.184 [0.663, 0.695] |
| greedy_nearest | – | 500 | 3.916 ± 1.210 [3.809, 4.021] | 0.426 ± 0.064 [0.421, 0.432] | 37.272 ± 17.541 [35.764, 38.835] | 0.016 ± 0.003 [0.016, 0.016] | 0.000 ± 0.000 [0.000, 0.000] | 0.000 ± 0.000 [0.000, 0.000] | 0.528 ± 0.210 [0.510, 0.546] |
| rh_rk | – | 500 | 3.987 ± 1.201 [3.880, 4.091] | 0.447 ± 0.068 [0.440, 0.452] | 30.990 ± 16.741 [29.556, 32.498] | 31.924 ± 5.100 [31.479, 32.374] | 0.000 ± 0.000 [0.000, 0.000] | 0.000 ± 0.000 [0.000, 0.000] | 0.661 ± 0.195 [0.644, 0.678] |
| ppo_entity_hard | 3 | 500 | 3.989 ± 0.011 [3.883, 4.093] | 0.447 ± 0.003 [0.441, 0.453] | 36.019 ± 0.850 [34.770, 37.300] | 0.319 ± 0.014 [0.318, 0.321] | 0.000 ± 0.000 [0.000, 0.000] | 0.000 ± 0.000 [0.000, 0.000] | 0.675 ± 0.010 [0.660, 0.690] |
| ppo_entity_hard_lookahead | 3 | 500 | 3.958 ± 0.014 [3.853, 4.061] | 0.437 ± 0.001 [0.431, 0.442] | 31.858 ± 0.869 [30.636, 33.143] | 15.498 ± 1.277 [14.965, 16.043] | 0.000 ± 0.000 [0.000, 0.000] | 0.000 ± 0.000 [0.000, 0.000] | 0.699 ± 0.005 [0.684, 0.714] |

## Paired comparisons on test_id (weighted score J)

| condition | A | B | metric | n | mean A | mean B | mean Δ [95 % CI] | r_rb | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| test_id | ppo_entity_hard | random | weighted_score | 500 | 0.3646 | 0.1042 | +0.2604 [+0.2524, +0.2684] | +1.00 | 1.3e-83 | 1e-82 |
| test_id | ppo_entity_hard | greedy_value_rate | weighted_score | 500 | 0.3646 | 0.3645 | +0.0001 [-0.0023, +0.0024] | +0.01 | 0.91 | 1 |
| test_id | ppo_entity_hard | greedy_nearest | weighted_score | 500 | 0.3646 | 0.3275 | +0.0371 [+0.0328, +0.0415] | +0.74 | 4e-46 | 2e-45 |
| test_id | ppo_entity_hard | rh_rk | weighted_score | 500 | 0.3646 | 0.3622 | +0.0024 [-0.0007, +0.0055] | -0.00 | 0.99 | 1 |
| test_id | ppo_entity_hard_lookahead | random | weighted_score | 500 | 0.3862 | 0.1042 | +0.2820 [+0.2736, +0.2903] | +1.00 | 1.3e-83 | 1e-82 |
| test_id | ppo_entity_hard_lookahead | greedy_value_rate | weighted_score | 500 | 0.3862 | 0.3645 | +0.0217 [+0.0189, +0.0245] | +0.70 | 2.6e-42 | 1.1e-41 |
| test_id | ppo_entity_hard_lookahead | greedy_nearest | weighted_score | 500 | 0.3862 | 0.3275 | +0.0587 [+0.0542, +0.0632] | +0.95 | 2.1e-76 | 1.3e-75 |
| test_id | ppo_entity_hard_lookahead | rh_rk | weighted_score | 500 | 0.3862 | 0.3622 | +0.0240 [+0.0209, +0.0272] | +0.70 | 1.5e-41 | 4.4e-41 |

## Generalization grid: paired ΔJ

| condition | A | B | metric | n | mean A | mean B | mean Δ [95 % CI] | r_rb | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| rate_x0 | ppo_entity_hard | greedy_value_rate | weighted_score | 100 | 0.5029 | 0.5060 | -0.0031 [-0.0080, +0.0017] | -0.11 | 0.32 | 1 |
| rate_x0 | ppo_entity_hard | greedy_nearest | weighted_score | 100 | 0.5029 | 0.4507 | +0.0521 [+0.0412, +0.0632] | +0.84 | 3.5e-13 | 2e-11 |
| rate_x0 | ppo_entity_hard | rh_rk | weighted_score | 100 | 0.5029 | 0.5077 | -0.0048 [-0.0109, +0.0016] | -0.27 | 0.019 | 0.87 |
| rate_x0 | ppo_entity_hard_lookahead | greedy_value_rate | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| rate_x0 | ppo_entity_hard_lookahead | greedy_nearest | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| rate_x0 | ppo_entity_hard_lookahead | rh_rk | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| rate_x2 | ppo_entity_hard | greedy_value_rate | weighted_score | 100 | 0.2390 | 0.2401 | -0.0011 [-0.0056, +0.0033] | -0.06 | 0.61 | 1 |
| rate_x2 | ppo_entity_hard | greedy_nearest | weighted_score | 100 | 0.2390 | 0.2189 | +0.0201 [+0.0132, +0.0271] | +0.59 | 2.5e-07 | 1.3e-05 |
| rate_x2 | ppo_entity_hard | rh_rk | weighted_score | 100 | 0.2390 | 0.2332 | +0.0059 [-0.0003, +0.0122] | +0.21 | 0.071 | 1 |
| rate_x2 | ppo_entity_hard_lookahead | greedy_value_rate | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| rate_x2 | ppo_entity_hard_lookahead | greedy_nearest | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| rate_x2 | ppo_entity_hard_lookahead | rh_rk | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| critical_50 | ppo_entity_hard | greedy_value_rate | weighted_score | 100 | 0.3744 | 0.3684 | +0.0060 [+0.0013, +0.0107] | +0.24 | 0.035 | 1 |
| critical_50 | ppo_entity_hard | greedy_nearest | weighted_score | 100 | 0.3744 | 0.3469 | +0.0275 [+0.0207, +0.0343] | +0.73 | 1.9e-10 | 1.1e-08 |
| critical_50 | ppo_entity_hard | rh_rk | weighted_score | 100 | 0.3744 | 0.3760 | -0.0016 [-0.0080, +0.0045] | +0.00 | 0.97 | 1 |
| critical_50 | ppo_entity_hard_lookahead | greedy_value_rate | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| critical_50 | ppo_entity_hard_lookahead | greedy_nearest | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| critical_50 | ppo_entity_hard_lookahead | rh_rk | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| critical_10 | ppo_entity_hard | greedy_value_rate | weighted_score | 100 | 0.4030 | 0.4046 | -0.0017 [-0.0071, +0.0037] | -0.02 | 0.87 | 1 |
| critical_10 | ppo_entity_hard | greedy_nearest | weighted_score | 100 | 0.4030 | 0.3784 | +0.0246 [+0.0149, +0.0350] | +0.47 | 4.8e-05 | 0.0024 |
| critical_10 | ppo_entity_hard | rh_rk | weighted_score | 100 | 0.4030 | 0.4059 | -0.0030 [-0.0092, +0.0036] | -0.20 | 0.078 | 1 |
| critical_10 | ppo_entity_hard_lookahead | greedy_value_rate | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| critical_10 | ppo_entity_hard_lookahead | greedy_nearest | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| critical_10 | ppo_entity_hard_lookahead | rh_rk | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| deadline_tight | ppo_entity_hard | greedy_value_rate | weighted_score | 100 | 0.3075 | 0.3088 | -0.0013 [-0.0066, +0.0037] | -0.06 | 0.63 | 1 |
| deadline_tight | ppo_entity_hard | greedy_nearest | weighted_score | 100 | 0.3075 | 0.2858 | +0.0218 [+0.0139, +0.0299] | +0.54 | 2.5e-06 | 0.00013 |
| deadline_tight | ppo_entity_hard | rh_rk | weighted_score | 100 | 0.3075 | 0.3053 | +0.0022 [-0.0049, +0.0095] | -0.02 | 0.88 | 1 |
| deadline_tight | ppo_entity_hard_lookahead | greedy_value_rate | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| deadline_tight | ppo_entity_hard_lookahead | greedy_nearest | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| deadline_tight | ppo_entity_hard_lookahead | rh_rk | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| radar_heavy | ppo_entity_hard | greedy_value_rate | weighted_score | 100 | 0.3437 | 0.3412 | +0.0025 [-0.0035, +0.0084] | +0.03 | 0.82 | 1 |
| radar_heavy | ppo_entity_hard | greedy_nearest | weighted_score | 100 | 0.3437 | 0.3123 | +0.0313 [+0.0229, +0.0399] | +0.68 | 3.1e-09 | 1.7e-07 |
| radar_heavy | ppo_entity_hard | rh_rk | weighted_score | 100 | 0.3437 | 0.3317 | +0.0120 [+0.0051, +0.0191] | +0.36 | 0.0019 | 0.094 |
| radar_heavy | ppo_entity_hard_lookahead | greedy_value_rate | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| radar_heavy | ppo_entity_hard_lookahead | greedy_nearest | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| radar_heavy | ppo_entity_hard_lookahead | rh_rk | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| no_requirements | ppo_entity_hard | greedy_value_rate | weighted_score | 100 | 0.4302 | 0.4253 | +0.0049 [+0.0001, +0.0098] | +0.14 | 0.21 | 1 |
| no_requirements | ppo_entity_hard | greedy_nearest | weighted_score | 100 | 0.4302 | 0.3912 | +0.0390 [+0.0286, +0.0500] | +0.72 | 5.1e-10 | 2.8e-08 |
| no_requirements | ppo_entity_hard | rh_rk | weighted_score | 100 | 0.4302 | 0.4368 | -0.0066 [-0.0120, -0.0008] | -0.47 | 4.8e-05 | 0.0024 |
| no_requirements | ppo_entity_hard_lookahead | greedy_value_rate | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| no_requirements | ppo_entity_hard_lookahead | greedy_nearest | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| no_requirements | ppo_entity_hard_lookahead | rh_rk | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| size_small | ppo_entity_hard | greedy_value_rate | weighted_score | 100 | 0.2897 | 0.2888 | +0.0009 [-0.0048, +0.0069] | -0.07 | 0.57 | 1 |
| size_small | ppo_entity_hard | greedy_nearest | weighted_score | 100 | 0.2897 | 0.2512 | +0.0384 [+0.0274, +0.0495] | +0.66 | 9.2e-09 | 5e-07 |
| size_small | ppo_entity_hard | rh_rk | weighted_score | 100 | 0.2897 | 0.2879 | +0.0018 [-0.0069, +0.0107] | -0.01 | 0.94 | 1 |
| size_small | ppo_entity_hard_lookahead | greedy_value_rate | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| size_small | ppo_entity_hard_lookahead | greedy_nearest | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| size_small | ppo_entity_hard_lookahead | rh_rk | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| size_large | ppo_entity_hard | greedy_value_rate | weighted_score | 100 | 0.4248 | 0.4214 | +0.0034 [-0.0015, +0.0083] | +0.19 | 0.093 | 1 |
| size_large | ppo_entity_hard | greedy_nearest | weighted_score | 100 | 0.4248 | 0.3863 | +0.0385 [+0.0302, +0.0470] | +0.84 | 2.9e-13 | 1.7e-11 |
| size_large | ppo_entity_hard | rh_rk | weighted_score | 100 | 0.4248 | 0.4213 | +0.0035 [-0.0036, +0.0108] | +0.06 | 0.59 | 1 |
| size_large | ppo_entity_hard_lookahead | greedy_value_rate | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| size_large | ppo_entity_hard_lookahead | greedy_nearest | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| size_large | ppo_entity_hard_lookahead | rh_rk | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| endurance_x0.6 | ppo_entity_hard | greedy_value_rate | weighted_score | 100 | 0.3494 | 0.3464 | +0.0031 [-0.0027, +0.0087] | +0.14 | 0.22 | 1 |
| endurance_x0.6 | ppo_entity_hard | greedy_nearest | weighted_score | 100 | 0.3494 | 0.3115 | +0.0380 [+0.0277, +0.0486] | +0.72 | 4.8e-10 | 2.7e-08 |
| endurance_x0.6 | ppo_entity_hard | rh_rk | weighted_score | 100 | 0.3494 | 0.3361 | +0.0133 [+0.0051, +0.0214] | +0.32 | 0.0052 | 0.25 |
| endurance_x0.6 | ppo_entity_hard_lookahead | greedy_value_rate | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| endurance_x0.6 | ppo_entity_hard_lookahead | greedy_nearest | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |
| endurance_x0.6 | ppo_entity_hard_lookahead | rh_rk | weighted_score | 0 | nan | nan | +0.0000 [+nan, +nan] | +0.00 | 1 | 1 |

### rate_x0

| method | seeds | scenarios | weighted_score | value_fraction | makespan | completion_rate |
|---|---|---|---|---|---|---|
| greedy_value_rate | – | 100 | 0.506 ± 0.116 [0.484, 0.529] | 0.506 ± 0.116 [0.484, 0.529] | 1071.674 ± 281.268 [1019.170, 1128.195] | 0.913 ± 0.065 [0.901, 0.926] |
| greedy_nearest | – | 100 | 0.451 ± 0.131 [0.426, 0.476] | 0.451 ± 0.131 [0.426, 0.476] | 971.559 ± 228.611 [928.602, 1017.229] | 0.882 ± 0.075 [0.867, 0.896] |
| rh_rk | – | 100 | 0.508 ± 0.120 [0.485, 0.531] | 0.508 ± 0.120 [0.485, 0.531] | 1029.678 ± 259.231 [981.590, 1081.056] | 0.914 ± 0.068 [0.901, 0.927] |
| ppo_entity_hard | 3 | 100 | 0.503 ± 0.002 [0.480, 0.526] | 0.503 ± 0.002 [0.480, 0.526] | 1044.836 ± 8.866 [995.040, 1096.553] | 0.911 ± 0.002 [0.898, 0.924] |

### rate_x2

| method | seeds | scenarios | weighted_score | value_fraction | makespan | completion_rate |
|---|---|---|---|---|---|---|
| greedy_value_rate | – | 100 | 0.240 ± 0.102 [0.220, 0.260] | 0.240 ± 0.102 [0.220, 0.260] | 1555.156 ± 222.772 [1510.993, 1597.494] | 0.636 ± 0.212 [0.594, 0.677] |
| greedy_nearest | – | 100 | 0.219 ± 0.105 [0.199, 0.239] | 0.219 ± 0.105 [0.199, 0.239] | 1456.149 ± 243.133 [1408.493, 1503.368] | 0.696 ± 0.171 [0.662, 0.729] |
| rh_rk | – | 100 | 0.233 ± 0.108 [0.212, 0.254] | 0.233 ± 0.108 [0.212, 0.254] | 1552.334 ± 223.168 [1508.098, 1595.853] | 0.633 ± 0.220 [0.589, 0.675] |
| ppo_entity_hard | 3 | 100 | 0.239 ± 0.000 [0.219, 0.258] | 0.239 ± 0.000 [0.219, 0.258] | 1521.593 ± 13.606 [1479.031, 1562.634] | 0.675 ± 0.002 [0.637, 0.713] |

### critical_50

| method | seeds | scenarios | weighted_score | value_fraction | makespan | completion_rate |
|---|---|---|---|---|---|---|
| greedy_value_rate | – | 100 | 0.368 ± 0.101 [0.348, 0.388] | 0.368 ± 0.101 [0.348, 0.388] | 1531.541 ± 214.857 [1489.920, 1574.529] | 0.783 ± 0.128 [0.756, 0.807] |
| greedy_nearest | – | 100 | 0.347 ± 0.110 [0.325, 0.368] | 0.347 ± 0.110 [0.325, 0.368] | 1429.220 ± 161.306 [1397.701, 1460.187] | 0.773 ± 0.102 [0.753, 0.792] |
| rh_rk | – | 100 | 0.376 ± 0.112 [0.354, 0.398] | 0.376 ± 0.112 [0.354, 0.398] | 1482.998 ± 208.661 [1443.154, 1523.454] | 0.796 ± 0.127 [0.771, 0.820] |
| ppo_entity_hard | 3 | 100 | 0.374 ± 0.001 [0.354, 0.394] | 0.374 ± 0.001 [0.354, 0.394] | 1512.090 ± 8.408 [1475.278, 1548.979] | 0.800 ± 0.004 [0.775, 0.822] |

### critical_10

| method | seeds | scenarios | weighted_score | value_fraction | makespan | completion_rate |
|---|---|---|---|---|---|---|
| greedy_value_rate | – | 100 | 0.405 ± 0.120 [0.381, 0.428] | 0.405 ± 0.120 [0.381, 0.428] | 1526.713 ± 225.360 [1482.271, 1569.868] | 0.880 ± 0.143 [0.851, 0.907] |
| greedy_nearest | – | 100 | 0.378 ± 0.125 [0.355, 0.403] | 0.378 ± 0.125 [0.355, 0.403] | 1509.776 ± 234.357 [1463.115, 1554.674] | 0.889 ± 0.126 [0.864, 0.913] |
| rh_rk | – | 100 | 0.406 ± 0.125 [0.382, 0.431] | 0.406 ± 0.125 [0.382, 0.431] | 1498.537 ± 232.311 [1452.682, 1543.003] | 0.886 ± 0.150 [0.856, 0.914] |
| ppo_entity_hard | 3 | 100 | 0.403 ± 0.003 [0.379, 0.427] | 0.403 ± 0.003 [0.379, 0.427] | 1525.686 ± 5.295 [1486.748, 1563.574] | 0.889 ± 0.002 [0.862, 0.914] |

### deadline_tight

| method | seeds | scenarios | weighted_score | value_fraction | makespan | completion_rate |
|---|---|---|---|---|---|---|
| greedy_value_rate | – | 100 | 0.309 ± 0.113 [0.287, 0.331] | 0.309 ± 0.113 [0.287, 0.331] | 1519.975 ± 218.952 [1478.218, 1561.949] | 0.778 ± 0.142 [0.749, 0.805] |
| greedy_nearest | – | 100 | 0.286 ± 0.115 [0.263, 0.308] | 0.286 ± 0.115 [0.263, 0.308] | 1440.294 ± 239.676 [1394.067, 1486.685] | 0.776 ± 0.127 [0.750, 0.800] |
| rh_rk | – | 100 | 0.305 ± 0.120 [0.282, 0.329] | 0.305 ± 0.120 [0.282, 0.329] | 1486.941 ± 215.844 [1444.842, 1530.491] | 0.778 ± 0.158 [0.746, 0.808] |
| ppo_entity_hard | 3 | 100 | 0.308 ± 0.001 [0.286, 0.329] | 0.308 ± 0.001 [0.286, 0.329] | 1497.661 ± 8.263 [1456.235, 1539.129] | 0.785 ± 0.001 [0.758, 0.809] |

### radar_heavy

| method | seeds | scenarios | weighted_score | value_fraction | makespan | completion_rate |
|---|---|---|---|---|---|---|
| greedy_value_rate | – | 100 | 0.341 ± 0.122 [0.317, 0.365] | 0.341 ± 0.122 [0.317, 0.365] | 1559.225 ± 283.112 [1503.826, 1615.366] | 0.744 ± 0.200 [0.704, 0.783] |
| greedy_nearest | – | 100 | 0.312 ± 0.120 [0.289, 0.336] | 0.312 ± 0.120 [0.289, 0.336] | 1504.784 ± 276.817 [1449.715, 1558.609] | 0.775 ± 0.161 [0.743, 0.806] |
| rh_rk | – | 100 | 0.332 ± 0.126 [0.307, 0.357] | 0.332 ± 0.126 [0.307, 0.357] | 1550.429 ± 282.166 [1495.125, 1604.662] | 0.750 ± 0.197 [0.710, 0.788] |
| ppo_entity_hard | 3 | 100 | 0.344 ± 0.004 [0.320, 0.367] | 0.344 ± 0.004 [0.320, 0.367] | 1535.256 ± 1.257 [1481.019, 1588.668] | 0.762 ± 0.005 [0.724, 0.798] |

### no_requirements

| method | seeds | scenarios | weighted_score | value_fraction | makespan | completion_rate |
|---|---|---|---|---|---|---|
| greedy_value_rate | – | 100 | 0.425 ± 0.099 [0.406, 0.445] | 0.425 ± 0.099 [0.406, 0.445] | 1471.066 ± 206.337 [1430.542, 1512.167] | 0.885 ± 0.121 [0.860, 0.907] |
| greedy_nearest | – | 100 | 0.391 ± 0.110 [0.369, 0.413] | 0.391 ± 0.110 [0.369, 0.413] | 1423.791 ± 167.082 [1391.936, 1456.988] | 0.883 ± 0.083 [0.866, 0.898] |
| rh_rk | – | 100 | 0.437 ± 0.104 [0.416, 0.457] | 0.437 ± 0.104 [0.416, 0.457] | 1455.747 ± 207.728 [1416.153, 1496.916] | 0.897 ± 0.113 [0.874, 0.918] |
| ppo_entity_hard | 3 | 100 | 0.430 ± 0.002 [0.410, 0.450] | 0.430 ± 0.002 [0.410, 0.450] | 1461.331 ± 12.154 [1427.300, 1497.453] | 0.898 ± 0.005 [0.878, 0.916] |

### size_small

| method | seeds | scenarios | weighted_score | value_fraction | makespan | completion_rate |
|---|---|---|---|---|---|---|
| greedy_value_rate | – | 100 | 0.289 ± 0.126 [0.264, 0.313] | 0.289 ± 0.126 [0.264, 0.313] | 1521.389 ± 233.619 [1476.228, 1567.741] | 0.620 ± 0.189 [0.583, 0.657] |
| greedy_nearest | – | 100 | 0.251 ± 0.118 [0.228, 0.274] | 0.251 ± 0.118 [0.228, 0.274] | 1416.037 ± 243.763 [1369.477, 1463.611] | 0.680 ± 0.165 [0.647, 0.712] |
| rh_rk | – | 100 | 0.288 ± 0.136 [0.261, 0.314] | 0.288 ± 0.136 [0.261, 0.314] | 1486.335 ± 241.050 [1439.117, 1532.833] | 0.647 ± 0.197 [0.608, 0.685] |
| ppo_entity_hard | 3 | 100 | 0.290 ± 0.001 [0.265, 0.314] | 0.290 ± 0.001 [0.265, 0.314] | 1490.463 ± 8.684 [1445.101, 1535.438] | 0.660 ± 0.007 [0.625, 0.694] |

### size_large

| method | seeds | scenarios | weighted_score | value_fraction | makespan | completion_rate |
|---|---|---|---|---|---|---|
| greedy_value_rate | – | 100 | 0.421 ± 0.105 [0.401, 0.441] | 0.421 ± 0.105 [0.401, 0.441] | 1463.537 ± 238.277 [1417.002, 1509.521] | 0.904 ± 0.071 [0.890, 0.918] |
| greedy_nearest | – | 100 | 0.386 ± 0.102 [0.366, 0.406] | 0.386 ± 0.102 [0.366, 0.406] | 1365.958 ± 218.367 [1323.722, 1408.867] | 0.880 ± 0.066 [0.867, 0.893] |
| rh_rk | – | 100 | 0.421 ± 0.107 [0.400, 0.442] | 0.421 ± 0.107 [0.400, 0.442] | 1418.434 ± 205.733 [1377.959, 1459.430] | 0.905 ± 0.073 [0.890, 0.919] |
| ppo_entity_hard | 3 | 100 | 0.425 ± 0.001 [0.404, 0.445] | 0.425 ± 0.001 [0.404, 0.445] | 1413.325 ± 8.617 [1376.565, 1451.330] | 0.909 ± 0.000 [0.896, 0.921] |

### endurance_x0.6

| method | seeds | scenarios | weighted_score | value_fraction | makespan | completion_rate |
|---|---|---|---|---|---|---|
| greedy_value_rate | – | 100 | 0.346 ± 0.107 [0.326, 0.367] | 0.346 ± 0.107 [0.326, 0.367] | 1236.742 ± 173.028 [1203.216, 1269.952] | 0.739 ± 0.149 [0.710, 0.767] |
| greedy_nearest | – | 100 | 0.311 ± 0.102 [0.292, 0.331] | 0.311 ± 0.102 [0.292, 0.331] | 1213.501 ± 184.033 [1177.294, 1248.560] | 0.757 ± 0.120 [0.734, 0.780] |
| rh_rk | – | 100 | 0.336 ± 0.120 [0.313, 0.359] | 0.336 ± 0.120 [0.313, 0.359] | 1219.017 ± 173.450 [1184.596, 1251.854] | 0.743 ± 0.168 [0.711, 0.775] |
| ppo_entity_hard | 3 | 100 | 0.349 ± 0.001 [0.329, 0.369] | 0.349 ± 0.001 [0.329, 0.369] | 1218.593 ± 1.081 [1186.717, 1250.450] | 0.757 ± 0.000 [0.730, 0.784] |
