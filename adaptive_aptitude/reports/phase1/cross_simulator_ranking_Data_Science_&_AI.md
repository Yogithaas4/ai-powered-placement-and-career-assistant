# Cross-simulator ranking agreement -- Data Science & AI

| algorithm              |   roc_auc_bkt_generative |   calibration_error_bkt_generative |   rank_bkt_generative |   roc_auc_irt_generative |   calibration_error_irt_generative |   rank_irt_generative | agreement   |
|:-----------------------|-------------------------:|-----------------------------------:|----------------------:|-------------------------:|-----------------------------------:|----------------------:|:------------|
| bkt                    |                   0.5477 |                             0.002  |                     1 |                   0.5264 |                             0.211  |                     1 | yes         |
| bkt_ema_epsilon_greedy |                   0.5456 |                             0.0014 |                     2 |                   0.5248 |                             0.2091 |                     2 | yes         |
| ema_only               |                   0.5434 |                             0.09   |                     3 |                   0.5186 |                             0.1272 |                     3 | yes         |
| random                 |                   0.4924 |                             0.0016 |                     4 |                   0.5137 |                             0.0337 |                     5 | close       |
| rule_weakest_topic     |                   0.4905 |                             0.0019 |                     5 |                   0.5145 |                             0.0341 |                     4 | close       |

'NO -- flips' means this algorithm's relative ranking changed meaningfully between the BKT-shaped ground truth and the IRT-shaped ground truth -- its apparent advantage under one may be an artifact of matching that simulator's assumptions rather than a real property of the algorithm.
