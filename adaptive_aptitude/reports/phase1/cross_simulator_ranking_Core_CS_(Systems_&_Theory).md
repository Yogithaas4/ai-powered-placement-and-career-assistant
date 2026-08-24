# Cross-simulator ranking agreement -- Core CS (Systems & Theory)

| algorithm              |   roc_auc_bkt_generative |   calibration_error_bkt_generative |   rank_bkt_generative |   roc_auc_irt_generative |   calibration_error_irt_generative |   rank_irt_generative | agreement   |
|:-----------------------|-------------------------:|-----------------------------------:|----------------------:|-------------------------:|-----------------------------------:|----------------------:|:------------|
| bkt                    |                   0.5017 |                             0.0348 |                     3 |                   0.4976 |                             0.2282 |                     5 | NO -- flips |
| bkt_ema_epsilon_greedy |                   0.506  |                             0.0366 |                     1 |                   0.5017 |                             0.2225 |                     3 | NO -- flips |
| ema_only               |                   0.502  |                             0.0832 |                     2 |                   0.5003 |                             0.1045 |                     4 | NO -- flips |
| random                 |                   0.4986 |                             0.0052 |                     4 |                   0.5137 |                             0.0301 |                     1 | NO -- flips |
| rule_weakest_topic     |                   0.4943 |                             0.0027 |                     5 |                   0.5135 |                             0.0306 |                     2 | NO -- flips |

'NO -- flips' means this algorithm's relative ranking changed meaningfully between the BKT-shaped ground truth and the IRT-shaped ground truth -- its apparent advantage under one may be an artifact of matching that simulator's assumptions rather than a real property of the algorithm.
