# Cross-simulator ranking agreement -- Engineering Mathematics

| algorithm              |   roc_auc_bkt_generative |   calibration_error_bkt_generative |   rank_bkt_generative |   roc_auc_irt_generative |   calibration_error_irt_generative |   rank_irt_generative | agreement   |
|:-----------------------|-------------------------:|-----------------------------------:|----------------------:|-------------------------:|-----------------------------------:|----------------------:|:------------|
| bkt                    |                   0.5318 |                             0.1043 |                     1 |                   0.5115 |                             0.2734 |                     3 | NO -- flips |
| bkt_ema_epsilon_greedy |                   0.5273 |                             0.1007 |                     2 |                   0.5088 |                             0.2819 |                     4 | NO -- flips |
| ema_only               |                   0.5186 |                             0.0863 |                     3 |                   0.5048 |                             0.0904 |                     5 | NO -- flips |
| random                 |                   0.4953 |                             0.001  |                     5 |                   0.5149 |                             0.032  |                     1 | NO -- flips |
| rule_weakest_topic     |                   0.497  |                             0.0027 |                     4 |                   0.5134 |                             0.0293 |                     2 | NO -- flips |

'NO -- flips' means this algorithm's relative ranking changed meaningfully between the BKT-shaped ground truth and the IRT-shaped ground truth -- its apparent advantage under one may be an artifact of matching that simulator's assumptions rather than a real property of the algorithm.
