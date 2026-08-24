# Cross-simulator ranking agreement -- Aptitude

| algorithm              |   roc_auc_bkt_generative |   calibration_error_bkt_generative |   rank_bkt_generative |   roc_auc_irt_generative |   calibration_error_irt_generative |   rank_irt_generative | agreement   |
|:-----------------------|-------------------------:|-----------------------------------:|----------------------:|-------------------------:|-----------------------------------:|----------------------:|:------------|
| bkt                    |                   0.5194 |                             0.0777 |                     1 |                   0.5035 |                             0.118  |                     4 | NO -- flips |
| bkt_ema_epsilon_greedy |                   0.5157 |                             0.0829 |                     2 |                   0.5086 |                             0.1148 |                     3 | close       |
| ema_only               |                   0.5133 |                             0.0921 |                     3 |                   0.503  |                             0.1078 |                     5 | NO -- flips |
| random                 |                   0.4992 |                             0.0041 |                     4 |                   0.5183 |                             0.0359 |                     1 | NO -- flips |
| rule_weakest_topic     |                   0.4965 |                             0.0038 |                     5 |                   0.5136 |                             0.0325 |                     2 | NO -- flips |

'NO -- flips' means this algorithm's relative ranking changed meaningfully between the BKT-shaped ground truth and the IRT-shaped ground truth -- its apparent advantage under one may be an artifact of matching that simulator's assumptions rather than a real property of the algorithm.
