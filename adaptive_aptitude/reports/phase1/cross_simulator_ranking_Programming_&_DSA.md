# Cross-simulator ranking agreement -- Programming & DSA

| algorithm              |   roc_auc_bkt_generative |   calibration_error_bkt_generative |   rank_bkt_generative |   roc_auc_irt_generative |   calibration_error_irt_generative |   rank_irt_generative | agreement   |
|:-----------------------|-------------------------:|-----------------------------------:|----------------------:|-------------------------:|-----------------------------------:|----------------------:|:------------|
| bkt                    |                   0.5052 |                             0.0278 |                     3 |                   0.4914 |                             0.2259 |                     5 | NO -- flips |
| bkt_ema_epsilon_greedy |                   0.5138 |                             0.0211 |                     1 |                   0.4977 |                             0.2217 |                     4 | NO -- flips |
| ema_only               |                   0.5061 |                             0.0911 |                     2 |                   0.502  |                             0.109  |                     3 | close       |
| random                 |                   0.4899 |                             0.0008 |                     5 |                   0.516  |                             0.0337 |                     1 | NO -- flips |
| rule_weakest_topic     |                   0.4918 |                             0.0019 |                     4 |                   0.5158 |                             0.0332 |                     2 | NO -- flips |

'NO -- flips' means this algorithm's relative ranking changed meaningfully between the BKT-shaped ground truth and the IRT-shaped ground truth -- its apparent advantage under one may be an artifact of matching that simulator's assumptions rather than a real property of the algorithm.
