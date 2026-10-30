# Saved ACT checkpoint transition comparison

Offline comparison only: no motor commands were sent and the deployed model was not changed.

Protocol: one existing heldout episode, 67 fixed observation frames per checkpoint, full 50-action predictions; 41 checkpoints compared. Scores are diagnostic errors, not task-success rates.

| Step | Critical composite score | Critical gripper MAE (units) | Uniform chunk normalized MAE |
|---:|---:|---:|---:|
| 5000 | 0.4129 | 5.797 | 0.1926 |
| 14000 | 0.3578 | 5.346 | 0.2022 |
| 17500 | 0.3680 | 5.300 | 0.2083 |
| 18500 | 0.3738 | 5.414 | 0.2153 |
| 19500 | 0.3755 | 5.490 | 0.2072 |
| 15500 | 0.3762 | 5.575 | 0.2030 |

The 14,000-step checkpoint has the lowest composite error, but its pickup-closing gripper error is slightly higher than the 5,000-step baseline. Review per-event tradeoffs in summary.json before choosing a physical evaluation candidate.

The 17,500-step checkpoint is the proposed physical-evaluation candidate: it was the only candidate with lower gripper MAE for both pickup closing (5.475 → 5.423) and release opening (6.120 → 5.177). The pickup difference is small and these values use the existing holdout episode. No physical success or independent generalization is established.
