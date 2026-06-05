# v21 Feature Pruning — Ablation Study (2026-06-05)

## Champion v19 Feature Importances (from champion.pkl)

Extracted via `model.calibrated_classifiers_[0].estimator.feature_importances_`:

```
 #  Feature                              Imp    %      Cumul%
 1. btc_inslot_ret                       1187  12.3%   12.3%
 2. x_ob_drift_x_inslot                   749   7.8%   20.1%
 3. btc_pre_5m_ret                         606   6.3%   26.3%
 4. btc_vwap_up                            572   5.9%   32.3%
 5. ob_mid_drift                           426   4.4%   36.7%
 6. btc_up_w1                              384   4.0%   40.7%
 7. btc_vwap_dn                            384   4.0%   44.7%
 8. btc_vwap_spread                        333   3.5%   48.1%
 9. btc_up_w2                              277   2.9%   51.0%
10. btc_pre_30m_ret                        236   2.4%   53.4%
--- top 10 = 53.4% cumulative ---
11. prev_slot_up_ratio_5                   205   2.1%   55.5%
12. hour_x_up_ratio                        201   2.1%   57.6%
13. ob_weighted_imb                        194   2.0%   59.6%
14. btc_pre_1h_ret                         194   2.0%   61.7%
15. ob_imb_w0                              190   2.0%   63.6%
16. btc_signal_conviction                  188   1.9%   65.6%
17. ob_mid                                 186   1.9%   67.5%
18. ob_ask_depth_5c                        180   1.9%   69.4%
19. ob_imb_w2                              175   1.8%   71.2%
20. btc_momentum                           174   1.8%   73.0%
--- top 20 = 73.0% cumulative ---
21. prev_slot_up_ratio_2                   172   1.8%   74.8%
22. btc_up_ratio                           170   1.8%   76.5%
23. x_depth_x_momentum                     154   1.6%   78.1%
24. btc_pre_4h_ret                         153   1.6%   79.7%
25. prev_slot_up_ratio_1                   152   1.6%   81.3%
26. btc_size_disparity                     149   1.5%   82.8%
27. prev_slot_up_ratio_3                   148   1.5%   84.4%
28. ob_imb_momentum                        146   1.5%   85.9%
29. x_imb_x_ur                            141   1.5%   87.3%
30. hour_cos                               132   1.4%   88.7%
--- top 30 = 88.7% cumulative ---
31. btc_buy_ratio                          128   1.3%   90.0%
32. ob_imb_w1                              126   1.3%   91.3%
33. hour_x_tw_ur                           119   1.2%   92.6%
34. btc_dist_1k                            118   1.2%   93.8%
35. prev_slot_up_ratio_4                   118   1.2%   95.0%
36. btc_up_w0                              115   1.2%   96.2%
37. btc_pre_1h_4h_ratio                    112   1.2%   97.4%
38. btc_up_ratio_zscore_20s                 97   1.0%   98.4%
39. btc_up_ratio_zscore_5s                  93   1.0%   99.3%
40. ob_total_depth                          64   0.7%  100.0%
```

## Pruning Rationale

### PRUNE_5 (40 → 35 features)
| Feature | Importance | Reason |
|---------|-----------|--------|
| ob_total_depth | 0.7% | Absolute value, leaks market-specific info |
| btc_up_ratio_zscore_5s | 1.0% | Noisy short-window zscore |
| btc_up_ratio_zscore_20s | 1.0% | Needs warm history, marginal |
| btc_pre_1h_4h_ratio | 1.2% | Complex ratio, cold buffer issues |
| btc_up_w0 | 1.2% | Earliest 30s window = most noise |

### PRUNE_10 (40 → 30 features)
All of PRUNE_5 plus:
| Feature | Importance | Reason |
|---------|-----------|--------|
| prev_slot_up_ratio_4 | 1.2% | 4 slots back, likely noise |
| btc_dist_1k | 1.2% | Weak round-number signal |
| hour_x_tw_ur | 1.2% | Interaction, overfit risk |
| ob_imb_w1 | 1.3% | INTERPOLATED in live (not real measurement) |
| hour_cos | 1.4% | Temporal, overfit risk on small dataset |

## v21 Training Script Design

- `scripts/train_v21_modal.py`
- Tests 3 variants: 40feat (baseline), 35feat, 30feat
- Shared Optuna tuning (optimized on 40-feature set)
- Independent walk-forward evaluation for each variant
- Picks best variant that beats champion v19 (2/3 gate)
- Ties broken by fewer features (simpler model preferred)
- Saves ablation results in `champion_meta.json`

## Key Insight

Top 30 features capture 88.7% of total importance.
Bottom 10 features contribute only 11.3% total but add:
- Overfit risk (more params, same data)
- Live data quality issues (interpolation, cold buffers)
- Computational overhead (more features to build each tick)
