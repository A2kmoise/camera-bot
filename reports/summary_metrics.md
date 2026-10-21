# Person Tracker Evaluation & Parameter Sweep Summary

| total_frames | fps_mean | fps_p5 | fps_median | time_confirmed_pct | time_temporary_pct | time_unknown_pct | centring_mae | centring_rms | centring_max | continuity_rate | correct_identity_rate | id_switch_count | false_temporary_rate | unknown_when_should_know_rate | mean_recovery_time_s | t_half | floor | session |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- | :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| 150 | 12.188 | 5.820 | 14.155 | 32.667 | 0.667 | 66.667 | 0.025 | 0.048 | 0.118 | 0.333 | 0.333 | 33 | 0.000 | 0.456 | 0.167 | 1.000 | 0.300 | sweep_sample_replay_th1.0_fl0.30 |
| 150 | 13.905 | 6.723 | 14.875 | 32.667 | 0.667 | 66.667 | 0.025 | 0.048 | 0.118 | 0.333 | 0.333 | 33 | 0.000 | 0.456 | 0.167 | 2.000 | 0.300 | sweep_sample_replay_th2.0_fl0.30 |
| 150 | 16.520 | 13.644 | 15.920 | 32.667 | 0.667 | 66.667 | 0.025 | 0.048 | 0.118 | 0.333 | 0.333 | 33 | 0.000 | 0.456 | 0.167 | 5.000 | 0.300 | sweep_sample_replay_th5.0_fl0.30 |
