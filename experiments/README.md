# Experiments

Earlier experiments, ordered by date. Files were copied from the original branches (`history`, `J`, `S` are kept; `p` was deleted after copying).
Unless noted, training data is Car Hacking Challenge (`0_Preliminary/1_Submission`) and test data is the Car-Hacking Dataset (cross-dataset).

| Folder | Source | Summary |
|---|---|---|
| `2026-01-26_mirgu_xai` | main | Attention + CNN with attention-based XAI on MIRGU windows; `.pt` -> CSV export |
| `2026-01-29_cnn_baseline` | J | 9-feature encoding (window 64 / stride 32), causal CNN baseline, evaluation |
| `2026-01-30_cnn_dropout` | main, S | CNN + BatchNorm + Dropout, oversampling encoding, packet-level evaluation |
| `2026-02-02_cnn_attention_tcn` | J | CNN + DualStageAttention, first TCN, 5-channel TCN, notes |
| `2026-02-03_tcn_5ch_multicnn` | p | 5-feature encoding, TCN [32,64,128], first 2-stage CNN |
| `2026-02-04_6feat_tcn_multicnn` | history | 6 features (Is_Zero, Complexity, Hamming rate, Frequency), TCN, 2-stage CNN |
| `2026-02-09_markov_zscore_dualtcn` | history | Markov transition score, entropy z-score, 9-feature set, first Dual TCN |
| `2026-02-10_162feat_dualtcn` | history | 162 candidate features -> 30 selected, Dual TCN variants |
| `2026-02-11_spoofing_tuning` | history | 2-stage structure fixed; Spoofing 80% / 83%; 15-feature window-128 runs (`jaeho*`) |
| `2026-02-12_anchor_temporal` | history | Anchor IAT, temporal encoding, 12-feature runs (spoofing97 is evaluated on training-source data) |
| `2026-02-23_basic13feat` | history | Redesigned 13 features (dominance, top1 share, streak), window 256 |
| `2026-02-24_stage2_feat` | history | Stage-2 feature selection |
| `2026-02-26_12feat_audi` | history | 12 features, window 128; Audi test macro F1 0.914 |
| `2026-03-05_robust_scaling` | history | Robust scaling introduced; feature distribution check; result logs |
| `2026-03-08_threshold_tuning` | history | Fuzzing threshold sweep, dropout variants; `FINAL/` = 13-feature pipeline evaluated on Car Hacking Challenge `0_Training_test` |
| `2026-03-13_real_crossdataset` | history | Permutation feature importance on the final model, single-TCN comparison (the final pipeline itself, `0313/REAL`, is in the repository root) |

## Renamed files

| Original | New |
|---|---|
| `0130/찐최종최종최종 (1) (1).ipynb` | `2026-01-30_cnn_dropout/train_seqids_cnn.ipynb` |
| `advanced_model_evaluation - 복사본.ipynb` (J) | `2026-01-29_cnn_baseline/advanced_model_evaluation.ipynb` |
| `찐막.ipynb` (J) | `2026-02-02_cnn_attention_tcn/tcn_5channel.ipynb` |
| `정리`, `피쳐 정리` (J) | `notes_tcn.txt`, `notes_feature_v2.txt` |
| `TCN_Attention_model_윤주가고침.ipynb` (p) | `2026-02-03_tcn_5ch_multicnn/TCN_Attention_model_valfix.ipynb` |
| `Multi_TCN.ipynb` (history root) | `2026-02-09_markov_zscore_dualtcn/Multi_TCN_initial.ipynb` |
| `0204/새 폴더/` | `2026-02-04_6feat_tcn_multicnn/scripts/` |
| `0204/encoding`, `0209/entropy_zscore` | `encoding_snippet.py`, `entropy_zscore.py` |
| `0226/0226` | `2026-02-26_12feat_audi/NOTE.txt` |
| `0305/dataset_name/name` | `2026-03-05_robust_scaling/dataset_paths.txt` |
| `0305/result/윤주(*)_0226_938` | `2026-03-05_robust_scaling/result/yoonju_*_0226_938.txt` |

Empty placeholder files and byte-identical duplicates were not copied.
