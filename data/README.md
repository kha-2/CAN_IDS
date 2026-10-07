# Data

## MIRGU (`mirgu/`)

Windowed feature tables exported from the MIRGU training tensor
(`experiments/2026-01-26_mirgu_xai/visualization_mirgu_dataset.ipynb`), split by attack type.

| Folder | File | Contents |
|---|---|---|
| `dos/` | `windows_dos.zip` | `windows_dos.csv` (35 MB unzipped) |
| `fuzzing/` | `windows_fuzzing.zip` | `windows_fuzzing.csv` (76 MB unzipped) |

Each row is one packet inside a 64-packet window:

```text
window_idx, t, Global IAT, ID IAT, Entropy, Hamming, DLC, Delta Payload, Jitter, Byte Mean, Byte Std, window_y, window_label
```

`window_y`: 0 Normal, 1 DoS, 2 Fuzzing, 3 Replay, 4 Spoofing.

## Other datasets

The final pipeline uses the following datasets, which are not stored in this repository:

- Car Hacking Challenge 2021 (`0_Preliminary/1_Submission`): training
- Car-Hacking Dataset: cross-dataset test
