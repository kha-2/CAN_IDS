# CAN Intrusion Detection System

This project focuses on developing a **deep learning-based Intrusion Detection System (IDS)** for detecting attacks on in-vehicle CAN networks.

The main objective is to build a CAN IDS that does not overly depend on vehicle-specific CAN IDs or payload patterns, so that the model can maintain detection performance across different vehicles.

---

## Research Objective

Many CAN IDS approaches learn patterns that are strongly tied to a specific vehicle, which can reduce performance when the model is applied to another vehicle.

To improve cross-vehicle generalization, this project uses temporal and statistical characteristics of CAN traffic rather than directly relying on specific CAN IDs or payload values.

The following traffic classes are considered:

- Normal
- DoS
- Fuzzing
- Replay
- Spoofing

---

## CAN Feature Extraction

CAN packets are transformed into temporal and statistical features before being used as model inputs.

The initial experiments used features such as:

- Global Inter-Arrival Time
- CAN ID Inter-Arrival Time
- Payload Entropy
- Hamming Distance
- DLC
- Payload Delta
- Jitter
- Payload Byte Mean
- Payload Byte Standard Deviation

CAN packets are grouped into sliding windows and used as sequential inputs to the IDS.

```text
CAN Packets
     ↓
Feature Extraction
     ↓
Sliding Window
     ↓
Deep Learning IDS
     ↓
Packet-level Attack Classification
```

---

## Repository Structure

```text
CAN_IDS/
├── encoding/       # Final feature encoding (CAN CSV -> 13-channel windows, .npz)
├── scaling/        # Final feature scaling (RobustScaler-style, fit on train only)
├── model/          # Final 2-stage TCN training + evaluation, weights/
├── data/mirgu/     # MIRGU window CSVs (zipped)
└── experiments/    # All earlier experiments, by date (see experiments/README.md)
```

The final version corresponds to `0308/FINAL` on the `history` branch.
All original branches (`history`, `J`, `S`, `p`) are kept as-is; their contents were copied into `experiments/`.

---

## Final Pipeline

| Step | File | Description |
|---|---|---|
| 1. Encoding | `encoding/car_challenge_encoding.ipynb` | Car Hacking Challenge CSV -> 13 packet-level features, window 128 / stride 64 |
| 1. Encoding | `encoding/carhacking_encoding.ipynb` | Same features for the test dataset (`Base_dir` selects Car-Hacking or Audi) |
| 2. Scaling | `scaling/robust_scaling.ipynb` | Per-channel `(x - median) / IQR`, clip to ±5, map to [0, 1]; statistics from train only |
| 3. Model | `model/Multi_TCN.ipynb` | 2-stage TCN training and evaluation |
| | `model/weights/TCN{1,2}_0305_653.pth` | Trained weights for the notebook above |

### Features (13 channels)

| Ch | Feature | Ch | Feature |
|---|---|---|---|
| 0 | is_cid0 (CAN ID == 0x000) | 7 | norm_by_win_mean |
| 1 | DLC / 8 | 8 | freq_over_top1_log |
| 2 | payload relative change (Hamming / per-ID EMA) | 9 | top1_share |
| 3 | entropy × relative change | 10 | idx_gap_cv |
| 4 | freq_local | 11 | dominance_ratio |
| 5 | id_ent (ID entropy in window) | 12 | same_id_recent_k_surprise |
| 6 | streak_ratio | | |

### Model (2-stage TCN)

- **Backbone**: TemporalConvNet, channels [32, 64, 128], dilation [1, 2, 4], kernel 3, causal convolutions with residual connections, 1×1 conv classifier per packet.
- **Stage 1 (TCN1)**: channels `[0,1,2,3,4,5,12]` -> {Other, DoS, Fuzzing}.
- **Stage 2 (TCN2)**: channels `[4,9]` -> {Normal, Spoofing}, trained only on Normal/Spoofing packets.
- **Fusion**: DoS if p(DoS) ≥ 0.9, Fuzzing if p(Fuzzing) ≥ th_fuzz; otherwise Stage 2 decides Normal vs Spoofing.
- Adam (lr 1e-4), ReduceLROnPlateau, early stopping, 20 epochs.

### Results (stored notebook output)

Train: Car Hacking Challenge `1_Submission`, test: Car Hacking Challenge `0_Training_test`.

| Class | Accuracy |
|---|---|
| Normal | 99.41% |
| DoS | 100.00% |
| Fuzzing | 99.87% |
| Spoofing | 32.75% |
| Replay | not predicted |

Overall accuracy 0.9787, macro F1 (present classes) 0.6337.

### Known Issues

- Replay has no output class in the 2-stage model.
- `oversample_attack_windows` is called, but the training loader uses the original split.
- `FEATURE_NAMES` in the encoding notebooks still lists old feature names.
- File paths are hard-coded (`C:/Users/user/Desktop/IDS_masters/...`, `D:/IDS_masters/...`); the file name written by `robust_scaling.ipynb` differs from the one read by `Multi_TCN.ipynb`.
