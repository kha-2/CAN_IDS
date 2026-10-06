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
