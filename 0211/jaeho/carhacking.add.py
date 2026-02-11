import os
import pandas as pd
import numpy as np
import joblib
from numba import njit, float64, int64, uint64, types
from numba.typed import Dict
from tqdm import tqdm

# =========================================================
# 1. 설정 (Testing: BMW)
# =========================================================
# [경로 수정] 본인의 BMW 데이터 경로로 수정하세요
BASE_DIR = "C:/Users/user/Desktop/IDS_masters/9) Car-Hacking Dataset"
OUTPUT_DIR = "C:/Users/user/Desktop/IDS_masters/features"

# 기존 스케일러는 30개 피처용이므로 15개 피처에는 사용할 수 없어 주석 처리했습니다.
# SCALER_PATH = "C:/Users/user/Desktop/IDS_masters/features/robust_scaler_30feat.pkl" 

ATTACK_DIR = {"Dos": 1, "Fuzzing": 2, "Spoofing": 4}

WINDOW_SIZE, STRIDE = 128, 64

# 요청하신 15개 피처 이름
FEATURE_NAMES_15 = [
    "IAT", "Is_Zero", "Payload_Ent", "Complexity", 
    "Ham_Rate", "Freq", "Continuity", "Diff_Ent", "ID_Ent",
    "Freq_Slope", "Jit_Slope", "Ent_Slope", 
    "Freq_Fast_Z", "Jit_Fast_Z", "Bus_Load"
]

# =========================================================
# 2. Numba 엔진 (15개 피처 로직 이식)
# =========================================================
@njit
def popcount64(x):
    c = 0
    v = int64(x)
    while v:
        v &= v - int64(1)
        c += 1
    return c

@njit
def pack_payload_u64(row):
    v = uint64(0)
    for i in range(8):
        v |= uint64(row[i]) << (i * 8)
    return v

@njit
def update_ema_z(val, cid, ema_map, sq_ema_map, alpha):
    """
    EMA 기반 Z-Score 계산 (신규 로직)
    """
    if cid not in ema_map:
        ema_map[cid] = float64(val)
        sq_ema_map[cid] = float64(val ** 2)
        return 0.0

    mean = ema_map[cid]
    sq_mean = sq_ema_map[cid]
    
    var = sq_mean - (mean ** 2)
    if var < 0: var = 0.0
    std = np.sqrt(var)

    z = 0.0
    if std > 1e-9:
        z = (val - mean) / std
        if z > 5.0: z = 5.0
        elif z < -5.0: z = -5.0

    ema_map[cid] = (1.0 - alpha) * mean + alpha * val
    sq_ema_map[cid] = (1.0 - alpha) * sq_mean + alpha * (val ** 2)
    
    return z

@njit(fastmath=True)
def calculate_features_15_numba(timestamps, can_ids, payloads, dlcs):
    # dlcs는 서명 호환성을 위해 받지만 내부 계산에는 사용하지 않음
    n = len(timestamps)
    features = np.zeros((n, 15), dtype=np.float64)
    
    # --- [기존 변수들] ---
    last_time_map = Dict.empty(key_type=types.int64, value_type=types.float64)
    last_payload_map = Dict.empty(key_type=types.int64, value_type=types.uint64)
    last_id_map = Dict.empty(key_type=types.int64, value_type=types.float64)
    id_ham_ema = Dict.empty(key_type=types.int64, value_type=types.float64)
    alpha_ham = 0.05
    eps = 1e-9

    # --- [신규 변수들] ---
    last_freq_map = Dict.empty(key_type=types.int64, value_type=types.float64)
    last_jit_map_val = Dict.empty(key_type=types.int64, value_type=types.float64)
    last_ent_map = Dict.empty(key_type=types.int64, value_type=types.float64)
    last_iat_map = Dict.empty(key_type=types.int64, value_type=types.float64)
    
    # Z-Score용 EMA 맵 (Alpha=0.1 Fast)
    ema_freq = Dict.empty(key_type=types.int64, value_type=types.float64)
    sq_ema_freq = Dict.empty(key_type=types.int64, value_type=types.float64)
    ema_jit = Dict.empty(key_type=types.int64, value_type=types.float64)
    sq_ema_jit = Dict.empty(key_type=types.int64, value_type=types.float64)
    
    prev_global_time = timestamps[0]

    for i in range(n):
        # 64개마다 윈도우 빈도 초기화
        if (i % 64) == 0:
            last_id_map.clear()
            
        ts = timestamps[i]
        cid = can_ids[i]
        row = payloads[i]
        
        if np.isnan(ts): ts = prev_global_time
        
        # --- [공통 물리량 계산] ---
        # 1. IAT
        curr_iat = 0.0
        if cid in last_time_map:
            curr_iat = max(0.0, ts - last_time_map[cid])
        else:
            curr_iat = 0.001
        
        # 2. Frequency (1/IAT)
        curr_freq = 0.0
        if curr_iat > 1e-9:
            curr_freq = 1.0 / curr_iat
            
        # 3. Jitter (|Current IAT - Last IAT|)
        curr_jit = 0.0
        if cid in last_iat_map:
            curr_jit = np.abs(curr_iat - last_iat_map[cid])
        last_iat_map[cid] = curr_iat
        
        # --- [Part 1] 기존 9개 피처 ---
        cur_bytes = pack_payload_u64(row)
        rel_change = 0.0
        if cid in last_payload_map:
            diff = cur_bytes ^ last_payload_map[cid]
            h_dist = float64(popcount64(diff))
            if cid in id_ham_ema:
                avg_h = id_ham_ema[cid]
                rel_change = h_dist / (avg_h + 0.1) 
                id_ham_ema[cid] = (1.0 - alpha_ham) * avg_h + alpha_ham * h_dist
            else:
                rel_change = 1.0
                id_ham_ema[cid] = h_dist
        else:
            rel_change = 0.0
        last_payload_map[cid] = cur_bytes

        p_counts = np.zeros(256, dtype=np.int64)
        for b in row: p_counts[b] += 1
        ent = 0.0
        for c in p_counts:
            if c > 0:
                p = c / 8.0
                ent -= p * np.log(p)

        features[i, 0] = np.log1p(curr_iat * 1000.0) / 7.0  # 1. IAT
        features[i, 1] = 1.0 if cid == 0 else 0.0           # 2. Is_Zero
        features[i, 2] = ent / 2.1                          # 3. Payload_Ent
        features[i, 3] = np.log1p(ent * rel_change)         # 4. Complexity
        features[i, 4] = np.log1p(rel_change / (curr_iat + eps)) / 10.0 # 5. Ham_Rate
        
        cnt = last_id_map.get(cid, 0.0) + 1.0
        last_id_map[cid] = cnt
        features[i, 5] = cnt / 128.0                        # 6. Freq (Local)

        features[i, 6] = np.log1p(rel_change) / 5.0         # 7. Continuity

        diffs = np.zeros(7, dtype=np.int64)
        for b_idx in range(7):
            diffs[b_idx] = (int64(row[b_idx+1]) - int64(row[b_idx])) % 256
        d_counts = Dict.empty(key_type=types.int64, value_type=types.float64)
        for d in diffs: d_counts[d] = d_counts.get(d, 0.0) + 1.0
        d_ent = 0.0
        for dv in d_counts:
            p = d_counts[dv] / 7.0
            d_ent -= p * np.log(p + 1e-9)
        features[i, 7] = d_ent / 1.94                          # 8. Diff_Ent

        if i >= 127:
            win_id_counts = Dict.empty(key_type=types.int64, value_type=types.float64)
            for j in range(i-127, i+1):
                wid = can_ids[j]
                win_id_counts[wid] = win_id_counts.get(wid, 0.0) + 1.0
            wi_ent = 0.0
            for k_id in win_id_counts:
                pk = win_id_counts[k_id] / 128.0
                wi_ent -= pk * np.log(pk + 1e-9)
            features[i, 8] = wi_ent / 4.85                      # 9. ID_Ent
        else:
            features[i, 8] = 0.0

        # --- [Part 2] 신규 6개 피처 ---
        
        # 10. Freq_Slope
        if cid in last_freq_map:
            features[i, 9] = curr_freq - last_freq_map[cid]
        else:
            features[i, 9] = 0.0
        last_freq_map[cid] = curr_freq

        # 11. Jit_Slope
        if cid in last_jit_map_val:
            features[i, 10] = curr_jit - last_jit_map_val[cid]
        else:
            features[i, 10] = 0.0
        last_jit_map_val[cid] = curr_jit

        # 12. Ent_Slope
        if cid in last_ent_map:
            features[i, 11] = ent - last_ent_map[cid]
        else:
            features[i, 11] = 0.0
        last_ent_map[cid] = ent

        # 13. Freq_Fast_Z
        features[i, 12] = update_ema_z(curr_freq, cid, ema_freq, sq_ema_freq, 0.1)

        # 14. Jit_Fast_Z
        features[i, 13] = update_ema_z(curr_jit, cid, ema_jit, sq_ema_jit, 0.1)

        # 15. Bus_Load
        global_iat = ts - prev_global_time if i > 0 else 0.001
        features[i, 14] = 1.0 / (global_iat + 1e-9)
        
        prev_global_time = ts
        
        # 상태 업데이트
        last_time_map[cid] = ts

    return features

# =========================================================
# 3. 데이터 로딩 및 실행
# =========================================================
def parse_payload_str(data_tokens, dlc):
    payload = [int(tok.strip(), 16) if tok.strip() else 0 for tok in data_tokens[:dlc]]
    return (payload + [0]*8)[:8]

def process_csv_file_to_numpy(path, attack_id):
    rows_ts, rows_cid, rows_pay, rows_dlc, rows_lbl = [], [], [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4: continue
            try:
                ts = float(parts[0])
                cid = int(parts[1].strip(), 16)
                dlc = int(parts[2])
                payload = parse_payload_str(parts[3:-1], dlc)
                label = 1 if parts[-1].strip() == "T" else 0
                rows_ts.append(ts); rows_cid.append(cid); rows_dlc.append(dlc)
                rows_pay.append(payload)
                rows_lbl.append(label * attack_id if label else 0)
            except: continue
    return np.array(rows_ts), np.array(rows_cid), np.array(rows_pay, dtype=np.uint8), np.array(rows_dlc), np.array(rows_lbl)

def process_test():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # 통합 저장용 리스트
    all_X_list = []
    all_y_list = []
    
    # CSV용 리스트 (메모리 주의)
    all_features_list = []
    all_labels_list = []

    print(f"🚀 피처 추출 시작 (Mode: 15 Features)")
    print(f"⚠️ 주의: 스케일러가 적용되지 않습니다. (15 피처 호환성 문제)")

    for folder, attack_id in ATTACK_DIR.items():
        fpath = os.path.join(BASE_DIR, folder)
        if not os.path.exists(fpath): continue
        for fname in os.listdir(fpath):
            if not fname.endswith(".csv"): continue
            print(f"📂 BMW {fname} 처리 중...")
            
            ts, cid, pay, dlc, lbl = process_csv_file_to_numpy(os.path.join(fpath, fname), attack_id)
            
            # [변경됨] 15개 피처 추출 함수 호출
            feat_15 = calculate_features_15_numba(ts, cid, pay, dlc)
            
            # [CSV 통합용] 리스트 추가
            all_features_list.append(feat_15)
            all_labels_list.append(lbl)

            # [NPZ 통합용] 윈도우 변환 및 리스트 추가
            # TCN 학습을 위해 (Length, Feat) -> (Feat, Length)로 Transpose하여 저장
            for start in tqdm(range(0, len(feat_15) - WINDOW_SIZE + 1, STRIDE), desc="Windowing"):
                all_X_list.append(feat_15[start:start+WINDOW_SIZE].T)
                all_y_list.append(lbl[start:start+WINDOW_SIZE])

    # -----------------------------------------------------
    # 🏁 1. 통합 CSV 저장
    # -----------------------------------------------------
    if all_features_list:
        print("\n📦 통합 CSV 생성 중 (BMW/Test/15Feat)...")
        final_X_csv = np.vstack(all_features_list)
        final_y_csv = np.concatenate(all_labels_list)
        
        df_final = pd.DataFrame(final_X_csv, columns=FEATURE_NAMES_15)
        df_final['Label'] = final_y_csv
        
        csv_path = os.path.join(OUTPUT_DIR, "BMW_15feat_combined.csv")
        df_final.to_csv(csv_path, index=False)
        print(f"✅ 통합 CSV 저장 완료: {csv_path}")

    # -----------------------------------------------------
    # 🏁 2. 통합 NPZ 저장
    # -----------------------------------------------------
    if all_X_list:
        print("\n📦 통합 NPZ 생성 중 (BMW/Test/15Feat)...")
        final_X_npz = np.array(all_X_list, dtype=np.float32)
        final_y_npz = np.array(all_y_list, dtype=np.int64)
        
        npz_path = os.path.join(OUTPUT_DIR, "BMW_15feat_combined.npz")
        np.savez(npz_path, X=final_X_npz, y=final_y_npz)
        print(f"✅ 통합 NPZ 저장 완료: {npz_path}")
        print(f"   Shape: {final_X_npz.shape}")

if __name__ == "__main__":
    process_test()
    print("\n✅ 모든 작업 완료!")
