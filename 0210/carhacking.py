import os
import pandas as pd
import numpy as np
import joblib
from numba import njit, float32, int64, float64, uint64, types
from numba.typed import Dict
from tqdm import tqdm

# =========================================================
# 1. 설정 (Testing: BMW)
# =========================================================
# [경로 수정] 본인의 BMW 데이터 경로로 수정하세요
BASE_DIR = "D:/IDS_masters/dataset/9) Car-Hacking Dataset"
OUTPUT_DIR = "D:/IDS_masters/features/challenge/final_30"
SCALER_PATH = "D:/IDS_masters/features/robust_scaler_30feat.pkl"
ATTACK_DIR = {"Dos": 1, "Fuzzing": 2, "Spoofing": 4}

WINDOW_SIZE, STRIDE = 128, 64

# 🎯 최종 선정된 30개 정예 피처 인덱스
SELECTED_INDICES = [
    156, 157, 158,  # Structural
    8, 11, 59, 107, 32, 140, 29,  # Ratio
    142, 34, 31, 46,  # Z-score
    98, 14, 97, 85,   # Slope/Log
    3, 42, 45, 69, 54, 30, 141, 135, # Baseline
    121, 35, 1        # Auxiliary
]

FEATURE_NAMES_30 = [
    "Is_Zero_ID", "ID_Priority", "Silence_Period",
    "IAT_Med_Ratio", "IAT_Slow_Ratio", "BSum_Slow_Ratio", "Cont_Slow_Ratio", "Ent_Med_Ratio", "BitEnt_Med_Ratio", "Ent_Fast_Ratio",
    "BitEnt_Slow_Z", "Ent_Slow_Z", "Ent_Med_Z", "BMean_Slow_Z",
    "Cont_Slope", "Jit_Slope", "Cont_Log", "Ham_Log",
    "IAT_Fast_Mean", "BMean_Med_Mean", "BMean_Slow_Mean", "BStd_Slow_Mean", "BSum_Med_Mean", "Ent_Med_Mean", "BitEnt_Slow_Mean", "BitEnt_Fast_Mean",
    "Freq_Log", "Ent_Slow_Ratio", "IAT_Log"
]

# =========================================================
# 2. Numba 엔진 (162개 생성 로직 유지)
# =========================================================
@njit
def popcount64(x):
    c = 0; v = uint64(x)
    while v > 0: v &= v - uint64(1); c += 1
    return c

@njit
def pack_payload_u64(row):
    v = uint64(0)
    for i in range(8): v |= uint64(row[i]) << (i * 8)
    return v

@njit(fastmath=True)
def calculate_features_ultra_numba(timestamps, can_ids, payloads, dlcs):
    n = len(timestamps)
    features = np.zeros((n, 162), dtype=np.float32)
    
    last_time_map = Dict.empty(key_type=types.int64, value_type=types.float64)
    last_iat_map = Dict.empty(key_type=types.int64, value_type=types.float64)
    last_payload_map = Dict.empty(key_type=types.int64, value_type=types.uint64)
    last_bmean_map = Dict.empty(key_type=types.int64, value_type=types.float64)
    last_raw_map = Dict.empty(key_type=types.int64, value_type=types.float64[:])
    
    ema_fast = Dict.empty(key_type=types.int64, value_type=types.float64[:])
    sq_ema_fast = Dict.empty(key_type=types.int64, value_type=types.float64[:])
    ema_med = Dict.empty(key_type=types.int64, value_type=types.float64[:])
    sq_ema_med = Dict.empty(key_type=types.int64, value_type=types.float64[:])
    ema_slow = Dict.empty(key_type=types.int64, value_type=types.float64[:])
    sq_ema_slow = Dict.empty(key_type=types.int64, value_type=types.float64[:])
    
    alphas = np.array([0.1, 0.01, 0.001], dtype=np.float64)
    eps = 1e-9

    for i in range(n):
        ts, cid, row, dlc = timestamps[i], can_ids[i], payloads[i], dlcs[i]
        iat = max(0.0, ts - last_time_map[cid]) if cid in last_time_map else 0.0
        jit = abs(iat - last_iat_map[cid]) if cid in last_iat_map else 0.0
        p_counts = np.zeros(256, dtype=np.float64)
        b_sum, b_sq_sum = 0.0, 0.0
        for b in row: p_counts[b] += 1.0; b_sum += b; b_sq_sum += b**2
        ent = 0.0
        for c in p_counts:
            if c > 0: p = c / 8.0; ent -= p * np.log2(p)
        b_mean = b_sum / 8.0
        b_std = np.sqrt(max(0.0, (b_sq_sum / 8.0) - (b_mean**2)))
        b_range = float64(np.max(row) - np.min(row))
        cur_bytes = pack_payload_u64(row)
        xor_val, ham = 0.0, 0.0
        if cid in last_payload_map:
            diff = cur_bytes ^ last_payload_map[cid]
            xor_val, ham = float64(diff), float64(popcount64(diff))
        cont = 1.0 - (ham / 64.0)
        mom = abs(b_mean - last_bmean_map[cid]) if cid in last_bmean_map else 0.0
        freq = 1.0 / (iat + eps)
        bit_ent = ent / 8.0
        
        raws = np.array([iat, jit, ent, b_mean, b_sum, b_std, b_range, ham, cont, mom, freq, bit_ent, xor_val], dtype=np.float64)
        
        if cid not in last_raw_map:
            last_raw_map[cid] = raws
            ema_fast[cid] = raws.copy(); sq_ema_fast[cid] = raws**2
            ema_med[cid] = raws.copy(); sq_ema_med[cid] = raws**2
            ema_slow[cid] = raws.copy(); sq_ema_slow[cid] = raws**2
            
        slopes = raws - last_raw_map[cid]
        
        col_idx = 0
        e_f, sq_f = ema_fast[cid], sq_ema_fast[cid]
        e_m, sq_m = ema_med[cid], sq_ema_med[cid]
        e_s, sq_s = ema_slow[cid], sq_ema_slow[cid]
        
        for q_idx in range(13):
            val = raws[q_idx]
            features[i, col_idx] = val; features[i, col_idx+1] = np.log1p(val); features[i, col_idx+2] = slopes[q_idx]; col_idx += 3
            
            e_f[q_idx] = (1-alphas[0])*e_f[q_idx] + alphas[0]*val
            sq_f[q_idx] = (1-alphas[0])*sq_f[q_idx] + alphas[0]*(val**2)
            std_f = np.sqrt(max(0.0, sq_f[q_idx] - e_f[q_idx]**2))
            features[i, col_idx] = e_f[q_idx]; features[i, col_idx+1] = (val - e_f[q_idx]) / (std_f + eps); features[i, col_idx+2] = val / (e_f[q_idx] + eps); col_idx += 3
            
            e_m[q_idx] = (1-alphas[1])*e_m[q_idx] + alphas[1]*val
            sq_m[q_idx] = (1-alphas[1])*sq_m[q_idx] + alphas[1]*(val**2)
            std_m = np.sqrt(max(0.0, sq_m[q_idx] - e_m[q_idx]**2))
            features[i, col_idx] = e_m[q_idx]; features[i, col_idx+1] = (val - e_m[q_idx]) / (std_m + eps); features[i, col_idx+2] = val / (e_m[q_idx] + eps); col_idx += 3
            
            e_s[q_idx] = (1-alphas[2])*e_s[q_idx] + alphas[2]*val
            sq_s[q_idx] = (1-alphas[2])*sq_s[q_idx] + alphas[2]*(val**2)
            std_s = np.sqrt(max(0.0, sq_s[q_idx] - e_s[q_idx]**2))
            features[i, col_idx] = e_s[q_idx]; features[i, col_idx+1] = (val - e_s[q_idx]) / (std_s + eps); features[i, col_idx+2] = val / (e_s[q_idx] + eps); col_idx += 3

        features[i, col_idx] = 1.0 if cid == 0 else 0.0
        features[i, col_idx+1] = 1.0 / (cid + 1.0)
        features[i, col_idx+2] = iat
        features[i, col_idx+3] = float(dlc)
        features[i, col_idx+4] = 0.0
        features[i, col_idx+5] = 0.0

        last_time_map[cid], last_iat_map[cid] = ts, iat
        last_payload_map[cid], last_bmean_map[cid], last_raw_map[cid] = cur_bytes, b_mean, raws

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
    
    if os.path.exists(SCALER_PATH):
        print(f"✅ 스케일러 로드: {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)
    else:
        print("❌ 스케일러가 없습니다. 소나타 학습 코드를 먼저 실행하세요!")
        return

    # 통합 저장용 리스트
    all_X_list = []
    all_y_list = []
    
    # CSV용 리스트 (메모리 주의)
    all_features_list = []
    all_labels_list = []

    for folder, attack_id in ATTACK_DIR.items():
        fpath = os.path.join(BASE_DIR, folder)
        if not os.path.exists(fpath): continue
        for fname in os.listdir(fpath):
            if not fname.endswith(".csv"): continue
            print(f"🚀 BMW {fname} 처리 중...")
            
            ts, cid, pay, dlc, lbl = process_csv_file_to_numpy(os.path.join(fpath, fname), attack_id)
            
            # 162개 추출 -> 30개 필터링
            feat_162 = calculate_features_ultra_numba(ts, cid, pay, dlc)
            feat_30 = feat_162[:, SELECTED_INDICES]
            
            # 정규화
            feat_30 = scaler.transform(feat_30)
            
            # [CSV 통합용] 리스트 추가
            all_features_list.append(feat_30)
            all_labels_list.append(lbl)

            # [NPZ 통합용] 윈도우 변환 및 리스트 추가
            for start in tqdm(range(0, len(feat_30) - WINDOW_SIZE + 1, STRIDE)):
                all_X_list.append(feat_30[start:start+WINDOW_SIZE].T)
                all_y_list.append(lbl[start:start+WINDOW_SIZE])

    # -----------------------------------------------------
    # 🏁 1. 통합 CSV 저장 (Testing)
    # -----------------------------------------------------
    if all_features_list:
        print("\n📦 통합 CSV 생성 중 (BMW/Test)...")
        final_X_csv = np.vstack(all_features_list)
        final_y_csv = np.concatenate(all_labels_list)
        
        df_final = pd.DataFrame(final_X_csv, columns=FEATURE_NAMES_30)
        df_final['Label'] = final_y_csv
        
        csv_path = os.path.join(OUTPUT_DIR, "carhacking_30_combined_S.csv")
        df_final.to_csv(csv_path, index=False)
        print(f"✅ 통합 CSV 저장 완료: {csv_path}")

    # -----------------------------------------------------
    # 🏁 2. 통합 NPZ 저장 (Testing)
    # -----------------------------------------------------
    if all_X_list:
        print("\n📦 통합 NPZ 생성 중 (BMW/Test)...")
        final_X_npz = np.array(all_X_list, dtype=np.float32)
        final_y_npz = np.array(all_y_list, dtype=np.int64)
        
        npz_path = os.path.join(OUTPUT_DIR, "carhacking_30_combined.npz")
        np.savez(npz_path, X=final_X_npz, y=final_y_npz)
        print(f"✅ 통합 NPZ 저장 완료: {npz_path}")

if __name__ == "__main__":
    process_test()
    print("\n✅ BMW(Test) 30개 피처 정규화 및 통합 저장 완료!")