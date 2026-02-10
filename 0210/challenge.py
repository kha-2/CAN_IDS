import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
from numba import njit, float32, int64, float64, uint64, types
from numba.typed import Dict
from tqdm import tqdm

# =========================================================
# 1. 설정 (Configuration)
# =========================================================
# [경로 설정] 본인의 경로로 수정하세요
TRAIN_BASE_PATH = "D:/IDS_masters/dataset/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training"
OUTPUT_DIR = "D:/IDS_masters/features/challenge/final_30"
SCALER_PATH = "D:/IDS_masters/features/robust_scaler_30feat.pkl"

WINDOW_SIZE, STRIDE = 128, 64
LABEL_MAP = {"Normal": 0, "Flooding": 1, "DoS": 1, "Fuzzing": 2, "Replay": 3, "Spoofing": 4}

# 🎯 최종 선정된 30개 정예 피처 인덱스 (162개 기준)
SELECTED_INDICES = [
    156, 157, 158,  # Structural
    8, 11, 59, 107, 32, 140, 29,  # Ratio (Spoofing Killer)
    142, 34, 31, 46,  # Z-score
    98, 14, 97, 85,   # Slope/Log
    3, 42, 45, 69, 54, 30, 141, 135, # Baseline
    121, 35, 1        # Auxiliary
]

# 30개 피처 이름
FEATURE_NAMES_30 = [
    "Is_Zero_ID", "ID_Priority", "Silence_Period",
    "IAT_Med_Ratio", "IAT_Slow_Ratio", "BSum_Slow_Ratio", "Cont_Slow_Ratio", "Ent_Med_Ratio", "BitEnt_Med_Ratio", "Ent_Fast_Ratio",
    "BitEnt_Slow_Z", "Ent_Slow_Z", "Ent_Med_Z", "BMean_Slow_Z",
    "Cont_Slope", "Jit_Slope", "Cont_Log", "Ham_Log",
    "IAT_Fast_Mean", "BMean_Med_Mean", "BMean_Slow_Mean", "BStd_Slow_Mean", "BSum_Med_Mean", "Ent_Med_Mean", "BitEnt_Slow_Mean", "BitEnt_Fast_Mean",
    "Freq_Log", "Ent_Slow_Ratio", "IAT_Log"
]

# =========================================================
# 2. Numba 엔진 (162개 생성 후 30개 추출)
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
# 3. 데이터 처리 및 통합 저장
# =========================================================
def parse_id(id_val):
    try: return int(id_val, 16) if isinstance(id_val, str) else int(id_val)
    except: return 0

def parse_payload_str(s):
    try:
        parts = str(s).split()
        vals = [int(p, 16) for p in parts]
        return vals + [0]*(8-len(vals)) if len(vals) < 8 else vals[:8]
    except: return [0]*8

def process_training():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    csv_files = glob.glob(os.path.join(TRAIN_BASE_PATH, "*.csv"))
    
    scaler = RobustScaler()
    is_scaler_fitted = False
    
    # 통합 저장용 리스트 (CSV용)
    all_features_list = []
    all_labels_list = []

    for f in csv_files:
        print(f"\n[START] {os.path.basename(f)} 처리 중...")
        df = pd.read_csv(f)
        cols = {c.lower().strip(): c for c in df.columns}
        
        ts = df[cols.get('timestamp', df.columns[0])].astype(np.float64).values
        cid = df[cols.get('arbitration_id', cols.get('id', df.columns[1]))].apply(parse_id).astype(np.int64).values
        dlc_col = cols.get('dlc', df.columns[2])
        dlc = df[dlc_col].astype(np.int64).values if dlc_col in df.columns else np.full(len(ts), 8, dtype=np.int64)
        pay = np.vstack(df[cols.get('data', cols.get('payload', df.columns[3]))].apply(parse_payload_str).values).astype(np.uint8)
        
        label_col = cols.get('subclass', cols.get('class', None))
        lbl = np.array([LABEL_MAP.get(str(l).strip(), 0) for l in df[label_col].astype(str)] if label_col else np.zeros(len(df)), dtype=np.int64)

        # 162개 추출 -> 30개 필터링
        feat_162 = calculate_features_ultra_numba(ts, cid, pay, dlc)
        feat_30 = feat_162[:, SELECTED_INDICES]

        # 정규화
        if not is_scaler_fitted:
            print("⚖️ 스케일러 학습(fit) 및 저장 중...")
            feat_30 = scaler.fit_transform(feat_30)
            joblib.dump(scaler, SCALER_PATH)
            is_scaler_fitted = True
        else:
            feat_30 = scaler.transform(feat_30)

        # -------------------------------------------------
        # [CSV 통합용] 메모리에 추가 (주의: 메모리 부족 시 배치 처리 필요)
        # -------------------------------------------------
        # 전체를 다 넣으면 너무 크므로, 10개 중 1개만 샘플링하거나, 
        # 필요하다면 전체를 다 넣습니다. 여기서는 전체를 넣습니다.
        all_features_list.append(feat_30)
        all_labels_list.append(lbl)

        # -------------------------------------------------
        # [NPZ 저장] TCN 학습용 (파일별 분리 저장 유지 권장)
        # -------------------------------------------------
        X_list, y_list = [], []
        for start in tqdm(range(0, len(feat_30) - WINDOW_SIZE + 1, STRIDE)):
            X_list.append(feat_30[start:start+WINDOW_SIZE].T)
            y_list.append(lbl[start:start+WINDOW_SIZE])

        if X_list:
            npz_name = os.path.join(OUTPUT_DIR, f"training_30_{os.path.basename(f)}.npz")
            np.savez(npz_name, X=np.array(X_list, dtype=np.float32), y=np.array(y_list, dtype=np.int64))
            print(f"🎉 NPZ 저장 완료: {npz_name}")

    # -----------------------------------------------------
    # 🏁 모든 파일 처리 후 통합 CSV 1개 생성 (Training)
    # -----------------------------------------------------
    if all_features_list:
        print("\n📦 통합 CSV 생성 중 (Training)...")
        final_X = np.vstack(all_features_list)
        final_y = np.concatenate(all_labels_list)
        
        df_final = pd.DataFrame(final_X, columns=FEATURE_NAMES_30)
        df_final['Label'] = final_y
        
        # 파일 저장
        csv_path = os.path.join(OUTPUT_DIR, "training_30_combined_S.csv")
        df_final.to_csv(csv_path, index=False)
        print(f"✅ 통합 CSV 저장 완료: {csv_path}")

if __name__ == "__main__":
    process_training()
    print("\n✅ 소나타(Train) 30개 피처 정규화 및 통합 저장 완료!")