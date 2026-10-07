import os
import glob
import pandas as pd
import numpy as np
from numba import njit, float64, int64, types
from numba.typed import Dict

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
# [수정] 본인 경로에 맞게 수정하세요
BASE_PATH = "C:/Users/user/Desktop/IDS_masters/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/1_Submission"
PT_SAVE_PATH = "C:/Users/user/Desktop/IDS_masters/training_dataset_add.pt"
CSV_SAVE_PATH = "C:/Users/user/Desktop/IDS_masters/training_dataset_add.csv"

WINDOW_SIZE = 128
STRIDE = 64

# 라벨 맵핑
LABEL_MAP = {
    "Normal": 0, "Flooding": 1, "DoS": 1, 
    "Fuzzing": 2, "Replay": 3, "Spoofing": 4
}

# 피처 이름 (9개 원본 + 6개 추가 = 15개)
FEATURE_NAMES = [
    # --- [기존 9개] ---
    "1.IAT", "2.Is_Zero", "3.Payload_Ent", "4.Complexity", 
    "5.Ham_Rate", "6.Freq", "7.Continuity", "8.Diff_Ent", "9.ID_Ent",
    # --- [신규 추가 6개] ---
    # Spoofing 탐지의 핵심 (변화율 감지)
    "10.Freq_Slope",   
    "11.Jit_Slope",    
    "12.Ent_Slope",    
    # 일반화 성능 확보 (Z-Score)
    "13.Freq_Fast_Z",  
    "14.Jit_Fast_Z",   
    "15.Bus_Load"      
]

# ==========================================
# 2. Helper 함수
# ==========================================
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
    v = np.uint64(0)
    for i in range(8):
        v |= np.uint64(row[i]) << (i * 8)
    return v

@njit
def parse_id(id_val):
    return int(id_val)

@njit
def parse_payload_str_numba(s_arr):
    # Numba 내부에서는 문자열 파싱이 까다로우므로 
    # 메인 함수에서 미리 전처리해서 넘겨주는 방식을 사용합니다.
    pass

# [신규] 162개 피처 로직과 동일한 EMA Z-Score 계산 함수
@njit
def update_ema_z(val, cid, ema_map, sq_ema_map, alpha):
    """
    계산법: (Value - EMA_Mean) / EMA_Std
    이유: 차종마다 기본 빈도나 지터 값이 달라도, '평소보다 얼마나 튀었는지'를 
          표준화된 수치(Z)로 변환하여 일반화 성능을 극대화함.
    """
    # 1. 초기화
    if cid not in ema_map:
        ema_map[cid] = float64(val)
        sq_ema_map[cid] = float64(val ** 2)
        return 0.0

    mean = ema_map[cid]
    sq_mean = sq_ema_map[cid]
    
    # 2. 분산 & 표준편차
    var = sq_mean - (mean ** 2)
    if var < 0: var = 0.0
    std = np.sqrt(var)

    # 3. Z-Score 계산
    z = 0.0
    if std > 1e-9:
        z = (val - mean) / std
        # 학습 안정성을 위해 이상치 Clipping (-5 ~ 5)
        if z > 5.0: z = 5.0
        elif z < -5.0: z = -5.0

    # 4. EMA 업데이트 (Alpha 0.1 = Fast)
    ema_map[cid] = (1.0 - alpha) * mean + alpha * val
    sq_ema_map[cid] = (1.0 - alpha) * sq_mean + alpha * (val ** 2)
    
    return z

# ==========================================
# 3. 메인 피처 계산 함수 (수정됨)
# ==========================================
@njit(fastmath=True)
def calculate_features_numba(timestamps, can_ids, payloads):
    n = len(timestamps)
    # 15개 피처로 확장
    features = np.zeros((n, 15), dtype=np.float64)
    
    # --- [기존 변수들] ---
    last_time_map = Dict.empty(key_type=types.int64, value_type=types.float64)
    last_payload_map = Dict.empty(key_type=types.int64, value_type=types.uint64)
    last_id_map = Dict.empty(key_type=types.int64, value_type=types.float64)
    id_ham_ema = Dict.empty(key_type=types.int64, value_type=types.float64)
    alpha_ham = 0.05
    eps = 1e-9

    # --- [신규 변수들: 162개 로직용] ---
    # Slope 계산용 "이전 값" 저장 (Diff 방식)
    last_freq_map = Dict.empty(key_type=types.int64, value_type=types.float64)
    last_jit_map_val = Dict.empty(key_type=types.int64, value_type=types.float64)
    last_ent_map = Dict.empty(key_type=types.int64, value_type=types.float64)
    
    # Jitter 계산용 "이전 IAT"
    last_iat_map = Dict.empty(key_type=types.int64, value_type=types.float64)
    
    # Z-Score용 EMA 맵 (Alpha=0.1 Fast)
    ema_freq = Dict.empty(key_type=types.int64, value_type=types.float64)
    sq_ema_freq = Dict.empty(key_type=types.int64, value_type=types.float64)
    ema_jit = Dict.empty(key_type=types.int64, value_type=types.float64)
    sq_ema_jit = Dict.empty(key_type=types.int64, value_type=types.float64)
    
    prev_global_time = timestamps[0]

    for i in range(n):
        # [기존] 64개마다 윈도우 빈도 초기화
        if (i % 64) == 0:
            last_id_map.clear()
            
        ts = timestamps[i]
        cid = can_ids[i]
        row = payloads[i]
        
        if np.isnan(ts): ts = prev_global_time
        else: prev_global_time = ts

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
        
        # --- [Part 1] 기존 9개 피처 (코드 보존) ---
        
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

        # --- [Part 2] 신규 6개 피처 (스푸핑/일반화 특화) ---
        
        # 10. Freq_Slope (빈도 변화율)
        # 계산법: 현재 빈도 - 직전 빈도
        # 효과: 스푸핑 주입 시 순간적으로 빈도가 치솟는 '기울기'를 포착.
        if cid in last_freq_map:
            features[i, 9] = curr_freq - last_freq_map[cid]
        else:
            features[i, 9] = 0.0
        last_freq_map[cid] = curr_freq

        # 11. Jit_Slope (지터 변화율)
        # 계산법: 현재 지터 - 직전 지터
        # 효과: 공격 패킷이 끼어들 때 발생하는 미세한 타이밍 엇박자를 감지. 
        #       스푸핑 탐지율 1위 피처 (Importance 0.073)
        if cid in last_jit_map_val:
            features[i, 10] = curr_jit - last_jit_map_val[cid]
        else:
            features[i, 10] = 0.0
        last_jit_map_val[cid] = curr_jit

        # 12. Ent_Slope (엔트로피 변화율)
        # 계산법: 현재 엔트로피 - 직전 엔트로피
        # 효과: Fuzzing이나 Payload 변조 시 데이터 무질서도의 급격한 변화를 감지.
        if cid in last_ent_map:
            features[i, 11] = ent - last_ent_map[cid]
        else:
            features[i, 11] = 0.0
        last_ent_map[cid] = ent

        # 13. Freq_Fast_Z (빈도 이상치)
        # 계산법: (빈도 - 평균) / 표준편차 (EMA Alpha=0.1)
        # 효과: 차종이 바뀌어도 통계적으로 '튀는' 빈도를 잡아내어 일반화 성능 확보.
        features[i, 12] = update_ema_z(curr_freq, cid, ema_freq, sq_ema_freq, 0.1)

        # 14. Jit_Fast_Z (지터 이상치)
        # 계산법: (지터 - 평균) / 표준편차 (EMA Alpha=0.1)
        # 효과: 주기적인 신호에서 벗어난 비정상적인 타이밍을 잡아냄.
        features[i, 13] = update_ema_z(curr_jit, cid, ema_jit, sq_ema_jit, 0.1)

        # 15. Bus_Load (전체 부하)
        # 계산법: 1 / 전체 IAT (초당 전체 패킷 수)
        # 효과: Flooding 공격 시 전체 트래픽 밀도가 높아지는 것을 감지.
        global_iat = ts - prev_global_time if i > 0 else 0.001
        features[i, 14] = 1.0 / (global_iat + 1e-9)

        # 상태 업데이트
        last_time_map[cid] = ts

    return features

# ==========================================
# 4. 실행 로직 (Main)
# ==========================================
def parse_payload_str(s):
    try:
        parts = str(s).split()
        vals = [int(p, 16) for p in parts]
        return vals + [0]*(8-len(vals)) if len(vals) < 8 else vals[:8]
    except: return [0]*8

def main():
    csv_files = glob.glob(os.path.join(BASE_PATH, "*.csv"))
    if not csv_files:
        print(f"[ERROR] 파일 없음: {BASE_PATH}")
        return

    print(f"[INFO] 발견된 파일: {len(csv_files)}개")
    df_list = []
    for f in csv_files:
        print(f"   - 읽는 중: {os.path.basename(f)}")
        df_list.append(pd.read_csv(f, header=0))

    full_df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f"[INFO] 통합 완료. 패킷 수: {len(full_df)}")

    # 라벨 정리
    full_df["SubClass"] = full_df["SubClass"].astype(str).str.strip().replace("Flooding", "DoS")
    raw_labels = full_df["SubClass"].values
    packet_labels_int = np.vectorize(LABEL_MAP.get)(raw_labels).astype(np.int64)

    # Numpy 변환
    timestamps = full_df["Timestamp"].astype(np.float64).to_numpy()
    can_ids = full_df["Arbitration_ID"].apply(lambda x: int(str(x), 16) if isinstance(x, str) else int(x)).astype(np.int64).to_numpy()
    payload_array = np.vstack(full_df["Data"].apply(parse_payload_str).values).astype(np.uint8)

    # 피처 계산
    print("[INFO] 피처 계산 중 (15 Features)...")
    all_features = calculate_features_numba(timestamps, can_ids, payload_array)

    # CSV 저장
    df_packet = pd.DataFrame(all_features, columns=FEATURE_NAMES)
    df_packet["Label_Int"] = packet_labels_int
    df_packet["Label_Str"] = raw_labels
    df_packet.to_csv(CSV_SAVE_PATH, index=False)
    print(f"[DONE] CSV 저장 완료: {CSV_SAVE_PATH}")

    # NPZ 저장 (윈도우 생성)
    X_list = []
    y_list = []
    
    num_packets = all_features.shape[0]
    for start in range(0, num_packets - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        # (64, 15) -> (15, 64) Transpose for TCN Input
        X_list.append(all_features[start:end].T)
        y_list.append(packet_labels_int[start:end])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    np.savez(PT_SAVE_PATH.replace(".pt", ".npz"), X=X, y=y)
    print(f"[DONE] NPZ 저장 완료: {PT_SAVE_PATH.replace('.pt', '.npz')} (Shape: {X.shape})")

if __name__ == "__main__":
    main()
