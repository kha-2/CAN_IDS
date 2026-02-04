# %%
import os
import glob
import random
import pandas as pd
import numpy as np
import torch  
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from numba import njit, float32, int64, types
from numba.typed import Dict
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# %%
# ==========================================
# 1. 설정 (Configuration)
# ==========================================
# 실제 데이터가 있는 경로로 수정하세요
BASE_PATH = "C:/Users/user/Desktop/IDS_masters/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/1_Submission"
PT_SAVE_PATH = "C:/Users/user/Desktop/IDS_masters/dataset/training_dataset_0204_2차.pt"
CSV_SAVE_PATH = "C:/Users/user/Desktop/IDS_masters/dataset/training_dataset0204_2차.csv"
WINDOW_SIZE = 64
STRIDE = 32  # 50% Overlap

# 공격 라벨 정의
ATTACK_LABELS = {
    "Normal": 0,
    "Flooding": 1,
    "Fuzzing": 2,
    "Replay": 3,
    "Spoofing": 4
}
LABEL_MAP = {
    "Normal": 0,
    "Flooding": 1,   # 원본 명칭
    "DoS": 1,        # 혹시 나중에 DoS라는 문자열도 들어오면 같이 1로 처리
    "Fuzzing": 2,
    "Replay": 3,
    "Spoofing": 4,
}

FEATURE_NAMES = [
    "1. ID IAT", 
    "3. Entropy", "4. Jitter", "5. ID Hamming", "6. Frequency"
]


# %%
@njit
def popcount64(x):
    # x: uint8 -> 0~255
    c = 0
    v = int64(x)
    while v:
        v &= v - np.uint64(1)
        c += 1
    return c

@njit
def pack_payload_u64(row):
    v = np.uint64(0)
    for i in range(8):
        v |= np.uint64(row[i]) << (i * 8)
    return v

@njit(fastmath=True)
def calculate_features_numba(timestamps, can_ids, dlcs, payloads):
    n = len(timestamps)
    features = np.zeros((n, 5), dtype=np.float32)
    
    last_time_map = Dict.empty(key_type=types.int64, value_type=types.float64) # 이전 패킷 시간
    last_iat_map = Dict.empty(key_type=types.int64, value_type=types.float64) #이전 IAT
    last_payload_map = Dict.empty(key_type=types.int64, value_type=types.uint64) #이전 ID Payload
    last_id_map = Dict.empty(key_type=types.int64, value_type=types.float32) # 윈도우 내 id 빈도수
    

    # iat_history_map = Dict.empty(key_type=types.int64, value_type=types.float64[:])
    # ent_history_map = Dict.empty(key_type=types.int64, value_type=types.float64[:])
    # count_map = Dict.empty(key_type=types.int64, value_type=types.int64)
    # global_count_map = Dict.empty(key_type=types.int64, value_type=types.int64)
    prev_global_time = timestamps[0]

    for i in range(n):
        if (i % 64) == 0:
            last_id_map.clear()
        ts = timestamps[i]
        cid = can_ids[i]
        if np.isnan(ts): ts = prev_global_time

        else: prev_global_time = ts
        
    
        
        # 1. [Index 1] ID IAT
        if cid in last_time_map: 
            id_iat = max(0.0, ts - last_time_map[cid])
        else: 
            id_iat = 0.0
        features[i, 0] = float32(np.log1p(id_iat * 1000.0) / 7.0) 
        
        last_time_map[cid] = ts # 다음 계산을 위해 업데이트
        
        # --- 페이로드 관련 공통 준비 (Entropy, Mean, Std용) ---
        counts = np.zeros(256, dtype=np.int32)
        row = payloads[i]
        s = 0.0
        for b_idx in range(8):
            val = row[b_idx]
            counts[val] += 1
            s += val

        # # 2.[index 2] : ID가 0x000인지 여부(Dos 구분에 매우 중요!)
        # is_zero_id = 1.0 if cid == 0 else 0.0
        # features[i, 1] = float32(is_zero_id)
        

        # 3. [Index 2] Entropy
        ent = 0.0
        for c in counts:
            if c > 0:
                p = c / 8.0
                ent -= p * np.log(p)
        features[i, 1] = float32(ent / 2.1)

        
        
        # 4. [Index 3] Jitter
        if cid in last_iat_map: 
            jitter = abs(id_iat - last_iat_map[cid])
        else: 
            jitter = 0.0
        features[i, 2] = float32(min(jitter, 0.05) / 0.05)
        last_iat_map[cid] = id_iat # 업데이트

        # 5. ID Hamming
        cur_bytes = pack_payload_u64(row)

        if cid in last_payload_map:
            diff = cur_bytes ^ last_payload_map[cid]
            id_ham = popcount64(diff)

        else:
            id_ham = 0

        last_payload_map[cid] = cur_bytes
        features[i,3] = float32(id_ham / 64)

        # 6. Frequency
        if cid in last_id_map:
            cnt = last_id_map[cid]+1

        else:
            cnt = 1

        last_id_map[cid] = int64(cnt)
        features[i,4] = float32(cnt / 64)

    return features

# %%
# ==========================================
# 2. 헬퍼 함수 (ID 파싱, Payload 파싱)
# ==========================================
def parse_id(id_val):
    if isinstance(id_val, str):
        try:
            return int(id_val, 16)
        except:
            return 0
    return int(id_val)

def parse_payload_str(s, max_len=8):
    """'00 00 A1 ...' 형태의 문자열을 길이 8의 리스트로 변환"""
    parts = str(s).split()
    vals = []
    for p in parts:
        if p != "":
            try:
                vals.append(int(p, 16))
            except:
                pass
    
    if len(vals) < max_len:
        vals += [0] * (max_len - len(vals))
    return vals[:max_len]

# %%
# 2. 데이터셋 생성 함수 수정
def make_dataset_from_csv(
    BASE_PATH: str,
    window_size: int = 64,
    stride: int = 32
):
    # 스크린샷 기준 컬럼: timestamp, can_id, dlc, payload, label1, label2
    # header가 없다면 None으로 읽음
    df = pd.read_csv(BASE_PATH, header=None)

    # [수정] 라벨이 포함된 4, 5번 인덱스 컬럼(Normal/Attack, Replay)은 사용하지 않음
    # 필요한 0~3번 컬럼만 슬라이싱
    df = df.iloc[:, :4]
    df.columns = ["timestamp", "can_id", "dlc", "payload"]

    print(f"[INFO] 데이터 로드 완료. 총 패킷 수: {len(df)}")

    # 데이터 타입 변환
    timestamps = df["timestamp"].astype(np.float64).to_numpy()
    can_ids = df["can_id"].apply(parse_id).astype(np.int64).to_numpy()
    dlcs = df["dlc"].astype(np.int64).to_numpy()
    
    # Payload 처리 (16진수 문자열 -> 8바이트 배열)
    payload_list = df["payload"].apply(parse_payload_str).values
    payload_array = np.vstack(payload_list).astype(np.uint8)

    # 3. 전체 데이터에 대해 피처 미리 계산 (Numba 가속 함수 사용)
    # calculate_features_numba는 (N, 9) 형태의 행렬을 반환함
    print("[INFO] 피처 계산 중...")
    all_features = calculate_features_numba(timestamps, can_ids, payload_array)

    # 4. 윈도우 슬라이싱 (9, 64) 형태로 변환
    N = len(all_features)
    windows_features = []

    for start in range(0, N - window_size + 1, stride):
        end = start + window_size
        # (window_size, 9) -> (9, window_size)로 전치하여 모델 입력 규격 맞춤
        win_feat = all_features[start:end].T
        windows_features.append(win_feat)

    if not windows_features:
        raise RuntimeError("윈도우를 생성할 수 없습니다. 데이터가 WINDOW_SIZE보다 적습니다.")

    X = np.stack(windows_features, axis=0)  # (num_windows, 9, 64)
    return X

# %%
# ==========================================
# 4. 윈도우 자르기 (Sequence Labeling 수정 완료)
# ==========================================
def slice_windows(features, packet_labels, window_size, stride):
    """
    packet_labels: 이미 0, 1, 2, 3, 4가 마킹된 패킷 라벨 배열
    반환값: 
      - X: (N, 9, 64)
      - y: (N, 64)  <-- 벡터 형태
    """
    n_samples = len(features)
    if n_samples < window_size: return None, None
    
    n_windows = (n_samples - window_size) // stride + 1
    
    X_list = []
    y_list = []
  
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        
        # 1. Feature 자르기
        win_feat = features[start:end]
        
        # 2. Label 자르기 (이미 ID가 부여되어 있으므로 그대로 자름)
        # [중요] np.where 등을 쓰지 않고 그대로 가져옵니다.
        win_y = packet_labels[start:end]
        
        # 길이 체크 (마지막 짜투리 방지)
        if len(win_y) == window_size:
            X_list.append(win_feat.T) # (9, 64)로 전치
            y_list.append(win_y)      # (64,) 벡터
        
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

# %%

def make_dataset_with_labels(csv_path, window_size=64, stride=32):
    # 1. 데이터 로드 (모든 컬럼 읽기)
    # 스크린샷 기준: 0:ts, 1:id, 2:dlc, 3:payload, 4:label1(Normal/Attack), 5:label2(공격유형)
    df = pd.read_csv(csv_path, header=None)
    
    # 분석용 라벨 추출 (5번 컬럼: Replay, DoS 등 상세 라벨)
    raw_labels = df.iloc[:, 5].values 
    
    # 피처 추출용 데이터 정리
    timestamps = df.iloc[:, 0].astype(np.float64).to_numpy()
    can_ids = df.iloc[:, 1].apply(parse_id).astype(np.int64).to_numpy()
    dlcs = df.iloc[:, 2].astype(np.int64).to_numpy()
    payload_array = np.vstack(df.iloc[:, 3].apply(parse_payload_str).values).astype(np.uint8)

    # 2. 피처 계산 (Numba)
    all_features = calculate_features_numba(timestamps, can_ids,payload_array)

    X_list = []
    label_list = []

    # 3. 윈도우 슬라이싱 및 라벨 결정
    for start in range(0, len(all_features) - window_size + 1, stride):
        end = start + window_size
        
        # 모델용 피처 (9, 64)
        X_list.append(all_features[start:end].T)
        
        # [중요] CSV 확인용 라벨 결정: 윈도우 내의 마지막 패킷 라벨 혹은 대표 라벨 사용
        # 여기서는 사람이 읽기 편하도록 해당 윈도우의 '마지막 패킷 라벨'을 할당합니다.
        label_list.append(raw_labels[start:end])

    return np.array(X_list), label_list

# %%
def main():
    # 1. CSV 파일 목록
    csv_files = glob.glob(os.path.join(BASE_PATH, "*.csv"))
    if not csv_files:
        print(f"[ERROR] 해당 경로에 CSV 파일이 없습니다: {BASE_PATH}")
        return

    print(f"[INFO] 발견된 파일: {len(csv_files)}개")
    for f in csv_files:
        print("   -", os.path.basename(f))

    # 2. CSV 통합
    df_list = []
    for file in csv_files:
        print(f"[READING] {os.path.basename(file)} 읽는 중...")
        temp_df = pd.read_csv(file, header=0)
        df_list.append(temp_df)

    full_df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f"[INFO] 통합 완료. 총 패킷 수: {len(full_df)}")

    # 3. 라벨 정리 (Flooding → DoS 이름 통일)
    full_df["SubClass"] = full_df["SubClass"].astype(str).str.strip()
    full_df["SubClass"] = full_df["SubClass"].replace("Flooding", "DoS")

    # 패킷 단위 문자열 라벨
    raw_labels = full_df["SubClass"].astype(str).values

    # 4. 피처 계산에 필요한 컬럼 → numpy
    timestamps = full_df["Timestamp"].astype(np.float64).to_numpy()
    can_ids    = full_df["Arbitration_ID"].apply(parse_id).astype(np.int64).to_numpy()
    dlcs       = full_df["DLC"].astype(np.int64).to_numpy()
    payload_array = np.vstack(
        full_df["Data"].apply(parse_payload_str).values
    ).astype(np.uint8)

    print("[INFO] 통합 데이터 피처 계산 중...")
    all_features = calculate_features_numba(timestamps, can_ids, dlcs, payload_array)
    # all_features.shape = (패킷 수, 9)

    # 5. ===== 패킷 레벨 CSV 저장 =====
    packet_labels_int = np.vectorize(LABEL_MAP.get)(raw_labels).astype(np.int64)

    df_packet = pd.DataFrame(all_features, columns=FEATURE_NAMES)
    df_packet["Label_Int"] = packet_labels_int
    df_packet["Label_Str"] = raw_labels
    df_packet.to_csv(CSV_SAVE_PATH, index=False)
    print(f"[DONE] .csv 패킷 단위 저장 완료: {CSV_SAVE_PATH}")
    print(f"       형태: {df_packet.shape} (행: 패킷 수, 열: 특징+라벨)")

    # 6. ===== 윈도우 레벨 PT 저장 =====
    X_list = []
    y_list = []  # 이제 각 윈도우당 64개의 라벨을 저장

    num_packets = all_features.shape[0]
    for start in range(0, num_packets - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        
        # Feature: (64, 9) → (9, 64) 로 transpose
        X_list.append(all_features[start:end].T)
        
        # ✨ 핵심 변경: 윈도우 내 64개 패킷의 라벨을 모두 저장
        window_labels = packet_labels_int[start:end]  # (64,) 형태
        y_list.append(window_labels)

    X = np.array(X_list)     # (num_windows, 9, 64)
    y = np.array(y_list)     # (num_windows, 64)  ← 변경됨!

    np.savez(
    'C:/Users/user/Desktop/IDS_masters/dataset/carchallenge_training_0204_2차.npz',
    X=X.astype(np.float32),
    y=y.astype(np.int64)
    )

    print(f" Saved dataset")

if __name__ == "__main__":
    main()


