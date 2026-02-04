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
    features = np.zeros((n, 6), dtype=np.float32)
    
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

        # 2.[index 2] : ID가 0x000인지 여부(Dos 구분에 매우 중요!)
        is_zero_id = 1.0 if cid == 0 else 0.0
        features[i, 1] = float32(is_zero_id)
        

        # 3. [Index 2] Entropy
        ent = 0.0
        for c in counts:
            if c > 0:
                p = c / 8.0
                ent -= p * np.log(p)
        features[i, 2] = float32(ent / 2.1)

        
        
        # 4. [Index 3] Jitter
        if cid in last_iat_map: 
            jitter = abs(id_iat - last_iat_map[cid])
        else: 
            jitter = 0.0
        features[i, 3] = float32(min(jitter, 0.05) / 0.05)
        last_iat_map[cid] = id_iat # 업데이트

        # 5. ID Hamming
        cur_bytes = pack_payload_u64(row)

        if cid in last_payload_map:
            diff = cur_bytes ^ last_payload_map[cid]
            id_ham = popcount64(diff)

        else:
            id_ham = 0

        last_payload_map[cid] = cur_bytes
        features[i,4] = float32(id_ham / 64)

        # 6. Frequency
        if cid in last_id_map:
            cnt = last_id_map[cid]+1

        else:
            cnt = 1

        last_id_map[cid] = int64(cnt)
        features[i,5] = float32(cnt / 64)

    return features

