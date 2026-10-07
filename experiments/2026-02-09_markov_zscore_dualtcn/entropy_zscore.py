import numpy as np
from numba import njit, types
from numba.typed import Dict

@njit(fastmath=True)
def calculate_entropy_zscore(can_ids, payloads):
    n = len(can_ids)
    # 결과 담을 배열 (1차원)
    features = np.zeros(n, dtype=np.float32)
    
    # [상태 저장용 맵] Key: CAN ID, Value: [Sum, Sum_Sq, Count]
    # 평균과 분산을 누적 계산하기 위한 저장소입니다.
    ent_history_map = Dict.empty(key_type=types.int64, value_type=types.float64[:])
    
    eps = 1e-7  # 0으로 나누기 방지용

    for i in range(n):
        cid = can_ids[i]
        
       

        # ---------------------------------------------------------
        #  히스토리 관리 및 Z-score 계산
        # ---------------------------------------------------------
        if cid not in ent_history_map:
            # [합계, 제곱합, 샘플수] 초기화
            ent_history_map[cid] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        hist = ent_history_map[cid]
        n_prev = hist[2] # 지금까지 쌓인 개수

        final_ent_val = 0.0

        # 데이터가 30개 이상 쌓였을 때부터 통계적 Z-score 계산
        if n_prev > 30:
            avg_ent = hist[0] / n_prev
            # 분산 공식: E[X^2] - (E[X])^2
            var_ent = (hist[1] / n_prev) - (avg_ent ** 2)
            std_ent = np.sqrt(np.maximum(var_ent, 0.0))
            
            # Z-score: (현재값 - 평균) / 표준편차
            z_ent = (curr_ent - avg_ent) / (std_ent + eps)
            
            # Clipping (-7.0 ~ 7.0)
            final_ent_val = np.maximum(np.minimum(z_ent, 7.0), -7.0)
        
        # 결과 저장
        features[i] = np.float32(final_ent_val)

        # ---------------------------------------------------------
        # 3. 히스토리 업데이트 (다음 패킷을 위해)
        # ---------------------------------------------------------
        hist[0] += curr_ent          # 합계 누적
        hist[1] += curr_ent ** 2     # 제곱합 누적
        hist[2] += 1.0               # 카운트 증가

    return features
