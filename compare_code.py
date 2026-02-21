"""
음악 기반 출석 판별기 (지능형 노이즈 제거 버전)
Usage: python attendance_check.py <professor_audio> <student_audio>
Requirements: pip install librosa scipy numpy noisereduce
"""
import sys
import numpy as np
import librosa
import noisereduce as nr
from scipy.signal import correlate

# ────────────────────────────────────────────
# 설정값 (환경에 따라 튜닝)
# ────────────────────────────────────────────
SAMPLE_RATE = 16000          # 리샘플링 sr
MAX_OFFSET_SEC = 4.0         # 교수-학생 녹음 시작 시간차 허용 범위 (초)

# 판단 임계값
CC_THRESHOLD = 0.25          # 크로스코릴레이션 최솟값 (노이즈 제거 후 조정 필요할 수 있음)

# ────────────────────────────────────────────
# 1. 전처리 (지능형 노이즈 제거 방식)
# ────────────────────────────────────────────
def preprocess_denoise(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """새 방식: 로드 → 정규화 → 지능형 노이즈 제거 (밴드패스 없음)"""
    # 오디오 로드 및 모노 변환
    y, _ = librosa.load(path, sr=sr, mono=True)
    
    # 볼륨 정규화 (기기마다 마이크 감도 다름)
    y = librosa.util.normalize(y)
    
    # 지능형 노이즈 제거 적용
    # stationary=True: 배경 소음이 일정하다고 가정
    # prop_decrease=0.8: 노이즈를 얼마나 줄일지 (0.0 ~ 1.0)
    y_denoised = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.8)
    
    return y_denoised

# ────────────────────────────────────────────
# 2. 유사도 계산
# ────────────────────────────────────────────
def cross_correlation_score(y1: np.ndarray, y2: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[float, float]:
    """
    슬라이딩 크로스코릴레이션 최대값 반환
    """
    max_lag_samples = int(MAX_OFFSET_SEC * sr)

    corr = correlate(y1, y2, mode='full')
    norm = np.sqrt(np.sum(y1 ** 2) * np.sum(y2 ** 2))
    if norm == 0:
        return 0.0, 0.0

    corr_normalized = corr / norm

    center = len(corr) // 2
    search_slice = corr_normalized[center - max_lag_samples: center + max_lag_samples + 1]

    peak_idx = int(np.argmax(np.abs(search_slice)))
    peak_val = float(np.abs(search_slice[peak_idx]))
    lag_sec = (peak_idx - max_lag_samples) / sr

    return peak_val, lag_sec

# ────────────────────────────────────────────
# 3. 최종 판단
# ────────────────────────────────────────────
def is_same_location(path_prof: str, path_student: str, verbose: bool = True) -> dict:
    """
    두 오디오 파일이 같은 장소에서 녹음되었는지 판별
    """
    # 전처리 (지능형 노이즈 제거 적용)
    y1 = preprocess_denoise(path_prof)
    y2 = preprocess_denoise(path_student)

    # 점수 계산
    cc_score, lag = cross_correlation_score(y1, y2)

    scores = {
        "cross_correlation": round(cc_score, 4),
    }

    # 지표 통과 여부
    cc_pass = cc_score >= CC_THRESHOLD
    result = cc_pass

    # 출력
    if verbose:
        print("\n" + "=" * 50)
        print(f"  교수자 파일: {path_prof}")
        print(f"  학생   파일: {path_student}")
        print("=" * 50)
        print(f"  Noise Reduction   : Applied (prop_decrease=0.8)")
        print(f"  Cross-Correlation : {cc_score:.4f}  {'✓' if cc_pass else '✗'}  (임계값 {CC_THRESHOLD})")
        print(f"  추정 시간 오프셋  : {lag:+.2f}초")
        print("-" * 50)
        print(f"  ▶ 판정: {'✅ 출석 (같은 장소)' if result else '❌ 불출석 (다른 장소)'}")
        print("=" * 50 + "\n")

    return {
        "result": result,
        "scores": scores,
        "lag_seconds": round(lag, 3),
    }

# ────────────────────────────────────────────
# 실행
# ────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python attendance_check.py <professor_audio> <student_audio>")
        sys.exit(1)

    path_prof    = sys.argv[1]
    path_student = sys.argv[2]

    output = is_same_location(path_prof, path_student, verbose=True)
    sys.exit(0 if output["result"] else 1)