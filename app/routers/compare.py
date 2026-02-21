import logging
import os
import tempfile

logger = logging.getLogger(__name__)

import boto3
import librosa
import noisereduce as nr
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy.signal import correlate

router = APIRouter()

S3_BUCKET = os.environ["S3_BUCKET"]

SAMPLE_RATE = 16000
MAX_OFFSET_SEC = 4.0
CC_THRESHOLD = 0.25


def preprocess_denoise(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    y = librosa.util.normalize(y)
    y = nr.reduce_noise(y=y, sr=SAMPLE_RATE, stationary=True, prop_decrease=0.8)
    return y


def cross_correlation_score(y1: np.ndarray, y2: np.ndarray) -> float:
    max_lag_samples = int(MAX_OFFSET_SEC * SAMPLE_RATE)
    corr = correlate(y1, y2, mode='full')
    norm = np.sqrt(np.sum(y1 ** 2) * np.sum(y2 ** 2))
    if norm == 0:
        return 0.0
    corr_normalized = corr / norm
    center = len(corr) // 2
    search_slice = corr_normalized[center - max_lag_samples: center + max_lag_samples + 1]
    return float(np.abs(search_slice).max())


class CompareRequest(BaseModel):
    source_key: str
    recording_key: str
    offset_milli: int


class CompareResponse(BaseModel):
    result: bool


@router.post("/compare", response_model=CompareResponse)
async def compare(req: CompareRequest):
    s3 = boto3.client("s3")

    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = os.path.join(tmpdir, "source.wav")
        recording_path = os.path.join(tmpdir, "recording.wav")

        try:
            s3.download_file(S3_BUCKET, req.source_key, source_path)
            s3.download_file(S3_BUCKET, req.recording_key, recording_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"S3 다운로드 실패: {e}")

        y_source = preprocess_denoise(source_path)
        y_recording = preprocess_denoise(recording_path)

    # offset_milli만큼 소스 앞부분 제거해서 학생 녹음 시작점과 맞춤
    offset_samples = int(req.offset_milli / 1000 * SAMPLE_RATE)
    y_source = y_source[offset_samples: offset_samples + 5 * SAMPLE_RATE]

    score = cross_correlation_score(y_source, y_recording)
    result = score >= CC_THRESHOLD
    logger.info(
        "compare | source=%s recording=%s offset_ms=%d score=%.4f threshold=%.2f result=%s",
        req.source_key, req.recording_key, req.offset_milli, score, CC_THRESHOLD, result,
    )
    return CompareResponse(result=result)
