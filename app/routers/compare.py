import logging
import os
import tempfile

logger = logging.getLogger("uvicorn.error")

import boto3
from botocore.exceptions import BotoCoreError, ClientError
import librosa
import noisereduce as nr
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from scipy.signal import correlate

try:
    from pydantic import ConfigDict  # pydantic v2
except ImportError:  # pydantic v1
    ConfigDict = None

router = APIRouter()

S3_BUCKET = os.environ["S3_BUCKET"]

SAMPLE_RATE = 16000
MAX_OFFSET_SEC = 4.0
CC_THRESHOLD = 0.17


def preprocess_denoise(path: str) -> np.ndarray:
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        if y.size == 0:
            raise ValueError("빈 오디오 파일입니다.")
        y = librosa.util.normalize(y)
        y = nr.reduce_noise(y=y, sr=SAMPLE_RATE, stationary=True, prop_decrease=0.8)
        if y.size == 0:
            raise ValueError("전처리 후 유효한 오디오 구간이 없습니다.")
        return y
    except Exception as e:
        raise ValueError(f"오디오 디코딩/전처리 실패: {e}")


def cross_correlation_score(y1: np.ndarray, y2: np.ndarray) -> float:
    if y1.size == 0 or y2.size == 0:
        return 0.0
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
    source_key: str = Field(alias="sourceKey")
    recording_key: str = Field(alias="recordingKey")
    offset_milli: int = Field(alias="offsetMilli", ge=0)

    if ConfigDict is not None:
        model_config = ConfigDict(populate_by_name=True)
    else:
        class Config:
            allow_population_by_field_name = True
            allow_population_by_alias = True


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
        except (ClientError, BotoCoreError) as e:
            raise HTTPException(status_code=400, detail=f"S3 다운로드 실패: {e}")

        try:
            y_source = preprocess_denoise(source_path)
            y_recording = preprocess_denoise(recording_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # offset_milli만큼 소스 앞부분 제거해서 학생 녹음 시작점과 맞춤
    offset_samples = int(req.offset_milli / 1000 * SAMPLE_RATE)
    if offset_samples >= len(y_source):
        raise HTTPException(status_code=400, detail="offsetMilli가 source 오디오 길이를 초과합니다.")
    y_source = y_source[offset_samples: offset_samples + 5 * SAMPLE_RATE]
    if y_source.size == 0 or y_recording.size == 0:
        raise HTTPException(status_code=400, detail="비교할 유효 오디오 구간이 없습니다.")

    score = cross_correlation_score(y_source, y_recording)
    result = score >= CC_THRESHOLD
    logger.info(
        "compare | source=%s recording=%s offset_ms=%d score=%.4f threshold=%.2f result=%s",
        req.source_key, req.recording_key, req.offset_milli, score, CC_THRESHOLD, result,
    )
    return CompareResponse(result=result)
