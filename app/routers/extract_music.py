import os
import tempfile
import uuid

import boto3
import yt_dlp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

S3_BUCKET = os.environ["S3_BUCKET"]


class ExtractMusicRequest(BaseModel):
    url: str


class ExtractMusicResponse(BaseModel):
    key: str


@router.post("/extract_music", response_model=ExtractMusicResponse)
async def extract_music(req: ExtractMusicRequest):
    key = f"music/{uuid.uuid4()}.wav"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_template = os.path.join(tmpdir, "audio.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }],
            "quiet": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([req.url])
        except yt_dlp.utils.DownloadError as e:
            raise HTTPException(status_code=400, detail=f"음원 추출 실패: {e}")

        wav_path = os.path.join(tmpdir, "audio.wav")

        s3 = boto3.client("s3")
        s3.upload_file(wav_path, S3_BUCKET, key)

    return ExtractMusicResponse(key=key)
