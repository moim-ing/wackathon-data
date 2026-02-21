import logging

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

logging.basicConfig(level=logging.INFO)
from app.routers import compare

app = FastAPI()

app.include_router(compare.router)
