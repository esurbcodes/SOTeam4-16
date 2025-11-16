# src/main.py
import os
from dotenv import load_dotenv
load_dotenv()  # load .env early

from fastapi import FastAPI
from mangum import Mangum

from src.api.routers import models as models_router
from src.api.routes_s3 import router as s3_router  # mounts /api/s3/*

app = FastAPI(title="SOTeam4P2 API")

# Mount everything under /api
app.include_router(models_router.router, prefix="/api")
app.include_router(s3_router)  # routes_s3 defines prefix="/api/s3"

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/env")
def get_env_values():
    return {
        "S3_BUCKET": os.getenv("S3_BUCKET"),
        "AWS_REGION": os.getenv("AWS_REGION"),
        "DATABASE_URL": os.getenv("DATABASE_URL"),
    }

# Single Lambda entrypoint
handler = Mangum(app)
