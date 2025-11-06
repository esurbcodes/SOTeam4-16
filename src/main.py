import os
from fastapi import FastAPI
from mangum import Mangum
from dotenv import load_dotenv
from src.api.routers import models as models_router

load_dotenv()

app = FastAPI(title="SOTeam4P2 API")

# Mount everything under /api
app.include_router(models_router.router, prefix="/api")

@app.get("/api/env")
def get_env_values():
    return {
        "S3_BUCKET": os.getenv("S3_BUCKET"),
        "AWS_REGION": os.getenv("AWS_REGION"),
        "DATABASE_URL": os.getenv("DATABASE_URL")
    }

# Single Lambda entrypoint
handler = Mangum(app)
