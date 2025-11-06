# src/main.py
from dotenv import load_dotenv
load_dotenv()  # <-- move to the very top, before local imports

from fastapi import FastAPI
from mangum import Mangum
from dotenv import load_dotenv
from src.api.routers import models as models_router
from src.api.routes_s3 import router as s3_router   # <-- add

app = FastAPI(title="SOTeam4P2 API")

# Mount everything under /api
app.include_router(models_router.router, prefix="/api")
# S3 Access
app.include_router(s3_router)                        # <-- add

@app.get("/api/env")
def get_env_values():
    return {
        "S3_BUCKET": os.getenv("S3_BUCKET"),
        "AWS_REGION": os.getenv("AWS_REGION"),
        "DATABASE_URL": os.getenv("DATABASE_URL")
    }

# Single Lambda entrypoint
handler = Mangum(app)
