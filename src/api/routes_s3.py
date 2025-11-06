# src/api/routes_s3.py
from fastapi import APIRouter, UploadFile, HTTPException, Body
from src.aws.s3_utils import upload_to_s3, download_from_s3

router = APIRouter(prefix="/api/s3", tags=["S3"])

@router.post("/put-text")
async def put_text(key: str, body: str = Body(...)):
    path = upload_to_s3(key, body.encode("utf-8"))
    return {"ok": True, "path": path}

@router.get("/get-text")
def get_text(key: str):
    try:
        data = download_from_s3(key)
        return {"ok": True, "key": key, "body": data.decode("utf-8")}
    except Exception:
        raise HTTPException(status_code=404, detail=f"not found: {key}")

@router.post("/upload")
async def upload_file(file: UploadFile):
    data = await file.read()
    path = upload_to_s3(f"uploads/{file.filename}", data)
    return {"ok": True, "path": path, "size": len(data)}
