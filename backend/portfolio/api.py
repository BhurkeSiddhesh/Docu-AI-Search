from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse
import logging

from backend.portfolio.service import (
    list_sources, discover_sources, upload_source, preview_source,
    confirm_mapping, import_source, list_transactions, summary,
    warnings, ignore_source, delete_source, PortfolioError
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])

def _handle_error(exc):
    if isinstance(exc, PortfolioError):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc), "errors": exc.errors})
    logger.exception("Portfolio API Error")
    raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/sources")
async def get_sources():
    try:
        return list_sources()
    except Exception as e:
        return _handle_error(e)


@router.post("/discover")
async def discover(request: Request):
    try:
        body = await request.json()
        paths = body.get("paths")
    except Exception:
        paths = None
    try:
        return discover_sources(paths)
    except Exception as e:
        return _handle_error(e)


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        return upload_source(file.filename, content)
    except Exception as e:
        return _handle_error(e)


@router.get("/sources/{source_id}/preview")
async def preview(source_id: str):
    try:
        return preview_source(source_id)
    except Exception as e:
        return _handle_error(e)


@router.post("/sources/{source_id}/mapping")
async def save_mapping(source_id: str, request: Request):
    try:
        payload = await request.json()
        return confirm_mapping(source_id, payload)
    except Exception as e:
        return _handle_error(e)


@router.post("/sources/{source_id}/import")
async def do_import(source_id: str):
    try:
        return import_source(source_id)
    except Exception as e:
        return _handle_error(e)


@router.post("/sources/{source_id}/ignore")
async def ignore(source_id: str):
    try:
        return ignore_source(source_id)
    except Exception as e:
        return _handle_error(e)


@router.delete("/sources/{source_id}")
async def delete_src(source_id: str):
    try:
        return delete_source(source_id)
    except Exception as e:
        return _handle_error(e)


@router.get("/transactions")
async def get_transactions(request: Request):
    try:
        filters = dict(request.query_params)
        limit = int(filters.pop("limit", 100))
        offset = int(filters.pop("offset", 0))
        return list_transactions(filters, limit, offset)
    except Exception as e:
        return _handle_error(e)


@router.get("/summary")
async def get_summary(request: Request):
    try:
        filters = dict(request.query_params)
        return summary(filters)
    except Exception as e:
        return _handle_error(e)


@router.get("/warnings")
async def get_warnings():
    try:
        return warnings()
    except Exception as e:
        return _handle_error(e)
