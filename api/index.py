from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="Component 2 API")


@app.get("/api")
def root():
    return {
        "status": "ok",
        "service": "c2_behavioral",
        "message": "Component 2 API is running on Vercel",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "c2_behavioral",
        "deployment": "vercel",
    }


@app.get("/api/behavioral/{participant_id}")
def behavioral(
    participant_id: str,
    window_end_date: Optional[str] = Query(default=None),
):
    try:
        from app import behavioral as component2_behavioral
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Component 2 backend failed to load: {exc}",
        ) from exc

    return component2_behavioral(
        participant_id=participant_id,
        window_end_date=window_end_date,
    )
