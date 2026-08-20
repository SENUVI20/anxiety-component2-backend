from typing import Optional

from fastapi import FastAPI, Query

from app import health as component2_health
from app import behavioral as component2_behavioral


app = FastAPI(title="Component 2 API")


@app.get("/api/health")
def health():
    return component2_health()


@app.get("/api/behavioral/{participant_id}")
def behavioral(
    participant_id: str,
    window_end_date: Optional[str] = Query(default=None),
):
    return component2_behavioral(
        participant_id=participant_id,
        window_end_date=window_end_date,
    )