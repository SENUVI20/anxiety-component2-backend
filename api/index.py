from __future__ import annotations

from typing import Optional

from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Query,
)


app = FastAPI(
    title="Component 2 API"
)


# ============================================================
# Root
# ============================================================

@app.get("/api")
def root():
    return {
        "status": "ok",
        "service": "c2_behavioral",
        "message": (
            "Component 2 API "
            "is running on Vercel"
        ),
    }


# ============================================================
# Health
# ============================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "c2_behavioral",
        "deployment": "vercel",
    }


# ============================================================
# Flutter participant-facing endpoint
#
# Flutter calls:
#
# GET /api/behavioral/{participant_id}
# Authorization: Bearer <Supabase token>
#
# Returns:
# observation_payload
# passive_metrics
# day_coverage
# ============================================================

@app.get(
    "/api/behavioral/{participant_id}"
)
def behavioral(
    participant_id: str,

    authorization: Optional[str] = Header(
        default=None
    ),
):

    try:

        from app import (
            behavioral
            as component2_behavioral
        )

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=(
                "Component 2 backend "
                "failed to load: "
                f"{exc}"
            ),
        ) from exc


    return component2_behavioral(

        participant_id=
            participant_id,

        authorization=
            authorization,
    )


# ============================================================
# Experimental numerical C2 endpoint
#
# GET /api/score/{participant_id}
#
# This stays separate from the participant UI.
# ============================================================

@app.get(
    "/api/score/{participant_id}"
)
def score(
    participant_id: str,

    window_end_date: Optional[str] = Query(
        default=None
    ),
):

    try:

        from app import (
            score
            as component2_score
        )

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=(
                "Component 2 backend "
                "failed to load: "
                f"{exc}"
            ),
        ) from exc


    return component2_score(

        participant_id=
            participant_id,

        window_end_date=
            window_end_date,
    )