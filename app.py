from __future__ import annotations

import os
from datetime import datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from supabase import Client, create_client

from score_service import load_artifacts, score_participant_events


load_dotenv()

LOCAL_TZ = ZoneInfo("Asia/Colombo")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()

SUPABASE_SERVER_KEY = (
    os.getenv("SUPABASE_SECRET_KEY", "").strip()
    or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
)

NORMALIZATION_LOOKBACK_DAYS = int(
    os.getenv("NORMALIZATION_LOOKBACK_DAYS", "90")
)


app = FastAPI(
    title="Component 2 Digital Phenotyping API",
    version="1.1.1",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


_supabase: Optional[Client] = None


# ============================================================
# Supabase
# ============================================================

def get_supabase() -> Client:
    global _supabase

    if _supabase is not None:
        return _supabase

    if not SUPABASE_URL:
        raise RuntimeError(
            "SUPABASE_URL is not configured."
        )

    if not SUPABASE_SERVER_KEY:
        raise RuntimeError(
            "No Supabase server key configured. "
            "Set SUPABASE_SECRET_KEY or "
            "SUPABASE_SERVICE_ROLE_KEY."
        )

    _supabase = create_client(
        SUPABASE_URL,
        SUPABASE_SERVER_KEY,
    )

    return _supabase


# ============================================================
# Time helpers
# ============================================================

def local_midnight_to_utc_iso(d) -> str:
    local_dt = datetime.combine(
        d,
        time.min,
        tzinfo=LOCAL_TZ,
    )

    utc_dt = local_dt.astimezone(
        timezone.utc
    )

    return utc_dt.isoformat().replace(
        "+00:00",
        "Z",
    )


# ============================================================
# Raw sensor-event retrieval
#
# IMPORTANT:
# Supabase/PostgREST may return only ~1000 rows per request.
# Therefore we paginate until every matching row is loaded.
# ============================================================

def fetch_sensor_events(
    participant_id: str,
    end_date_exclusive,
) -> pd.DataFrame:

    start_date = (
        end_date_exclusive
        - timedelta(
            days=NORMALIZATION_LOOKBACK_DAYS
        )
    )

    start_iso = local_midnight_to_utc_iso(
        start_date
    )

    end_iso = local_midnight_to_utc_iso(
        end_date_exclusive
    )

    page_size = 1000
    offset = 0
    all_rows = []

    while True:

        response = (
            get_supabase()
            .table("sensor_events")
            .select(
                "participant_code,"
                "event_time,"
                "event_type,"
                "value_json,"
                "source"
            )
            .eq(
                "participant_code",
                participant_id,
            )
            .gte(
                "event_time",
                start_iso,
            )
            .lt(
                "event_time",
                end_iso,
            )
            .order(
                "event_time"
            )
            .range(
                offset,
                offset + page_size - 1,
            )
            .execute()
        )

        batch = response.data or []

        all_rows.extend(batch)

        if len(batch) < page_size:
            break

        offset += page_size

    return pd.DataFrame(
        all_rows
    )


# ============================================================
# Participant authentication
#
# Flutter sends:
# Authorization: Bearer <Supabase access token>
# ============================================================

def verify_participant(
    participant_id: str,
    authorization: Optional[str],
) -> None:

    if (
        not authorization
        or not authorization
        .lower()
        .startswith("bearer ")
    ):
        raise HTTPException(
            status_code=401,
            detail="Missing Supabase bearer token.",
        )

    token = authorization.split(
        " ",
        1,
    )[1].strip()

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing Supabase bearer token.",
        )

    try:
        user_response = (
            get_supabase()
            .auth
            .get_user(token)
        )

        user = user_response.user

    except Exception as exc:
        raise HTTPException(
            status_code=401,
            detail="Invalid Supabase session.",
        ) from exc

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid Supabase session.",
        )

    try:
        response = (
            get_supabase()
            .table("participants")
            .select(
                "auth_user_id,"
                "participant_code,"
                "active"
            )
            .eq(
                "participant_code",
                participant_id,
            )
            .limit(1)
            .execute()
        )

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Supabase query failed: {exc}",
        ) from exc

    rows = response.data or []

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="Participant not found.",
        )

    participant = rows[0]

    auth_matches = (
        str(
            participant.get(
                "auth_user_id"
            )
        )
        == str(user.id)
    )

    active = (
        participant.get("active")
        is True
    )

    if (
        not auth_matches
        or not active
    ):
        raise HTTPException(
            status_code=403,
            detail="Participant identity mismatch.",
        )


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
def startup_check() -> None:
    load_artifacts()


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():
    try:
        _, metadata = load_artifacts()

        return {
            "status": "ok",
            "service": "c2_behavioral",

            "model_version": metadata.get(
                "model_version",
                "M2_mobile_screen_location_v1",
            ),

            "fusion_default": (
                "enabled"
                if os.getenv(
                    "ENABLE_C2_FUSION",
                    "0",
                ).strip() == "1"
                else "disabled"
            ),

            "supabase_configured": bool(
                SUPABASE_URL
                and SUPABASE_SERVER_KEY
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model service not ready: "
                f"{exc}"
            ),
        ) from exc


# ============================================================
# FLUTTER DISPLAY ENDPOINT
#
# GET /behavioral/{participant_id}
#
# Returns:
# observation_payload
# passive_metrics
# day_coverage
# checkin_history
#
# Does NOT expose experimental risk score.
# ============================================================

@app.get(
    "/behavioral/{participant_id}"
)
def behavioral(
    participant_id: str,
    authorization: Optional[str] = Header(
        default=None
    ),
):

    participant_id = (
        participant_id.strip()
    )

    if not participant_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "participant_id cannot be empty."
            ),
        )

    verify_participant(
        participant_id,
        authorization,
    )

    try:

        obs_response = (
            get_supabase()
            .table(
                "behavioral_observations"
            )
            .select("*")
            .eq(
                "participant_code",
                participant_id,
            )
            .order(
                "created_at",
                desc=True,
            )
            .limit(1)
            .execute()
        )

        feature_response = (
            get_supabase()
            .table(
                "daily_behavior_features"
            )
            .select("*")
            .eq(
                "participant_code",
                participant_id,
            )
            .order(
                "feature_date",
                desc=True,
            )
            .limit(14)
            .execute()
        )

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Supabase query failed: "
                f"{exc}"
            ),
        ) from exc

    observation_rows = (
        obs_response.data or []
    )

    features = list(
        reversed(
            feature_response.data
            or []
        )
    )

    # No processed observation yet
    if not observation_rows:

        return {
            "observation_payload": {
                "participant_id":
                    participant_id,

                "baseline_ready":
                    False,

                "reportable":
                    False,

                "observations":
                    {},

                "data_quality":
                    {},

                "blocking_issues": [
                    "baseline_building"
                ],
            },

            "passive_metrics": {
                "activity_data_available":
                    False,
            },

            "day_coverage": [],

            "checkin_history": [],

            "model_output": None,

            "model_status":
                "withheld_pending_validation",
        }

    stored = observation_rows[0]

    quality = (
        stored.get(
            "data_quality"
        )
        or {}
    )

    baseline_ready = bool(
        stored.get(
            "baseline_ready",
            False,
        )
    )

    reportable = bool(
        stored.get(
            "reportable",
            False,
        )
    )

    blocking_issues: list[str] = []

    if not baseline_ready:

        blocking_issues.append(
            "baseline_building"
        )

    elif int(
        quality.get(
            "recent_usable_days"
        )
        or 0
    ) < 3:

        blocking_issues.append(
            "insufficient_recent_data"
        )

    usable = [
        row
        for row in features
        if row.get(
            "usable_day"
        )
        is True
    ]

    recent = usable[-7:]


    # ========================================================
    # Passive metrics
    # ========================================================

    if recent:

        home_minutes = [
            float(
                row.get(
                    "home_minutes"
                )
                or 0.0
            )
            for row in recent
        ]

        average_home_minutes = (
            sum(home_minutes)
            / len(home_minutes)
        )

        home_hours = (
            average_home_minutes
            / 60.0
        )

        places = [
            float(
                row.get(
                    "significant_places"
                )
                or 0.0
            )
            for row in recent
        ]

        movement = [
            float(
                row[
                    "high_motion_fraction"
                ]
            )
            for row in recent
            if row.get(
                "high_motion_fraction"
            )
            is not None
        ]

        passive_metrics = {

            "home_hours":
                round(
                    home_hours,
                    2,
                ),

            "away_hours":
                round(
                    max(
                        0.0,
                        24.0
                        - home_hours,
                    ),
                    2,
                ),

            "significant_places":
                (
                    round(
                        sum(places)
                        / len(places),
                        1,
                    )
                    if places
                    else 0.0
                ),

            "activity_proxy_score":
                (
                    round(
                        sum(movement)
                        / len(movement),
                        4,
                    )
                    if movement
                    else None
                ),

            "activity_data_available":
                bool(movement),
        }

    else:

        passive_metrics = {
            "activity_data_available":
                False,
        }


    # ========================================================
    # Flutter observation payload
    # ========================================================

    observation_payload = {

        "participant_id":
            participant_id,

        "window": {

            "start":
                stored.get(
                    "window_start"
                ),

            "end":
                stored.get(
                    "window_end"
                ),
        },

        "baseline_ready":
            baseline_ready,

        "reportable":
            reportable,

        "observations":
            stored.get(
                "observations"
            )
            or {},

        "data_quality":
            quality,

        "change_detection":
            stored.get(
                "change_detection"
            ),

        "blocking_issues":
            blocking_issues,
    }


    day_coverage = [

        {
            "date":
                row.get(
                    "feature_date"
                ),

            "usable":
                bool(
                    row.get(
                        "usable_day"
                    )
                ),

            "location_coverage":
                row.get(
                    "location_coverage"
                ),

            "screen_coverage":
                row.get(
                    "screen_coverage"
                ),

            "movement_coverage":
                row.get(
                    "movement_coverage"
                ),
        }

        for row in features
    ]


    return {

        "observation_payload":
            observation_payload,

        "passive_metrics":
            passive_metrics,

        "day_coverage":
            day_coverage,

        "checkin_history": [],

        # Participant UI should not receive
        # experimental risk probability.
        "model_output": None,

        "model_status":
            stored.get(
                "model_status",
                "withheld_pending_validation",
            ),
    }


# ============================================================
# EXPERIMENTAL NUMERICAL MODEL ENDPOINT
#
# GET /score/{participant_id}
# ============================================================

@app.get(
    "/score/{participant_id}"
)
def score(
    participant_id: str,

    window_end_date: Optional[str] = Query(
        default=None
    ),
):

    participant_id = (
        participant_id.strip()
    )

    if not participant_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "participant_id cannot be empty."
            ),
        )

    # Resolve end date WITHOUT using the
    # separate datetime.date name.
    try:

        end_date = (

            datetime.strptime(
                window_end_date,
                "%Y-%m-%d",
            ).date()

            if window_end_date

            else datetime.now(
                LOCAL_TZ
            ).date()
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                "window_end_date "
                "must be YYYY-MM-DD."
            ),
        ) from exc


    # Fetch ALL participant rows with pagination
    try:

        events = fetch_sensor_events(
            participant_id,
            end_date,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                "Supabase query failed: "
                f"{exc}"
            ),
        ) from exc


    if events.empty:

        return {

            "subject_id":
                participant_id,

            "modality":
                "c2_behavioral",

            "score":
                None,

            "status":
                "insufficient_data",

            "fusion_eligible":
                False,

            "behavioral_vulnerability_score":
                None,

            "reason":
                (
                    "No sensor events found "
                    "in the available history."
                ),

            "window_end":
                end_date.isoformat(),
        }


    # ========================================================
    # Score participant
    # ========================================================

    try:

        return score_participant_events(

            rows=events,

            participant_id=
                participant_id,

            window_end_date=
                end_date.isoformat(),

            normalization_rows=
                events,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Feature extraction/"
                "model scoring failed: "
                f"{exc}"
            ),
        ) from exc