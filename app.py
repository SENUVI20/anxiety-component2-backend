def fetch_sensor_events(
    participant_id: str,
    end_date_exclusive: date,
) -> pd.DataFrame:
    """
    Fetch all available sensor events needed for C2 scoring.

    Supabase/PostgREST limits individual responses, so retrieve
    the participant's history in pages rather than only the
    first ~1000 rows.
    """

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

    return pd.DataFrame(all_rows)