from google.cloud import bigquery
import pandas as pd


def get_channel_kpis_three_periods(
        channel_title: str,
        this_quarter: str,
        previous_quarter: str,
        same_quarter_last_year: str,
        PERIODS: dict):
    """
    Fetches and compares KPI metrics for a single YouTube channel across 3 specified periods.

    Args:
        channel_title (str): YouTube channel title (e.g., 'Veritasium')
        this_quarter (str):   Label for the “current” quarter (must be a key in PERIODS)
        previous_quarter (str):   Label for the quarter immediately before “this_quarter”
        same_quarter_last_year (str): Label for “same quarter last year”

    Returns:
        pd.DataFrame: Aggregated + derived metrics by period (one row per period)
    """
    # Ensure the provided quarter‐labels exist in the global PERIODS dict
    for label in (this_quarter, previous_quarter, same_quarter_last_year):
        if label not in PERIODS:
            raise KeyError(f"Quarter '{label}' not found in PERIODS")

    # Build a small dict of just the three periods we need
    periods = {
        this_quarter:           PERIODS[this_quarter],
        previous_quarter:       PERIODS[previous_quarter],
        same_quarter_last_year: PERIODS[same_quarter_last_year],
    }

    client = bigquery.Client()

    # Step 1: Get channel ID
    channel_id_query = """
        SELECT id
        FROM `electrify-production.youtube_public_data.channel_info`
        WHERE title = @channel_title
        LIMIT 1
    """
    id_job = client.query(
        channel_id_query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "channel_title", "STRING", channel_title)
            ]
        )
    )
    channel_id = id_job.to_dataframe().iloc[0]["id"]

    # Step 2: Build and run unioned query for all 3 periods
    period_queries = []
    for label, (start_date, end_date) in periods.items():
        query = f"""
            SELECT
                '{label}' AS period,
                DATE(day) AS day,
                SUM(views) AS views,
                SUM(estimated_hours_watched) AS watchtime_hours,
                SUM(estimated_revenue) AS revenue,
                SUM(subscribers_gained) AS subs_gained,
                SUM(subscribers_lost) AS subs_lost,
                SUM(views_premium) AS views_premium,
                SUM(estimated_hours_watched_premium) AS wth_premium,
                SUM(estimated_revenue_premium) AS revenue_premium,
                SUM(likes) AS likes,
                SUM(dislikes) AS dislikes,
                SUM(shares) AS shares,
                SUM(comments) AS comments,
                SUM(gross_revenue) AS gross_revenue,
                SUM(ad_impressions) AS ad_impressions
            FROM `electrify-production.youtube_private_data.daily_by_channel`
            WHERE id = '{channel_id}'
              AND day BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY day
        """
        period_queries.append(query)

    union_query = "\nUNION ALL\n".join(period_queries)
    df = client.query(union_query).to_dataframe()

    # Step 3: Aggregate
    df["net_subs"] = df["subs_gained"] - df["subs_lost"]
    grouped = df.groupby("period").agg({
        "views": "sum",
        "watchtime_hours": "sum",
        "revenue": "sum",
        "net_subs": "sum",
        "views_premium": "sum",
        "wth_premium": "sum",
        "revenue_premium": "sum",
        "likes": "sum",
        "dislikes": "sum",
        "shares": "sum",
        "comments": "sum",
        "gross_revenue": "sum",
        "ad_impressions": "sum"
    }).reset_index()

    # Step 4: Derived metrics
    grouped["AVD (mins)"] = grouped["watchtime_hours"] * 60 / grouped["views"]
    grouped["Revenue per 1k WTH"] = 1000 * \
        grouped["revenue"] / grouped["watchtime_hours"]
    grouped["% Likes"] = grouped["likes"] / \
        (grouped["likes"] + grouped["dislikes"])
    grouped["Net Likes per 1000 views"] = 1000 * \
        (grouped["likes"] - grouped["dislikes"]) / grouped["views"]
    grouped["Net Subs per 1000 views"] = 1000 * \
        grouped["net_subs"] / grouped["views"]
    grouped["Shares per 1000 views"] = 1000 * \
        grouped["shares"] / grouped["views"]
    grouped["Comments per 1000 views"] = 1000 * \
        grouped["comments"] / grouped["views"]
    grouped["% Views from YT Premium"] = grouped["views_premium"] / \
        grouped["views"]
    grouped["CPM"] = 1000 * grouped["gross_revenue"] / \
        grouped["ad_impressions"]
    grouped["RPM"] = 1000 * grouped["revenue"] / grouped["views"]

    # Final selection
    return grouped[[
        "period",
        "views",
        "watchtime_hours",
        "revenue",
        "net_subs",
        "AVD (mins)",
        "Revenue per 1k WTH",
        "% Likes",
        "Net Likes per 1000 views",
        "Net Subs per 1000 views",
        "Shares per 1000 views",
        "Comments per 1000 views",
        "% Views from YT Premium",
        "CPM",
        "RPM"
    ]].rename(columns={
        "views": "Total Views",
        "watchtime_hours": "Watch Time (hrs)",
        "revenue": "Estimated Revenue",
        "net_subs": "Net Subscriber Growth"
    }).sort_values("period").reset_index(drop=True)


def pivot_kpi_table(df_kpis: pd.DataFrame, value_columns: list = None) -> pd.DataFrame:
    """
    Pivots the KPI table so that each period becomes a column and each KPI becomes a row.

    Args:
        df_kpis (pd.DataFrame): Output from get_channel_kpis_three_periods
        value_columns (list): List of KPI metric names to pivot, defaults to all except 'period'

    Returns:
        pd.DataFrame: Pivoted table
    """
    if value_columns is None:
        value_columns = [col for col in df_kpis.columns if col != "period"]

    df_long = df_kpis.melt(id_vars="period", value_vars=value_columns,
                           var_name="Metric", value_name="Value")

    pivoted = df_long.pivot(
        index="Metric", columns="period", values="Value").reset_index()

    # Optional: order metrics based on your own preference
    preferred_order = [
        "Total Views",
        "AVD (mins)",
        "Watch Time (hrs)",
        "CPM",
        "RPM",
        "Revenue per 1k WTH",
        "Estimated Revenue",
        "Net Subscriber Growth",
        "% Views from YT Premium",
        "% Likes",
        "Net Likes per 1000 views",
        "Net Subs per 1000 views",
        "Shares per 1000 views",
        "Comments per 1000 views"


    ]

    pivoted["Metric"] = pd.Categorical(
        pivoted["Metric"], categories=preferred_order, ordered=True)
    pivoted = pivoted.sort_values("Metric")

    return pivoted


def summarize_device_type_kpis_by_title(
        channel_title: str,
        this_quarter: str,
        previous_quarter: str,
        same_quarter_last_year: str,
        PERIODS: dict,
        group_by_device: bool = True):
    """
        Fetches and summarizes KPIs by device type from BigQuery for a given channel title,
        using three explicitly specified quarter‐labels (instead of a periods dict).

        Args:
            channel_title (str): Channel title (e.g. 'Veritasium')
            this_quarter (str):   Label for the “current” quarter (must be a key in PERIODS)
            previous_quarter (str):   Label for the quarter immediately before this_quarter
            same_quarter_last_year (str): Label for “same quarter last year”
            group_by_device (bool): Whether to group results by device type.

        Returns:
            pd.DataFrame: KPI summary table by period (and optionally device type).
        """
    # 1) Verify that the three quarter‐labels exist in the global PERIODS dict
    for label in (this_quarter, previous_quarter, same_quarter_last_year):
        if label not in PERIODS:
            raise KeyError(f"Quarter '{label}' not found in PERIODS")

        # 2) Build a small periods dict containing only those three
    periods = {
        this_quarter:           PERIODS[this_quarter],
        previous_quarter:       PERIODS[previous_quarter],
        same_quarter_last_year: PERIODS[same_quarter_last_year],
    }
    client = bigquery.Client()

    # Step 1: Get channel ID from title
    channel_id_query = """
        SELECT id
        FROM `electrify-production.youtube_public_data.channel_info`
        WHERE title = @channel_title
        LIMIT 1
    """
    id_result = client.query(
        channel_id_query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter(
                "channel_title", "STRING", channel_title)]
        )
    ).to_dataframe()

    if id_result.empty:
        raise ValueError(f"No channel found with title '{channel_title}'")

    channel_id = id_result.iloc[0]["id"]

    # Step 2: Read in full dataset from BigQuery (filtering early via channel ID)
    device_query = f"""
        SELECT
            id, day, device_type, views, estimated_hours_watched,
            average_view_duration_mins, impressions, ctr
        FROM `electrify-production.youtube_private_data.daily_by_channel_and_devicetype`
        WHERE id = '{channel_id}'
    """
    df = client.query(device_query).to_dataframe()
    df['day'] = pd.to_datetime(df['day'])

    # Step 3: Summarize for each period
    results = []

    for period_label, (start, end) in periods.items():
        start_dt = pd.to_datetime(start).tz_localize('UTC')
        end_dt = pd.to_datetime(end).tz_localize('UTC')

        df_period = df[(df['day'] >= start_dt) & (df['day'] <= end_dt)]

        group_cols = ['device_type'] if group_by_device else []
        grouped = df_period.groupby(group_cols).agg({
            'views': 'sum',
            'estimated_hours_watched': 'sum',
            'average_view_duration_mins': 'mean',
            'impressions': 'sum',
            'ctr': 'mean'
        }).reset_index()

        grouped['Period'] = period_label
        results.append(grouped)

    # Step 4: Combine and tidy
    summary_df = pd.concat(results)

    summary_df = summary_df.rename(columns={
        'device_type': 'Device Type',
        'views': 'Total Views',
        'estimated_hours_watched': 'Watch Time (hrs)',
        'average_view_duration_mins': 'Avg View Duration (mins)',
        'impressions': 'Impressions',
        'ctr': 'CTR'
    })

    if group_by_device:
        summary_df = summary_df[['Period', 'Device Type', 'Total Views', 'Watch Time (hrs)',
                                 'Avg View Duration (mins)', 'Impressions', 'CTR']]
    else:
        summary_df = summary_df[['Period', 'Total Views', 'Watch Time (hrs)',
                                 'Avg View Duration (mins)', 'Impressions', 'CTR']]

    return summary_df.reset_index(drop=True)


def format_device_type_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the device summary from summarize_device_type_kpis_by_title and returns a flat table with:
      - One row per device type & metric (% total views, % total WTH)
      - One column per period
    """
    df = summary_df.copy()

    # Ensure clean types
    df['Period'] = df['Period'].astype(str)
    df['Device Type'] = df['Device Type'].astype(str)

    # Total views/WTH per period
    totals = df.groupby("Period")[["Total Views", "Watch Time (hrs)"]].sum().rename(
        columns=lambda x: f"Total {x}"
    )

    df = df.merge(totals, on="Period")

    # Calculate % share
    df["% total views"] = df["Total Views"] / df["Total Total Views"]
    df["% total WTHs"] = df["Watch Time (hrs)"] / df["Total Watch Time (hrs)"]

    # Keep only needed
    melted = df[["Period", "Device Type", "% total views", "% total WTHs"]]

    long = melted.melt(id_vars=["Period", "Device Type"],
                       var_name="Metric",
                       value_name="Value")

    final = long.pivot(index=["Device Type", "Metric"],
                       columns="Period", values="Value").reset_index()

    metric_order = ["% total views", "% total WTHs"]
    final["Metric"] = pd.Categorical(
        final["Metric"], categories=metric_order, ordered=True)
    final = final.sort_values(["Device Type", "Metric"])

    return final


def get_agg_video_level_actuals(
        channel_names: list,
        this_quarter: str,
        previous_quarter: str,
        same_quarter_last_year: str,
        PERIODS: dict):
    """
    Aggregates actuals at the video level across 3 specified periods for multiple channels.

    Args:
        channel_names (list): List of YouTube channel names.
        this_quarter (str):             Label for the “current” quarter (must be a key in PERIODS)
        previous_quarter (str):         Label for the quarter immediately before this_quarter
        same_quarter_last_year (str):   Label for “same quarter last year”

    Returns:
        pd.DataFrame: Aggregated + derived metrics by period and form (Short-Form, Long-Form).
    """
    # 1) Verify that the three quarter-labels exist in the global PERIODS dict
    for label in (this_quarter, previous_quarter, same_quarter_last_year):
        if label not in PERIODS:
            raise KeyError(f"Quarter '{label}' not found in PERIODS")

    # 2) Build a small periods dict containing only those three
    periods = {
        this_quarter:           PERIODS[this_quarter],
        previous_quarter:       PERIODS[previous_quarter],
        same_quarter_last_year: PERIODS[same_quarter_last_year],
    }

    # Build union query
    period_queries = []
    for channel_name in channel_names:
        for label, (start_date, end_date) in periods.items():
            query = f"""
                SELECT
                    '{label}' AS period,
                    CASE
                        WHEN is_short THEN 'Short-Form'
                        ELSE 'Long-Form'
                    END AS form_type,
                    SUM(views) AS views,
                    SUM(estimated_hours_watched) AS watchtime_hours,
                    SUM(estimated_revenue) AS revenue,
                    SUM(subscribers_gained) AS subs_gained,
                    SUM(subscribers_lost) AS subs_lost,
                    SUM(views_premium) AS views_premium,
                    SUM(estimated_hours_watched_premium) AS wth_premium,
                    SUM(estimated_revenue_premium) AS revenue_premium,
                    SUM(likes) AS likes,
                    SUM(dislikes) AS dislikes,
                    SUM(shares) AS shares,
                    SUM(comments) AS comments,
                    SUM(gross_revenue) AS gross_revenue,
                    SUM(ad_impressions) AS ad_impressions,
                    SUM(CASE WHEN DATE(published_at) = DATE(day) THEN 1 ELSE 0 END) AS num_videos_published
                FROM `electrify-production.youtube_private_data.daily_by_video__{channel_name}`
                WHERE day BETWEEN '{start_date}' AND '{end_date}'
                  AND privacy_status = 'public'
                GROUP BY form_type
            """
            period_queries.append(query)

    union_query = "\nUNION ALL\n".join(period_queries)

    # Wrap and aggregate
    final_query = f"""
        WITH combined AS (
            {union_query}
        )
        SELECT
            period,
            form_type,
            SUM(views) AS views,
            SUM(watchtime_hours) AS watchtime_hours,
            SUM(revenue) AS revenue,
            SUM(subs_gained) - SUM(subs_lost) AS net_subs,
            SUM(views_premium) AS views_premium,
            SUM(wth_premium) AS wth_premium,
            SUM(revenue_premium) AS revenue_premium,
            SUM(likes) AS likes,
            SUM(dislikes) AS dislikes,
            SUM(shares) AS shares,
            SUM(comments) AS comments,
            SUM(gross_revenue) AS gross_revenue,
            SUM(ad_impressions) AS ad_impressions,
            SUM(num_videos_published) AS num_videos_published
        FROM combined
        GROUP BY period, form_type
        ORDER BY period, form_type
    """

    # Run query
    df = pd.read_gbq(final_query)

    # Derived metrics
    df["AVD (mins)"] = df["watchtime_hours"] * 60 / df["views"]
    df["Revenue per 1k WTH"] = 1000 * df["revenue"] / df["watchtime_hours"]
    df["% Likes"] = df["likes"] / (df["likes"] + df["dislikes"])
    df["Net Likes per 1000 views"] = 1000 * \
        (df["likes"] - df["dislikes"]) / df["views"]
    df["Net Subs per 1000 views"] = 1000 * df["net_subs"] / df["views"]
    df["Shares per 1000 views"] = 1000 * df["shares"] / df["views"]
    df["Comments per 1000 views"] = 1000 * df["comments"] / df["views"]
    df["% Views from YT Premium"] = df["views_premium"] / df["views"]
    df["CPM"] = 1000 * df["gross_revenue"] / df["ad_impressions"]
    df["RPM"] = 1000 * df["revenue"] / df["views"]

    # Select final columns
    df = df[[
        "period",
        "form_type",
        "num_videos_published",
        "views",
        "watchtime_hours",
        "revenue",
        "net_subs",
        "AVD (mins)",
        "Revenue per 1k WTH",
        "% Likes",
        "Net Likes per 1000 views",
        "Net Subs per 1000 views",
        "Shares per 1000 views",
        "Comments per 1000 views",
        "% Views from YT Premium",
        "CPM",
        "RPM"
    ]]

    # Rename clean
    df = df.rename(columns={
        "num_videos_published": "Num Videos Published",
        "views": "Total Views",
        "watchtime_hours": "Watch Time (hrs)",
        "revenue": "Estimated Revenue",
        "net_subs": "Net Subscriber Growth",
        "form_type": "Content Type"
    })

    return df


def prepare_pivot_kpi_table(df_kpis: pd.DataFrame, content_type: str = "Long-Form", value_columns: list = None) -> pd.DataFrame:
    """
    Filters by content type (Long-Form or Short-Form), pivots KPI table so that each period becomes a column 
    and each KPI becomes a row.

    Args:
        df_kpis (pd.DataFrame): Output from _get_agg_video_level_actuals
        content_type (str): "Long-Form" or "Short-Form"
        value_columns (list): List of KPI metric names to pivot, defaults to all except 'period'

    Returns:
        pd.DataFrame: Pivoted KPI table
    """

    # Step 1: Filter to the selected content type
    df_filtered = df_kpis[df_kpis['Content Type'] == content_type].copy()

    # Step 2: Drop 'Content Type' (no longer needed)
    df_filtered = df_filtered.drop(columns=['Content Type'])

    # Step 3: Set 'period' as index
    df_filtered = df_filtered.set_index('period')

    # Step 4: Pivot
    if value_columns is None:
        value_columns = [col for col in df_filtered.columns if col != "period"]

    df_long = df_filtered.reset_index().melt(
        id_vars="period",
        value_vars=value_columns,
        var_name="Metric",
        value_name="Value"
    )

    pivoted = df_long.pivot(
        index="Metric", columns="period", values="Value"
    ).reset_index()

    # Step 5: Optional: order metrics nicely
    preferred_order = [
        "Num Videos Published",
        "Total Views",
        "AVD (mins)",
        "Watch Time (hrs)",
        "CPM",
        "RPM",
        "Revenue per 1k WTH",
        "Estimated Revenue",
        "Net Subscriber Growth",
        "% Views from YT Premium",
        "% Likes",
        "Net Likes per 1000 views",
        "Net Subs per 1000 views",
        "Shares per 1000 views",
        "Comments per 1000 views"
    ]

    pivoted["Metric"] = pd.Categorical(
        pivoted["Metric"], categories=preferred_order, ordered=True
    )
    pivoted = pivoted.sort_values("Metric")

    return pivoted
