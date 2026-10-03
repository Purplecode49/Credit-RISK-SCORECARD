-- Stage 2.2: previous-application summary -> ONE row per applicant
-- Standard SQL: runs in DuckDB and in MySQL 8+ (window functions need MySQL 8).

WITH prev_ranked AS (
    SELECT
        SK_ID_CURR,
        NAME_CONTRACT_STATUS,
        AMT_CREDIT,
        -- WINDOW FUNCTION: number each person's earlier applications from most recent (1) to oldest.
        -- DAYS_DECISION is negative, so the LARGEST value (closest to 0) is the most recent.
        ROW_NUMBER() OVER (
            PARTITION BY SK_ID_CURR
            ORDER BY DAYS_DECISION DESC, SK_ID_PREV DESC   -- SK_ID_PREV breaks ties
        ) AS rn
    FROM previous_application
    WHERE DAYS_DECISION <= 0        -- only decisions made on/before the application date
),
prev_summary AS (
    SELECT
        SK_ID_CURR,
        COUNT(*)                                                           AS prev_n_apps,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END) AS prev_n_approved,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'  THEN 1 ELSE 0 END) AS prev_n_refused,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Canceled' THEN 1 ELSE 0 END) AS prev_n_canceled,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN AMT_CREDIT ELSE 0 END) AS prev_approved_credit,
        -- was the MOST RECENT earlier application refused? (1 = yes, 0 = no)
        MAX(CASE WHEN rn = 1 THEN (CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END) END) AS prev_last_refused
    FROM prev_ranked
    GROUP BY SK_ID_CURR
)
SELECT
    *,
    prev_n_refused * 1.0 / prev_n_apps AS prev_refusal_rate   -- prev_n_apps is always >= 1 here
FROM prev_summary;
