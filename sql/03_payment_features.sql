-- Stage 2.3: payment-behaviour summary -> ONE row per applicant
-- Standard SQL: runs in DuckDB and in MySQL 8+ (window functions need MySQL 8).

WITH inst_flagged AS (
    SELECT
        SK_ID_CURR,

        -- row-level flags: one value per monthly payment
        CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1 ELSE 0 END AS is_late,      -- paid after the due date
        CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT
             THEN DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT ELSE 0 END        AS days_late,    -- how many days after
        CASE WHEN AMT_PAYMENT < AMT_INSTALMENT THEN 1 ELSE 0 END         AS is_short,     -- paid less than was due

        -- WINDOW FUNCTION: number each person's payments from most recent (1) to oldest.
        ROW_NUMBER() OVER (
            PARTITION BY SK_ID_CURR
            ORDER BY DAYS_INSTALMENT DESC, SK_ID_PREV DESC, NUM_INSTALMENT_NUMBER DESC
        ) AS rn
    FROM installments
    WHERE DAYS_ENTRY_PAYMENT <= 0      -- LEAKAGE GUARD: only payments that had already happened by the application day
),
inst_summary AS (
    SELECT
        SK_ID_CURR,
        COUNT(*)                                          AS inst_n_payments,
        SUM(is_late)                                      AS inst_n_late,
        AVG(is_late * 1.0)                                AS inst_late_share,        -- share of payments made late
        AVG(days_late * 1.0)                              AS inst_avg_days_late,     -- average days late (on-time counts as 0)
        MAX(days_late)                                    AS inst_max_days_late,     -- worst single delay
        AVG(is_short * 1.0)                               AS inst_short_share,       -- share of payments paid short
        -- behaviour on the 3 MOST RECENT payments only (AVG ignores the NULLs for older payments)
        AVG(CASE WHEN rn <= 3 THEN is_late * 1.0 END)     AS inst_recent3_late_share
    FROM inst_flagged
    GROUP BY SK_ID_CURR
)
SELECT * FROM inst_summary;
