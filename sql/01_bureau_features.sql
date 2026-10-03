-- Stage 2.1: bureau summary -> ONE row per applicant
-- Standard SQL: runs in DuckDB and in MySQL 8+.

WITH bureau_summary AS (
    SELECT
        SK_ID_CURR,

        -- how many credits does the person have/had with other lenders?
        COUNT(*)                                                    AS bureau_n_loans,
        SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END)   AS bureau_n_active,

        -- money still owed vs money originally extended, on ACTIVE credits only
        SUM(CASE WHEN CREDIT_ACTIVE = 'Active' AND AMT_CREDIT_SUM_DEBT > 0 THEN AMT_CREDIT_SUM_DEBT ELSE 0 END) AS bureau_active_debt,   -- negative debts counted as 0
        SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN AMT_CREDIT_SUM      ELSE 0 END) AS bureau_active_credit,

        -- overdue behaviour
        MAX(CREDIT_DAY_OVERDUE)                                     AS bureau_max_overdue_days,
        SUM(CASE WHEN CREDIT_DAY_OVERDUE > 0 THEN 1 ELSE 0 END)     AS bureau_n_overdue,
        SUM(COALESCE(AMT_CREDIT_SUM_OVERDUE, 0))                    AS bureau_sum_overdue_amt,   -- money currently overdue
        SUM(CASE WHEN DAYS_CREDIT >= -365 THEN 1 ELSE 0 END)        AS bureau_n_recent_loans     -- credits opened in the last year
    FROM bureau
    WHERE DAYS_CREDIT <= 0          -- only credits that started on/before the application date
    GROUP BY SK_ID_CURR
)
SELECT
    *,
    -- utilisation = share of the active credit that is still owed (NULL if no active credit)
    bureau_active_debt / NULLIF(bureau_active_credit, 0) AS bureau_utilisation
FROM bureau_summary;
