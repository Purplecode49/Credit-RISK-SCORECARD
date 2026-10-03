-- Stage 2.4: the feature table -> ONE row per applicant (all 307,511 in the real data)
-- Needs three views holding the summaries from 2.1-2.3:
--   bureau_feat (01_bureau_features.sql), prev_feat (02_previous_application_features.sql), pay_feat (03_payment_features.sql)
-- Standard SQL: runs in DuckDB and in MySQL 8+.
--
-- Rule for people with NO history in a table (the LEFT JOIN leaves their summary columns empty/NULL):
--   * counts and amounts  -> COALESCE(x, 0)   "they have 0 loans / 0 debt" is literally true
--   * rates, maxima, "last" -> leave NULL     "unknown" is NOT the same as "perfect"; 0% late would be a false claim
--   * has_* flags          -> say clearly whether history exists at all

SELECT
    -- ---------- from application (the base table: keeps every applicant) ----------
    a.SK_ID_CURR,
    a.TARGET,                                 -- the label: 1 = payment difficulties
    a.NAME_CONTRACT_TYPE,
    CAST(FLOOR(-a.DAYS_BIRTH / 365.25) AS INTEGER) AS AGE_YEARS,          -- days since birth are stored as negative numbers
    CASE WHEN a.DAYS_EMPLOYED = 365243 THEN NULL ELSE -a.DAYS_EMPLOYED / 365.25 END AS YEARS_EMPLOYED,   -- 365243 = 'not employed' placeholder, NOT a tenure
    CASE WHEN a.DAYS_EMPLOYED = 365243 THEN 1 ELSE 0 END AS not_employed_flag,
    a.NAME_EDUCATION_TYPE,
    a.AMT_INCOME_TOTAL,
    a.AMT_CREDIT,
    a.AMT_ANNUITY,
    a.EXT_SOURCE_1,                           -- outside scores: NULL when missing (very common)
    a.EXT_SOURCE_2,
    a.EXT_SOURCE_3,
    a.AMT_CREDIT  / a.AMT_INCOME_TOTAL AS credit_income_ratio,    -- how big is the loan relative to income?
    a.AMT_ANNUITY / a.AMT_INCOME_TOTAL AS annuity_income_ratio,   -- how big is the yearly repayment relative to income?

    -- ---------- bureau (2.1) ----------
    CASE WHEN b.SK_ID_CURR IS NULL THEN 0 ELSE 1 END AS has_bureau,
    COALESCE(b.bureau_n_loans, 0)        AS bureau_n_loans,
    COALESCE(b.bureau_n_active, 0)       AS bureau_n_active,
    COALESCE(b.bureau_active_debt, 0)    AS bureau_active_debt,
    COALESCE(b.bureau_active_credit, 0)  AS bureau_active_credit,
    b.bureau_max_overdue_days,
    COALESCE(b.bureau_sum_overdue_amt, 0) AS bureau_sum_overdue_amt,
    COALESCE(b.bureau_n_recent_loans, 0)  AS bureau_n_recent_loans,
    COALESCE(b.bureau_n_overdue, 0)      AS bureau_n_overdue,
    b.bureau_utilisation,

    -- ---------- earlier applications with this bank (2.2) ----------
    CASE WHEN p.SK_ID_CURR IS NULL THEN 0 ELSE 1 END AS has_prev,
    COALESCE(p.prev_n_apps, 0)           AS prev_n_apps,
    COALESCE(p.prev_n_approved, 0)       AS prev_n_approved,
    COALESCE(p.prev_n_refused, 0)        AS prev_n_refused,
    COALESCE(p.prev_n_canceled, 0)       AS prev_n_canceled,
    COALESCE(p.prev_approved_credit, 0)  AS prev_approved_credit,
    p.prev_last_refused,
    p.prev_refusal_rate,

    -- ---------- payment behaviour (2.3) ----------
    CASE WHEN i.SK_ID_CURR IS NULL THEN 0 ELSE 1 END AS has_payments,
    COALESCE(i.inst_n_payments, 0)       AS inst_n_payments,
    COALESCE(i.inst_n_late, 0)           AS inst_n_late,
    i.inst_late_share,
    i.inst_avg_days_late,
    i.inst_max_days_late,
    i.inst_short_share,
    i.inst_recent3_late_share
FROM application a
LEFT JOIN bureau_feat b ON b.SK_ID_CURR = a.SK_ID_CURR
LEFT JOIN prev_feat   p ON p.SK_ID_CURR = a.SK_ID_CURR
LEFT JOIN pay_feat    i ON i.SK_ID_CURR = a.SK_ID_CURR;
