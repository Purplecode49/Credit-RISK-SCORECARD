"""Stage 6.2a (real data): tidy, import-ready tables for Power BI (Get Data > Text/CSV, one file per table)."""
import os, pandas as pd, duckdb
OUT, PBI = "scorecard/output", "powerbi"
os.makedirs(PBI, exist_ok=True)
con = duckdb.connect("credit_risk.duckdb", read_only=True)
app = con.execute("SELECT SK_ID_CURR, NAME_CONTRACT_TYPE, AMT_CREDIT, AMT_INCOME_TOTAL, AGE_YEARS FROM feature_table").df()
sc = pd.read_csv(f"{OUT}/scores.csv")
el = pd.read_csv(f"{OUT}/expected_loss_loans.csv")[["SK_ID_CURR", "pd", "lgd", "ead", "expected_loss", "expected_loss_severe"]].rename(columns={"pd": "pd_12m"})
bands = pd.DataFrame({"score_band": ["under 500", "500-519", "520-539", "540-559", "560-579", "580-599", "600 and over"],
                      "band_order": range(1, 8), "score_from": [0, 500, 520, 540, 560, 580, 600], "score_to": [499, 519, 539, 559, 579, 599, 999]})
sc["score_band"] = pd.cut(sc.score, bins=[-1e9, 500, 520, 540, 560, 580, 600, 1e9], labels=bands.score_band.tolist(), right=False).astype(str)
loans = (sc.merge(app, on="SK_ID_CURR").merge(el, on="SK_ID_CURR", how="left")
           .rename(columns={"SK_ID_CURR": "loan_id", "TARGET": "defaulted", "pd": "pd_scorecard", "NAME_CONTRACT_TYPE": "contract_type"}))
loans = loans[["loan_id", "dataset", "contract_type", "AGE_YEARS", "AMT_CREDIT", "AMT_INCOME_TOTAL", "score", "score_band", "pd_scorecard",
               "defaulted", "pd_12m", "lgd", "ead", "expected_loss", "expected_loss_severe"]]
loans.round(6).to_csv(f"{PBI}/fact_loans.csv", index=False)
bands.to_csv(f"{PBI}/dim_score_band.csv", index=False)
for src, dst in [("monitoring_monthly", "fact_simulated_monitoring_monthly"), ("feature_psi_monthly", "fact_simulated_feature_psi_monthly"),
                 ("feature_psi_test_vs_train", "fact_feature_psi_holdout_vs_train"), ("stress_scenarios", "fact_stress_scenarios"),
                 ("el_by_band", "fact_el_by_band"), ("lgd_sensitivity", "fact_lgd_ccf_sensitivity"),
                 ("calibration_check", "fact_calibration_check"), ("scorecard_points", "scorecard_points"), ("model_comparison", "model_comparison")]:
    pd.read_csv(f"{OUT}/{src}.csv").round(6).to_csv(f"{PBI}/{dst}.csv", index=False)
print({f: len(pd.read_csv(f"{PBI}/{f}")) for f in sorted(os.listdir(PBI)) if f.endswith(".csv")})
