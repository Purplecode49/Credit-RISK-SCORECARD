"""Stage 6.2b (real data): fill dashboard/template.html with the numbers from scorecard/output and write dashboard/dashboard.html"""
import json, pandas as pd, duckdb
OUT = "scorecard/output"
monthly = pd.read_csv(f"{OUT}/monitoring_monthly.csv")
fm = pd.read_csv(f"{OUT}/feature_psi_monthly.csv")
bands = pd.read_csv(f"{OUT}/el_by_band.csv")
stress = pd.read_csv(f"{OUT}/stress_scenarios.csv")
sens = pd.read_csv(f"{OUT}/lgd_sensitivity.csv")
calib = pd.read_csv(f"{OUT}/calibration_check.csv")
models = pd.read_csv(f"{OUT}/model_comparison.csv")
metrics = json.load(open(f"{OUT}/scorecard_metrics.json"))
psi = json.load(open(f"{OUT}/psi_summary.json"))
loans = pd.read_csv(f"{OUT}/expected_loss_loans.csv")
assump = json.load(open(f"{OUT}/el_assumptions.json"))
con = duckdb.connect("credit_risk.duckdb", read_only=True)
n_train, n_test = (con.execute(f"select count(*) from {t}").fetchone()[0] for t in ("train_set", "test_set"))

drift_feature = psi["tilt_feature"]
drift_months = set(monthly[monthly.dataset == "drifting"].month)
fd = fm[fm.month.isin(drift_months)]
ft = fd.groupby("feature").psi.max().rename("max_test").reset_index()
ft["months_over"] = ft.feature.map(fd.assign(o=lambda d: d.psi > 0.10).groupby("feature").o.sum())
ft["status"] = ft.max_test.map(lambda v: "Stable" if v < 0.10 else ("Watch" if v < 0.25 else "Major shift"))
tv = pd.read_csv(f"{OUT}/feature_psi_test_vs_train.csv").rename(columns={"psi_test_vs_train": "psi_whole"})
ft = ft.merge(tv[["feature", "psi_whole"]], on="feature").sort_values("max_test", ascending=False)

sev = loans.groupby("band")[["expected_loss", "expected_loss_severe"]].sum()
extra = sev.expected_loss_severe - sev.expected_loss
mid = extra[["500-519", "520-539", "540-559", "560-579"]].sum() / extra.sum() * 100
low = bands.band.isin(["under 500", "500-519"])
m = models.set_index("model")
sc_auc, gb_auc, gb_same = m.loc["Scorecard", "test_AUC"], m.loc["GB on all features", "test_AUC"], m.loc["GB on scorecard features", "test_AUC"]
note = (f"Boosting on every feature scores {gb_auc - sc_auc:+.3f} AUC against the scorecard ({gb_auc:.3f} against {sc_auc:.3f}). Given the same features as the scorecard it gains only "
        f"{gb_same - sc_auc:+.3f}, so most of the gain comes from extra inputs and missing-value handling, not from the model type. Its train-to-hold-out gap is also larger "
        f"({m.loc['GB on all features', 'train_minus_test_AUC']:.3f} against {m.loc['Scorecard', 'train_minus_test_AUC']:.3f}). The scorecard stays the explainable choice.")
cal = dict(before=float(calib.test_pct.iloc[0]), actual=float(calib.test_pct.iloc[2]))
drift = fm.pivot(index="month", columns="feature", values="psi")[drift_feature]
data = {
  "monthly": monthly.to_dict("records"),
  "drift_psi": {k: float(v) for k, v in drift.items()},
  "drift_feature": drift_feature,
  "features": ft.to_dict("records"),
  "bands": bands.to_dict("records"),
  "stress": stress.to_dict("records"),
  "sens": sens.to_dict("records"),
  "models": models.to_dict("records"),
  "metrics": metrics,
  "calib": cal,
  "pd12_avg": float(loans.pd.mean() * 100),
  "score_psi_test": psi["score_psi_test"],
  "score_psi_max": psi["score_psi_max_month"],
  "sev_mid_share": float(mid),
  "low_share_loans": float(bands.loc[low, "loans"].sum() / bands.loans.sum() * 100),
  "low_share_el": float(bands.loc[low, "share_of_EL_pct"].sum()),
  "n_train": int(n_train), "n_test": int(n_test),
  "model_note": note,
  "assump": assump,
}
html = open("dashboard/template.html").read().replace("/*__DATA__*/null", json.dumps(data, default=float))
open("dashboard/dashboard.html", "w").write(html)
print(len(html), "bytes |", note)
