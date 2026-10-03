"""
Stage 6.1 (real data): drift monitoring with PSI (Population Stability Index).

  PSI = sum over bins of (share_now - share_reference) x ln(share_now / share_reference)
  < 0.10 stable | 0.10 - 0.25 some shift, look into it | > 0.25 major shift

The real data has no dates, so there is no real monthly history. Two things are done instead:
  1. REAL CHECK   hold-out vs training: PSI should be close to 0, because both are random halves of the same population.
  2. SIMULATION   twelve "months" of 3,000 loans are drawn from the hold-out. Months 1-6 are plain random draws (stable).
                  From month 7 the draws are tilted toward applicants with a LOW EXT_SOURCE_2, as if the outside score
                  supplier's population had shifted. In months 10-12 extra defaults are added (+1, +2, +3 points) to mimic
                  a downturn the model cannot see. EVERYTHING in the monthly table is SIMULATED.

Run:  python scorecard/09_psi_monitoring.py     (from the project folder)
"""
import json, os, sys
import numpy as np, pandas as pd, duckdb

sys.path.insert(0, os.path.dirname(__file__))
from woe_binning import assign_bins

OUT = "scorecard/output"
EPS, N_MONTH, SEED = 0.0005, 3000, 7
LAMBDA = [0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]            # tilt strength per simulated month
EXTRA_DEFAULT_PP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3]              # extra defaults added to the labels (percentage points)

con = duckdb.connect("credit_risk.duckdb", read_only=True)
train = con.execute("SELECT * FROM train_set").df()
test = con.execute("SELECT * FROM test_set").df()
sc = pd.read_csv(f"{OUT}/scores.csv")[["SK_ID_CURR", "score", "score_exact", "pd"]]
train, test = train.merge(sc, on="SK_ID_CURR"), test.merge(sc, on="SK_ID_CURR")
spec = json.load(open(f"{OUT}/bins.json"))
feats = json.load(open(f"{OUT}/model.json"))["features"]


def psi(r, n):
    r, n = np.maximum(r, EPS), np.maximum(n, EPS)
    return float(((n - r) * np.log(n / r)).sum())


status = lambda v: "Stable" if v < 0.10 else ("Watch" if v < 0.25 else "Major shift")
edges = np.quantile(train.score_exact, np.linspace(0, 1, 11)[1:-1])
cut = lambda s: np.digitize(s, edges)
ref_score = np.bincount(cut(train.score_exact), minlength=10) / len(train)
ref_feat, test_bins = {}, {}
for f in feats:
    lab_tr, lab_te = assign_bins(train[f], spec[f]), assign_bins(test[f], spec[f])
    order = list(dict.fromkeys(list(spec[f]["bins"]) + list(lab_tr.unique()) + list(lab_te.unique())))
    ref_feat[f] = lab_tr.value_counts(normalize=True).reindex(order).fillna(0)
    test_bins[f] = lab_te

# ---------------------------------------------------------------- 1. real check: hold-out vs training
tot = psi(ref_score, np.bincount(cut(test.score_exact), minlength=10) / len(test))
fr = pd.DataFrame([{"feature": f, "psi_test_vs_train": psi(ref_feat[f].values,
                     test_bins[f].value_counts(normalize=True).reindex(ref_feat[f].index).fillna(0).values)} for f in feats])
fr["status"] = fr.psi_test_vs_train.map(status)
fr.sort_values("psi_test_vs_train", ascending=False).to_csv(f"{OUT}/feature_psi_test_vs_train.csv", index=False)
print(f"REAL CHECK, hold-out vs training: score PSI {tot:.5f} ({status(tot)}) | largest feature PSI {fr.psi_test_vs_train.max():.5f}")

# ---------------------------------------------------------------- 2. simulation
rng = np.random.default_rng(SEED)
x = test.EXT_SOURCE_2.fillna(test.EXT_SOURCE_2.median()).values
z = (x - x.mean()) / x.std()
rows, frows = [], []
for m in range(12):
    w = np.exp(-LAMBDA[m] * z); w /= w.sum()
    idx = rng.choice(len(test), N_MONTH, replace=False, p=w)
    g = test.iloc[idx]
    y = g.TARGET.values.copy()
    k = int(round(EXTRA_DEFAULT_PP[m] / 100 * N_MONTH))
    if k:
        zeros = np.flatnonzero(y == 0); y[rng.choice(zeros, k, replace=False)] = 1
    label = f"S{m+1:02d}"
    sp = psi(ref_score, np.bincount(cut(g.score_exact), minlength=10) / len(g))
    rows.append({"month": label, "dataset": "baseline" if m < 6 else "drifting", "applications": N_MONTH, "avg_score": g.score.mean(),
                 "score_psi": sp, "avg_pd_pct": 100 * g.pd.mean(), "actual_default_rate_pct": 100 * y.mean(),
                 "gap_pp": 100 * (y.mean() - g.pd.mean()), "status": status(sp)})
    for f in feats:
        sh = test_bins[f].iloc[idx].value_counts(normalize=True).reindex(ref_feat[f].index).fillna(0).values
        v = psi(ref_feat[f].values, sh)
        frows.append({"month": label, "feature": f, "psi": v, "status": status(v)})
monthly, fm = pd.DataFrame(rows), pd.DataFrame(frows)
monthly.to_csv(f"{OUT}/monitoring_monthly.csv", index=False)
fm.to_csv(f"{OUT}/feature_psi_monthly.csv", index=False)

pd.set_option("display.width", 200)
print("\nSIMULATED MONTHLY MONITORING"); print(monthly.round(3).to_string(index=False))
piv = fm.pivot(index="month", columns="feature", values="psi")
print("\nFEATURE PSI, simulated months: max and months over 0.10")
print(pd.DataFrame({"max_psi": piv.max(), "months_over_0.10": (piv > 0.10).sum()}).sort_values("max_psi", ascending=False).round(3).to_string())
print("\nSANITY: training vs itself", round(psi(ref_score, np.bincount(cut(train.score_exact), minlength=10) / len(train)), 6),
      "| train scores moved down 20 points ->", round(psi(ref_score, np.bincount(cut(train.score_exact - 20), minlength=10) / len(train)), 4))
json.dump({"score_psi_test": tot, "score_psi_max_month": float(monthly.score_psi.max()), "simulated": True,
           "lambda": LAMBDA, "extra_default_pp": EXTRA_DEFAULT_PP, "tilt_feature": "EXT_SOURCE_2", "cohort_size": N_MONTH},
          open(f"{OUT}/psi_summary.json", "w"), indent=1)
