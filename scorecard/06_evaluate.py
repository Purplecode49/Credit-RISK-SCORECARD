"""
Stage 3.6: evaluate the scorecard on train AND test.

  AUC   = chance a random bad scores lower than a random good (0.5 = coin flip, 1 = perfect)
  Gini  = 2 * AUC - 1
  KS    = biggest gap between the share of bads and the share of goods at or below a score cut-off
Also: score bands on test, and predicted vs actual default rate (calibration).

Run:  python scorecard/06_evaluate.py     (from the credit_risk folder)
"""
import json
import numpy as np, pandas as pd

OUT = "scorecard/output"
scores = pd.read_csv(f"{OUT}/scores.csv")


def auc(score, y):
    """Higher score = safer, so AUC is computed on the negative score (bad = positive class)."""
    r = pd.Series(-score).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ks(score, y):
    d = pd.DataFrame({"s": score, "y": y}).sort_values("s")
    cum_bad = d.y.cumsum() / d.y.sum()
    cum_good = (1 - d.y).cumsum() / (1 - d.y).sum()
    gap = (cum_bad - cum_good)
    return gap.max(), d.s.values[gap.values.argmax()]


rows = {}
for name in ("train", "test"):
    d = scores[scores.dataset == name]
    a = auc(d.score_exact.values, d.TARGET.values)
    k, cut = ks(d.score_exact.values, d.TARGET.values)
    rows[name] = {"people": len(d), "default_rate_pct": 100 * d.TARGET.mean(), "avg_pd_pct": 100 * d.pd.mean(),
                  "AUC": a, "Gini": 2 * a - 1, "KS": k, "KS_cutoff_score": cut}
res = pd.DataFrame(rows).T
print("DISCRIMINATION AND CALIBRATION")
print(res.round(3).to_string())
json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in rows.items()}, open(f"{OUT}/scorecard_metrics.json", "w"), indent=1)

edges = [-np.inf, 500, 520, 540, 560, 580, 600, np.inf]
labs = ["under 500", "500-519", "520-539", "540-559", "560-579", "580-599", "600 and over"]
out = []
for name in ("train", "test"):
    d = scores[scores.dataset == name].copy()
    d["band"] = pd.cut(d.score, bins=edges, labels=labs, right=False)
    g = d.groupby("band", observed=True).agg(people=("TARGET", "size"), defaults=("TARGET", "sum"),
                                              avg_pd_pct=("pd", lambda s: 100 * s.mean()))
    g["default_rate_pct"] = 100 * g.defaults / g.people
    g["share_of_people_pct"] = 100 * g.people / g.people.sum()
    g.insert(0, "dataset", name)
    out.append(g.reset_index())
bands = pd.concat(out)
bands.to_csv(f"{OUT}/score_bands_train_test.csv", index=False)
t = bands[bands.dataset == "test"].drop(columns="dataset")
print("\nTEST SET BY SCORE BAND")
print(t.round(2).to_string(index=False))

# is the ordering kept? default rate must fall as score rises
print("\nordering kept on test (default rate falls band by band):", bool(t.default_rate_pct.is_monotonic_decreasing))

# drift in who applies: how many test people fall in each band vs train
tr = bands[bands.dataset == "train"].set_index("band").share_of_people_pct
te = bands[bands.dataset == "test"].set_index("band").share_of_people_pct
print("\nSHARE OF PEOPLE IN EACH BAND (%), train vs test")
print(pd.DataFrame({"train": tr, "test": te}).round(1).to_string())
print(f"\nmean score: train {scores[scores.dataset=='train'].score.mean():.1f}, test {scores[scores.dataset=='test'].score.mean():.1f}")
