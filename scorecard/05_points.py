"""
Stage 3.5: turn the logistic regression into a POINTS scorecard.

Scaling (the classic convention):
    BASE_SCORE points at good:bad odds of BASE_ODDS to 1, and every PDO points DOUBLE the good:bad odds.
    factor = PDO / ln(2)
    offset = BASE_SCORE - factor * ln(BASE_ODDS)
    score  = offset + factor * ln(good/bad odds)  =  offset - factor * (intercept + sum(weight_i * WOE_i))

Points of one bin of feature i:
    points = -(weight_i * WOE_bin + intercept / n_features) * factor + offset / n_features
(the intercept and offset are shared out equally, so the points of one person's bins simply ADD UP to the score)

Run:  python scorecard/05_points.py     (from the credit_risk folder)
"""
import json, os, sys
import numpy as np, pandas as pd, duckdb

sys.path.insert(0, os.path.dirname(__file__))
from woe_binning import transform_woe, assign_bins

OUT = "scorecard/output"
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20
factor = PDO / np.log(2)
offset = BASE_SCORE - factor * np.log(BASE_ODDS)

model = json.load(open(f"{OUT}/model.json"))
spec = json.load(open(f"{OUT}/bins.json"))
feats, b0, w = model["features"], model["intercept"], model["weights"]
n = len(feats)

con = duckdb.connect("credit_risk.duckdb", read_only=True)
train = con.execute("SELECT * FROM train_set").df()
test = con.execute("SELECT * FROM test_set").df()
woe_tab = pd.read_csv(f"{OUT}/woe_tables.csv")

# ------------------------------------------------------------- the points table
rows = []
for f in feats:
    for label, woe in zip(spec[f]["bins"], spec[f]["woe"]):
        exact = -(w[f] * woe + b0 / n) * factor + offset / n
        rows.append({"feature": f, "bin": label, "woe": woe, "points_exact": exact, "points": int(round(exact))})
card = pd.DataFrame(rows)
stats = woe_tab[woe_tab.feature.isin(feats)][["feature", "bin", "people", "bad_rate_pct"]]
card = card.merge(stats, on=["feature", "bin"], how="left")[["feature", "bin", "people", "bad_rate_pct", "woe", "points"]]
card.to_csv(f"{OUT}/scorecard_points.csv", index=False)
json.dump({"base_score": BASE_SCORE, "base_odds": BASE_ODDS, "pdo": PDO, "factor": factor, "offset": offset},
          open(f"{OUT}/scaling.json", "w"), indent=1)
print(f"factor = {factor:.4f} | offset = {offset:.4f}  (base {BASE_SCORE} points at {BASE_ODDS}:1 good:bad, +{PDO} points doubles the odds)")

lookup = {(r.feature, r.bin): r.points for r in card.itertuples()}
neutral = {f: int(round(-(b0 / n) * factor + offset / n)) for f in feats}          # points of a WOE-0 (average) bin


def score_people(df):
    pts = pd.DataFrame({f: [lookup.get((f, b), neutral[f]) for b in assign_bins(df[f], spec[f])] for f in feats}, index=df.index)
    exact = offset - factor * (b0 + sum(w[f] * transform_woe(df, {f: spec[f]})[f] for f in feats))
    return pts, pts.sum(axis=1), exact


def pd_from_score(s):
    return 1 / (1 + np.exp((s - offset) / factor))


res = []
for name, df in (("train", train), ("test", test)):
    pts, s_int, s_exact = score_people(df)
    res.append(pd.DataFrame({"SK_ID_CURR": df.SK_ID_CURR, "dataset": name, "TARGET": df.TARGET,
                             "score": s_int.values, "score_exact": s_exact.values,
                             "pd": pd_from_score(s_exact.values)}))
    if name == "train":
        first_pts, first_id = pts.iloc[0], df.SK_ID_CURR.iloc[0]
scores = pd.concat(res, ignore_index=True)
scores.to_csv(f"{OUT}/scores.csv", index=False)

# ------------------------------------------------------------- checks
print("\nPOINTS RANGE PER FEATURE (best bin minus worst bin = how far this feature can move a score)")
span = card.groupby("feature").points.agg(lowest="min", highest="max")
span["range"] = span.highest - span.lowest
print(span.sort_values("range", ascending=False).to_string())
print(f"neutral (WOE = 0) points per feature: {neutral[feats[0]]}-ish; best possible total {int(span.highest.sum())}, worst possible total {int(span.lowest.sum())}")

print("\nHAND CHECK: applicant", first_id)
b = {f: assign_bins(train[f].iloc[[0]], spec[f]).iloc[0] for f in feats}
for f in feats:
    print(f"   {f:26s} bin {b[f]:>22s} -> {first_pts[f]:>3d} points")
print(f"   sum of points = {int(first_pts.sum())} | exact score from the model = {scores.score_exact.iloc[0]:.2f}")

print("\nROUNDING: largest gap between the whole-number score and the exact score:", round((scores.score - scores.score_exact).abs().max(), 2), "points")
print("DOUBLING CHECK on the exact formula:")
for s in (520, 560, 600):
    p1, p2 = pd_from_score(s), pd_from_score(s + PDO)
    print(f"   score {s}: PD {100*p1:.2f}%, good:bad {(1-p1)/p1:.2f} | score {s+PDO}: PD {100*p2:.2f}%, good:bad {(1-p2)/p2:.2f} | ratio {((1-p2)/p2)/((1-p1)/p1):.3f}")

print("\nSCORE DISTRIBUTION")
print(scores.groupby("dataset").score.describe().round(1).to_string())

tr = scores[scores.dataset == "train"].copy()
edges = [-np.inf, 500, 520, 540, 560, 580, 600, np.inf]
labs = ["under 500", "500-519", "520-539", "540-559", "560-579", "580-599", "600 and over"]
tr["band"] = pd.cut(tr.score, bins=edges, labels=labs, right=False)
g = tr.groupby("band", observed=True).agg(people=("TARGET", "size"), defaults=("TARGET", "sum"), avg_pd_pct=("pd", lambda s: 100 * s.mean()))
g["default_rate_pct"] = 100 * g.defaults / g.people
g["good_to_bad"] = (g.people - g.defaults) / g.defaults
g["odds_ratio_vs_band_below"] = g.good_to_bad / g.good_to_bad.shift(1)
print("\nTRAIN SET BY SCORE BAND")
print(g.round(2).to_string())
print("\nsaved: scorecard_points.csv, scores.csv, scaling.json")
