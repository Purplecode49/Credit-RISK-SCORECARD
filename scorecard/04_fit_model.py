"""
Stage 3.4: fit the logistic regression on the WOE values of the selected features (TRAIN set only).

  * every feature is replaced by the WOE of its bin
  * the model learns one weight per feature + a starting value (intercept)
  * a feature is removed if its weight has the WRONG SIGN or is not statistically significant (p > 0.05),
    one at a time, refitting after each removal (backward elimination)

Weights should all be NEGATIVE: a higher WOE means "safer", so it must push the default probability down.

Run:  python scorecard/04_fit_model.py     (from the credit_risk folder)
"""
import json, os, sys
import numpy as np, pandas as pd, duckdb
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(__file__))
from woe_binning import transform_woe

OUT = "scorecard/output"
P_LIMIT = 0.05

con = duckdb.connect("credit_risk.duckdb", read_only=True)
train = con.execute("SELECT * FROM train_set").df()
test = con.execute("SELECT * FROM test_set").df()
spec = json.load(open(f"{OUT}/bins.json"))
selected = [d["feature"] for d in json.load(open(f"{OUT}/selected_features.json"))]

Wtr = transform_woe(train, spec)[selected]
Wte = transform_woe(test, spec)[selected]
ytr, yte = train.TARGET.values, test.TARGET.values


def fit(cols):
    return sm.Logit(ytr, sm.add_constant(Wtr[cols])).fit(disp=0)


def table(m):
    t = pd.DataFrame({"weight": m.params, "std_err": m.bse, "p_value": m.pvalues})
    t["flag"] = ""
    t.loc[(t.index != "const") & (t.weight > 0), "flag"] = "WRONG SIGN"
    t.loc[(t.index != "const") & (t.p_value > P_LIMIT), "flag"] += " not significant"
    return t


def auc(score, y):
    r = pd.Series(score).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


cols = list(selected)
m = fit(cols)
print(f"FIRST FIT: all {len(cols)} selected features")
print(table(m).round(4).to_string())

steps = []
while True:
    t = table(m).drop("const")
    wrong = t[t.weight > 0]
    if len(wrong):
        drop, why = wrong.p_value.idxmax(), "wrong sign"
    elif t.p_value.max() > P_LIMIT:
        drop, why = t.p_value.idxmax(), f"p-value {t.p_value.max():.3f} above {P_LIMIT}"
    else:
        break
    steps.append((drop, why))
    cols.remove(drop)
    m = fit(cols)

print("\nREMOVED, ONE AT A TIME")
for d, why in steps:
    print(f"  - {d}: {why}")
if not steps:
    print("  (nothing needed removing)")

final = table(m)
print(f"\nFINAL MODEL: {len(cols)} features")
print(final.round(4).to_string())

# how much does each feature move the score? weight x spread of its WOE values
spread = Wtr[cols].std()
imp = (final.weight.drop("const").abs() * spread)
imp = (100 * imp / imp.sum()).sort_values(ascending=False).round(1)
print("\nSHARE OF THE MODEL'S MOVEMENT (|weight| x spread of WOE), %")
print(imp.to_string())

# predictions
Xtr = sm.add_constant(Wtr[cols], has_constant="add")
Xte = sm.add_constant(Wte[cols], has_constant="add")
pd_tr, pd_te = m.predict(Xtr).values, m.predict(Xte).values
print("\nSANITY CHECKS")
print(f"average predicted PD, train: {100*pd_tr.mean():.2f}% vs actual default rate {100*ytr.mean():.2f}%")
print(f"average predicted PD, test : {100*pd_te.mean():.2f}% vs actual default rate {100*yte.mean():.2f}%")
print(f"AUC train {auc(pd_tr, ytr):.3f} | AUC test {auc(pd_te, yte):.3f}   (full evaluation comes in 3.6)")
print(f"lowest and highest predicted PD on test: {100*pd_te.min():.2f}% and {100*pd_te.max():.2f}%")

# hand check for one person: intercept + sum(weight x WOE), then the logistic step
i = 0
row = Wtr[cols].iloc[i]
total = final.weight["const"] + (final.weight.drop("const") * row).sum()
by_hand = 1 / (1 + np.exp(-total))
print(f"\nHAND CHECK, applicant {train.SK_ID_CURR.iloc[i]}: total = {total:.4f} -> PD = 1/(1+e^-total) = {100*by_hand:.3f}% | model says {100*pd_tr[i]:.3f}%")

# save
json.dump({"features": cols, "intercept": float(final.weight["const"]),
           "weights": {c: float(final.weight[c]) for c in cols}, "removed": steps},
          open(f"{OUT}/model.json", "w"), indent=1)
open(f"{OUT}/model_summary.txt", "w").write(str(m.summary()))
print(f"\nsaved {OUT}/model.json and model_summary.txt")
