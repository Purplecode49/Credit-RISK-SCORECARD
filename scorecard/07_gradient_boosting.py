"""
Stage 4: gradient-boosting benchmark on the SAME random train/hold-out split as the scorecard.

  * model A: gradient boosting on all 34 raw features (what a "no rules" model could use)
  * model B: gradient boosting on the scorecard's 8 features (same information, so it isolates the model type)
  * scorecard: the points model from Stage 3 (scores.csv)
Settings are chosen on a validation slice = a random 20% of the training set. The test set is touched once, at the end.

Run:  python scorecard/07_gradient_boosting.py     (from the credit_risk folder)
"""
import itertools, json
import numpy as np, pandas as pd, duckdb
from sklearn.ensemble import HistGradientBoostingClassifier

OUT = "scorecard/output"
con = duckdb.connect("credit_risk.duckdb", read_only=True)
train = con.execute("SELECT * FROM train_set ORDER BY SK_ID_CURR").df().sample(frac=1, random_state=42).reset_index(drop=True)   # random order: there is no application date
test = con.execute("SELECT * FROM test_set").df()
model = json.load(open(f"{OUT}/model.json"))

DROP = ["SK_ID_CURR", "TARGET"]
ALL = [c for c in train.columns if c not in DROP]
EIGHT = model["features"]
CATS = ["NAME_CONTRACT_TYPE", "NAME_EDUCATION_TYPE"]


def prep(df, cols):
    X = df[cols].copy()
    for c in cols:
        if c in CATS:
            X[c] = X[c].astype("category")
    return X


def auc(score, y):
    r = pd.Series(score).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ks(score, y):
    d = pd.DataFrame({"s": score, "y": y}).sort_values("s", ascending=False)
    return ((d.y.cumsum() / d.y.sum()) - ((1 - d.y).cumsum() / (1 - d.y).sum())).max()


def metrics(p, y):
    a = auc(p, y)
    return {"AUC": a, "Gini": 2 * a - 1, "KS": ks(p, y)}


cut = int(len(train) * 0.8)
fit_part, valid_part = train.iloc[:cut], train.iloc[cut:]
GRID = [dict(max_depth=d, learning_rate=lr, min_samples_leaf=m)
        for d, lr, m in itertools.product([3, 5, 8], [0.08], [100, 400])]


def tune(cols):
    best = None
    for g in GRID:
        m = HistGradientBoostingClassifier(max_iter=500, early_stopping=False, l2_regularization=1.0,
                                           categorical_features="from_dtype", random_state=42, **g)
        # pick the number of trees on the validation slice
        m.set_params(max_iter=500)
        m.fit(prep(fit_part, cols), fit_part.TARGET)
        stage_auc = [auc(p[:, 1], valid_part.TARGET.values) for p in m.staged_predict_proba(prep(valid_part, cols))]
        k = int(np.argmax(stage_auc))
        if best is None or stage_auc[k] > best["valid_auc"]:
            best = {"params": g, "n_trees": k + 1, "valid_auc": stage_auc[k]}
    return best


results, tables = {}, {}
for name, cols in (("GB all features", ALL), ("GB on scorecard features", EIGHT)):
    b = tune(cols)
    print(f"{name}: chosen {b['params']}, {b['n_trees']} trees, validation AUC {b['valid_auc']:.4f}")
    m = HistGradientBoostingClassifier(max_iter=b["n_trees"], early_stopping=False, l2_regularization=1.0,
                                       categorical_features="from_dtype", random_state=42, **b["params"])
    m.fit(prep(train, cols), train.TARGET)                       # refit on the WHOLE training set
    p_tr, p_te = m.predict_proba(prep(train, cols))[:, 1], m.predict_proba(prep(test, cols))[:, 1]
    results[name] = {"train": metrics(p_tr, train.TARGET.values), "test": metrics(p_te, test.TARGET.values),
                     "avg_pd_test_pct": 100 * p_te.mean()}
    tables[name] = pd.DataFrame({"SK_ID_CURR": test.SK_ID_CURR, "TARGET": test.TARGET, "pd": p_te})

sc = pd.read_csv(f"{OUT}/scores.csv")
for ds, df in (("train", train), ("test", test)):
    d = sc[sc.dataset == ds]
    results.setdefault("Scorecard", {"avg_pd_test_pct": 100 * sc[sc.dataset == "test"].pd.mean()})[ds] = \
        metrics(-d.score_exact.values, d.TARGET.values)

rows = []
for name in ("Scorecard", "GB on scorecard features", "GB all features"):
    r = results[name]
    rows.append({"model": name, "train_AUC": r["train"]["AUC"], "test_AUC": r["test"]["AUC"], "test_Gini": r["test"]["Gini"],
                 "test_KS": r["test"]["KS"], "train_minus_test_AUC": r["train"]["AUC"] - r["test"]["AUC"],
                 "avg_pd_test_pct": r["avg_pd_test_pct"]})
comp = pd.DataFrame(rows)
print("\nCOMPARISON (test = hold-out applicants the models never saw; actual test default rate "
      f"{100*test.TARGET.mean():.2f}%)")
print(comp.round(3).to_string(index=False))
comp.to_csv(f"{OUT}/model_comparison.csv", index=False)

# does the scorecard's ranking agree with boosting's? and who do they disagree on?
gb = tables["GB all features"].set_index("SK_ID_CURR").pd
scd = sc[sc.dataset == "test"].set_index("SK_ID_CURR")
both = pd.DataFrame({"sc": scd.pd, "gb": gb}).dropna()
print(f"\nrank correlation between scorecard PD and boosting PD on test: {both.corr(method='spearman').iloc[0,1]:.3f}")

# top-risk slice: of the riskiest 10% by each model, what share actually defaulted?
y = scd.TARGET.reindex(both.index)
for lab, col in (("scorecard", "sc"), ("boosting (34)", "gb")):
    top = both[col].nlargest(int(0.1 * len(both))).index
    print(f"riskiest 10% by {lab:14s}: {100*y[top].mean():.1f}% defaulted (portfolio average {100*y.mean():.1f}%)")

tables["GB all features"].to_csv(f"{OUT}/gb_test_predictions.csv", index=False)
