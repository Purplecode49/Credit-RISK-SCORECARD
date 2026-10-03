"""
Stage 3.2: learn the bins and WOE of every feature from the TRAINING set, and rank the features by IV.

Run:  python scorecard/02_fit_bins.py          (from the credit_risk folder)
Writes: scorecard/output/bins.json, woe_tables.csv, iv_ranking.csv
"""
import json, os, sys
import duckdb, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from woe_binning import fit_binning, bin_table

os.makedirs("scorecard/output", exist_ok=True)
con = duckdb.connect("credit_risk.duckdb", read_only=True)
train = con.execute("SELECT * FROM train_set").df()

NOT_FEATURES = {"SK_ID_CURR", "TARGET"}
features = [c for c in train.columns if c not in NOT_FEATURES]
print(f"training rows: {len(train):,} | features to bin: {len(features)}")

spec = fit_binning(train, features)

# long table of every bin of every feature
tables = []
for f in features:
    t = bin_table(train, f, spec)
    t.insert(0, "feature", f)
    tables.append(t)
woe_tables = pd.concat(tables, ignore_index=True)
woe_tables.to_csv("scorecard/output/woe_tables.csv", index=False)

def strength(iv):
    return ("not useful" if iv < 0.02 else "weak" if iv < 0.1 else "medium" if iv < 0.3
            else "strong" if iv < 0.5 else "suspiciously strong")

rank = pd.DataFrame({
    "feature": features,
    "iv": [spec[f]["iv"] for f in features],
    "n_bins": [len(spec[f]["bins"]) for f in features],
    "has_missing_bin": ["Missing" in spec[f]["bins"] for f in features],
}).sort_values("iv", ascending=False).reset_index(drop=True)
rank["strength"] = rank.iv.map(strength)
rank.to_csv("scorecard/output/iv_ranking.csv", index=False)

with open("scorecard/output/bins.json", "w") as fh:
    json.dump(spec, fh, indent=1)

print(rank.round(4).to_string())
