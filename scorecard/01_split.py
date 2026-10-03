"""
Stage 3.1 (real data): split the feature table into a training set and a hold-out set.

The real Home Credit data has NO application date, so a time-based split is impossible. Instead:
  70% TRAIN  - the model learns from these
  30% TEST   - a random hold-out the model never sees while it is being built
Both parts keep the same default rate (stratified by TARGET). This checks out-of-sample performance, but it is NOT an
out-of-time test: it cannot show how the model copes with a changing population.

Run:  python scorecard/01_split.py
"""
import duckdb, numpy as np, pandas as pd

SEED, TEST_SHARE = 42, 0.30
con = duckdb.connect("credit_risk.duckdb")
ft = con.execute("SELECT SK_ID_CURR, TARGET FROM feature_table").df()
rng = np.random.default_rng(SEED)
is_test = np.zeros(len(ft), dtype=bool)
for y in (0, 1):                                              # stratify: split the defaulters and non-defaulters separately
    idx = np.flatnonzero(ft.TARGET.values == y)
    is_test[rng.choice(idx, int(round(TEST_SHARE * len(idx))), replace=False)] = True
ids = pd.DataFrame({"SK_ID_CURR": ft.SK_ID_CURR, "is_test": is_test})
con.execute("CREATE OR REPLACE TABLE split_ids AS SELECT * FROM ids")
con.execute("CREATE OR REPLACE TABLE train_set AS SELECT f.* FROM feature_table f JOIN split_ids s USING (SK_ID_CURR) WHERE NOT s.is_test")
con.execute("CREATE OR REPLACE TABLE test_set  AS SELECT f.* FROM feature_table f JOIN split_ids s USING (SK_ID_CURR) WHERE s.is_test")
for name in ("train_set", "test_set"):
    n, dr = con.execute(f"SELECT COUNT(*), ROUND(100.0*AVG(TARGET),2) FROM {name}").fetchone()
    print(f"{name:10s} rows={n:>8,}  default rate={dr}%")
assert con.execute("SELECT COUNT(*) FROM train_set t JOIN test_set s USING (SK_ID_CURR)").fetchone()[0] == 0, "overlap between train and test"
con.close()
