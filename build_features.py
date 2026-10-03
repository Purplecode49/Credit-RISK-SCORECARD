"""
Stage 2 runner: saves the three summaries as views, then builds the feature table.

Run:  python build_features.py
(In MySQL you would do the same with CREATE VIEW ... AS <query> and CREATE TABLE feature_table AS <query>.)
"""
import duckdb

def read_sql(path):
    return open(path).read().strip().rstrip(";")

con = duckdb.connect("credit_risk.duckdb")

views = {
    "bureau_feat": "sql/01_bureau_features.sql",
    "prev_feat":   "sql/02_previous_application_features.sql",
    "pay_feat":    "sql/03_payment_features.sql",
}
for name, path in views.items():
    con.execute(f"CREATE OR REPLACE VIEW {name} AS {read_sql(path)}")
    print(f"view  {name:12s} created from {path}")

con.execute(f"CREATE OR REPLACE TABLE feature_table AS {read_sql('sql/04_feature_table.sql')}")
rows, cols = con.execute("SELECT COUNT(*) FROM feature_table").fetchone()[0], len(con.execute("SELECT * FROM feature_table LIMIT 0").df().columns)
print(f"table feature_table created: {rows:,} rows x {cols} columns")
con.close()
