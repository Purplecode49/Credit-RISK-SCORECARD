"""
Stage 1 (real data): load the four real Home Credit files into DuckDB, keeping only the columns the project uses
and only the applicants that are in application_train.csv.

Run:  python load_real_data.py        (files in ./data, or set HC_DATA=/path/to/folder)
"""
import duckdb, os

DATA = os.environ.get("HC_DATA", "data")          # folder holding the four Kaggle CSV files
UP = DATA
INST = f"{DATA}/installments_payments.csv"
con = duckdb.connect("credit_risk.duckdb")
con.execute("PRAGMA threads=4")

con.execute(f"""CREATE OR REPLACE TABLE application AS
  SELECT SK_ID_CURR, TARGET, NAME_CONTRACT_TYPE, DAYS_BIRTH, DAYS_EMPLOYED, NAME_EDUCATION_TYPE,
         AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
  FROM read_csv_auto('{UP}/application_train.csv')""")
con.execute(f"""CREATE OR REPLACE TABLE bureau AS
  SELECT SK_ID_BUREAU, SK_ID_CURR, CREDIT_ACTIVE, DAYS_CREDIT, AMT_CREDIT_SUM, AMT_CREDIT_SUM_DEBT,
         AMT_CREDIT_SUM_OVERDUE, CREDIT_DAY_OVERDUE, CREDIT_TYPE
  FROM read_csv_auto('{UP}/bureau.csv') WHERE SK_ID_CURR IN (SELECT SK_ID_CURR FROM application)""")
con.execute(f"""CREATE OR REPLACE TABLE previous_application AS
  SELECT SK_ID_PREV, SK_ID_CURR, NAME_CONTRACT_STATUS, DAYS_DECISION, AMT_APPLICATION, AMT_CREDIT, CNT_PAYMENT
  FROM read_csv_auto('{UP}/previous_application.csv') WHERE SK_ID_CURR IN (SELECT SK_ID_CURR FROM application)""")
con.execute(f"""CREATE OR REPLACE TABLE installments AS
  SELECT SK_ID_PREV, SK_ID_CURR, NUM_INSTALMENT_NUMBER, DAYS_INSTALMENT, DAYS_ENTRY_PAYMENT, AMT_INSTALMENT, AMT_PAYMENT
  FROM read_csv_auto('{INST}') WHERE SK_ID_CURR IN (SELECT SK_ID_CURR FROM application)""")
for t in ("application", "bureau", "previous_application", "installments"):
    print(f"{t:22s} {con.execute(f'select count(*) from {t}').fetchone()[0]:>11,d} rows")
con.close()
