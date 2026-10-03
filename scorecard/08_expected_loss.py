"""
Stage 5 (real data): 12-month expected loss = PD x LGD x EAD for the hold-out portfolio, with sensitivity and stress tests.

PORTFOLIO  the 92,254 hold-out applicants, seen AT ORIGINATION (the real data has no dates, so every loan is treated as just booked).
           Their outcomes (TARGET) are not used for the expected loss; they are only used to check the PDs.
PD         scorecard PD = chance the loan EVER shows payment difficulties (lifetime), converted to a 12-month PD with a
           constant-hazard assumption: 1 - (1 - PD_life) ^ (months ahead / term). The logistic model is calibrated on the
           training set by construction, so no extra recalibration shift is applied; the hold-out check below confirms it.
LGD        ASSUMPTION: central by contract type (+5pp for the largest quarter of loans); low / high = central -/+ 15pp
EAD        cash loans: the full credit amount at origination; revolving: limit x (drawn share + undrawn share x CCF)
TERM       cash loans: months to repay credit with the given annuity at an assumed ANNUAL_RATE (capped at 120)
STRESS     score shift (-20 points = bad odds double), LGD add-on, higher CCF

Run:  python scorecard/08_expected_loss.py     (from the project folder)
"""
import json
import numpy as np, pandas as pd, duckdb

OUT = "scorecard/output"
ANNUAL_RATE = 0.22
REV_HORIZON = 24
LGD_CENTRAL = {"Cash loans": 0.65, "Revolving loans": 0.75}
LGD_BAND, BIG_LOAN_ADD = 0.15, 0.05
REV_DRAWN, CCF = 0.50, {"low": 0.60, "central": 0.75, "high": 0.90}
SCENARIOS = {"Base": (0, 0.00, CCF["central"]), "Moderate": (-10, 0.05, 0.85), "Severe": (-20, 0.10, 1.00)}
PDO = json.load(open(f"{OUT}/scaling.json"))["pdo"]

con = duckdb.connect("credit_risk.duckdb", read_only=True)
app = con.execute("SELECT SK_ID_CURR, NAME_CONTRACT_TYPE, AMT_CREDIT, AMT_ANNUITY FROM test_set").df()
sc = pd.read_csv(f"{OUT}/scores.csv")
n_test = len(app)
df = app.merge(sc[sc.dataset == "test"][["SK_ID_CURR", "TARGET", "score", "pd"]], on="SK_ID_CURR", how="inner").rename(columns={"pd": "pd_life"})
assert len(df) == n_test, "every hold-out loan must have exactly one score"
print(f"HOLD-OUT portfolio: {len(df):,} loans | average lifetime PD {100*df.pd_life.mean():.2f}% vs actual default rate {100*df.TARGET.mean():.2f}%")

is_rev = (df.NAME_CONTRACT_TYPE == "Revolving loans").values
i_m = ANNUAL_RATE / 12
d = (df.AMT_CREDIT / df.AMT_ANNUITY).replace([np.inf, -np.inf], np.nan)
d = d.fillna(d.median()).values                                  # 12 loans have no annuity: use the median
term = np.clip(-np.log(1 - np.minimum(d * i_m, 0.95)) / np.log(1 + i_m), 3, 120)
df["term_months"] = np.where(is_rev, REV_HORIZON, term)
big = df.AMT_CREDIT >= df.AMT_CREDIT.quantile(0.75)
df["lgd_central"] = np.minimum(df.NAME_CONTRACT_TYPE.map(LGD_CENTRAL) + BIG_LOAN_ADD * big, 1.0)
df["lgd_low"] = np.maximum(df.lgd_central - LGD_BAND, 0.0)
df["lgd_high"] = np.minimum(df.lgd_central + LGD_BAND, 1.0)


def ead_for(ccf):
    return np.where(is_rev, df.AMT_CREDIT.values * (REV_DRAWN + (1 - REV_DRAWN) * ccf), df.AMT_CREDIT.values)


def to_12m(pd_life):
    return 1 - (1 - np.clip(pd_life, 0, 0.999999)) ** (np.minimum(12.0, df.term_months.values) / df.term_months.values)


def shifted(pd_life, shift):
    odds = pd_life / (1 - pd_life) * 2 ** (-shift / PDO)
    return odds / (1 + odds)


df["pd_12m"] = to_12m(df.pd_life.values)
df["ead"] = ead_for(CCF["central"])
df["lgd"] = df.lgd_central
df["expected_loss"] = df.pd_12m * df.lgd * df.ead
s_shift, s_lgd, s_ccf = SCENARIOS["Severe"]
df["expected_loss_severe"] = to_12m(shifted(df.pd_life.values, s_shift)) * np.minimum(df.lgd + s_lgd, 1) * ead_for(s_ccf)
bins = [-np.inf, 500, 520, 540, 560, 580, 600, np.inf]
labs = ["under 500", "500-519", "520-539", "540-559", "560-579", "580-599", "600 and over"]
df["band"] = pd.cut(df.score, bins=bins, labels=labs, right=False)
df["pd"] = df.pd_12m
df[["SK_ID_CURR", "NAME_CONTRACT_TYPE", "score", "band", "pd", "pd_life", "lgd", "ead", "expected_loss", "expected_loss_severe", "TARGET"]].to_csv(
    f"{OUT}/expected_loss_loans.csv", index=False)

granted = df.AMT_CREDIT.sum()
print(f"credit at origination {granted/1e6:,.0f}m | EAD {df.ead.sum()/1e6:,.0f}m ({100*df.ead.sum()/granted:.0f}% of credit; revolving limits are only partly drawn)")
print(f"term of cash loans (months): median {df.term_months[~is_rev].median():.1f}, 5th-95th pct {np.percentile(df.term_months[~is_rev],5):.1f}-{np.percentile(df.term_months[~is_rev],95):.1f}")
print(f"average PD: lifetime {100*df.pd_life.mean():.2f}% -> 12-month {100*df.pd_12m.mean():.2f}%")
print(df.groupby("NAME_CONTRACT_TYPE").agg(loans=("pd", "size"), pd12_pct=("pd_12m", lambda s: 100 * s.mean()), lgd=("lgd", "mean"),
                                           ead_m=("ead", lambda s: s.sum() / 1e6), el_m=("expected_loss", lambda s: s.sum() / 1e6)).round(2).to_string())

b = df.groupby("band", observed=True).agg(loans=("pd", "size"), avg_pd_pct=("pd_12m", lambda s: 100 * s.mean()),
                                          ead_m=("ead", lambda s: s.sum() / 1e6), expected_loss_m=("expected_loss", lambda s: s.sum() / 1e6),
                                          actual_defaults=("TARGET", "sum"))
b["share_of_EAD_pct"] = 100 * b.ead_m / b.ead_m.sum()
b["share_of_EL_pct"] = 100 * b.expected_loss_m / b.expected_loss_m.sum()
b["EL_rate_pct"] = 100 * b.expected_loss_m / b.ead_m
b.reset_index().to_csv(f"{OUT}/el_by_band.csv", index=False)
print("\n12-MONTH EXPECTED LOSS BY SCORE BAND (base case)")
print(b.round(2).to_string())
tot_el, tot_ead = df.expected_loss.sum(), df.ead.sum()
print(f"\nTOTAL 12-month expected loss {tot_el/1e6:,.1f}m on EAD {tot_ead/1e6:,.0f}m -> {100*tot_el/tot_ead:.2f}%")

cc = pd.DataFrame({"measure": ["average PD from the scorecard", "after recalibration", "actual default rate"],
                   "test_pct": [100 * df.pd_life.mean(), 100 * df.pd_life.mean(), 100 * df.TARGET.mean()],
                   "expected_defaults": [df.pd_life.sum(), df.pd_life.sum(), float(df.TARGET.sum())]})
cc.to_csv(f"{OUT}/calibration_check.csv", index=False)
print("\nCALIBRATION CHECK (lifetime PD, hold-out): expected defaults", round(df.pd_life.sum()), "vs actual", int(df.TARGET.sum()))

rows = []
for lg in ("low", "central", "high"):
    for cc_ in ("low", "central", "high"):
        rows.append({"lgd_case": lg, "ccf_case": cc_, "expected_loss_m": (df.pd_12m * df[f"lgd_{lg}"] * ead_for(CCF[cc_])).sum() / 1e6})
sens = pd.DataFrame(rows)
sens["vs_central_pct"] = 100 * (sens.expected_loss_m / sens[(sens.lgd_case == "central") & (sens.ccf_case == "central")].expected_loss_m.iloc[0] - 1)
sens.to_csv(f"{OUT}/lgd_sensitivity.csv", index=False)
print("\nSENSITIVITY (m)")
print(sens.pivot(index="lgd_case", columns="ccf_case", values="expected_loss_m").reindex(["low", "central", "high"])[["low", "central", "high"]].round(1).to_string())

rows = []
for name, (shift, lgd_add, ccf) in SCENARIOS.items():
    p = to_12m(shifted(df.pd_life.values, shift)); lgd = np.minimum(df.lgd.values + lgd_add, 1.0); ead = ead_for(ccf); el = p * lgd * ead
    rows.append({"scenario": name, "score_shift": shift, "lgd_add_on_pp": 100 * lgd_add, "revolving_ccf": ccf, "avg_pd_pct": 100 * p.mean(),
                 "total_ead_m": ead.sum() / 1e6, "expected_loss_m": el.sum() / 1e6, "EL_rate_pct": 100 * el.sum() / ead.sum()})
st = pd.DataFrame(rows)
st["EL_vs_base_pct"] = 100 * (st.expected_loss_m / st.expected_loss_m.iloc[0] - 1)
st.to_csv(f"{OUT}/stress_scenarios.csv", index=False)
print("\nSTRESS TEST"); print(st.round(2).to_string(index=False))
sb = df.groupby("band", observed=True)[["expected_loss", "expected_loss_severe"]].sum() / 1e6
sb["extra_loss_m"] = sb.expected_loss_severe - sb.expected_loss
sb["share_of_extra_pct"] = 100 * sb.extra_loss_m / sb.extra_loss_m.sum()
print("\nWHERE THE EXTRA SEVERE-CASE LOSS LANDS (m)"); print(sb.round(2).to_string())
json.dump({"view": "origination", "annual_rate": ANNUAL_RATE, "lgd_central": LGD_CENTRAL, "lgd_band_pp": LGD_BAND, "big_loan_add": BIG_LOAN_ADD,
           "revolving_drawn_share": REV_DRAWN, "ccf": CCF, "revolving_horizon_months": REV_HORIZON, "recalibration_logit_shift": 0.0,
           "scenarios": SCENARIOS}, open(f"{OUT}/el_assumptions.json", "w"), indent=1)
