"""
Stage 3.3: choose the features that go into the scorecard.

  1. drop features with IV below MIN_IV            (they barely separate good from bad)
  2. drop features that overlap with a stronger one (correlation of WOE values >= MAX_CORR)
  3. VIF check on what is left                     (a stricter overlap test)
  4. direction check                               (does "higher value" always mean the same thing?)

All of this uses the TRAINING set only.

Run:  python scorecard/03_select_features.py      (from the credit_risk folder)
"""
import json, os, sys
import numpy as np, pandas as pd, duckdb

sys.path.insert(0, os.path.dirname(__file__))
from woe_binning import transform_woe

MIN_IV = 0.02
MAX_CORR = 0.70
OUT = "scorecard/output"

con = duckdb.connect("credit_risk.duckdb", read_only=True)
train = con.execute("SELECT * FROM train_set").df()
spec = json.load(open(f"{OUT}/bins.json"))
rank = pd.read_csv(f"{OUT}/iv_ranking.csv")          # already sorted by IV, strongest first
iv = dict(zip(rank.feature, rank.iv))
W = transform_woe(train, spec)                       # every feature replaced by the WOE of its bin


def family(f):
    if f.startswith("EXT_SOURCE"): return "Outside score"
    if f.startswith("inst_"):    return "Payment behaviour"
    if f.startswith("bureau_"):  return "Bureau"
    if f.startswith("prev_"):    return "Earlier applications"
    return "Loan and applicant"


def direction(f):
    fs = spec[f]
    if fs["type"] != "numeric":
        return "categories"
    w = [wv for b, wv in zip(fs["bins"], fs["woe"]) if b != "Missing"]
    if len(w) < 2:
        return "n/a"
    return "higher value = riskier" if w[-1] < w[0] else "higher value = safer"


def vif(df):
    """VIF of each column = 1 / (1 - R^2) when that column is predicted from all the others."""
    X = df.values.astype(float)
    out = {}
    for i, c in enumerate(df.columns):
        y, A = X[:, i], np.column_stack([np.ones(len(X)), np.delete(X, i, axis=1)])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[c] = 1 / (1 - r2)
    return pd.Series(out)


# ---------------------------------------------------------------- move 1: low IV
decision = {}
for f in rank.feature:
    if iv[f] < MIN_IV:
        decision[f] = ("dropped", f"IV {iv[f]:.3f} is below {MIN_IV}: barely separates good from bad")
cands = [f for f in rank.feature if f not in decision]
corr = W[cands].corr()

# ---------------------------------------------------------------- move 2: overlap (strongest first)
kept = []
for f in cands:
    clash = [(k, corr.loc[f, k]) for k in kept if abs(corr.loc[f, k]) >= MAX_CORR]
    if clash:
        k, r = max(clash, key=lambda t: abs(t[1]))
        decision[f] = ("dropped", f"overlaps with {k} (correlation {r:.2f}); {k} has the higher IV")
    else:
        kept.append(f)
        decision[f] = ("kept", "")

# ---------------------------------------------------------------- moves 3 and 4: checks on the shortlist
V = vif(W[kept])
table = pd.DataFrame({"feature": rank.feature, "iv": rank.iv,
                      "decision": [decision[f][0] for f in rank.feature],
                      "reason": [decision[f][1] for f in rank.feature]})
table.to_csv(f"{OUT}/feature_decisions.csv", index=False)
corr.to_csv(f"{OUT}/correlation_matrix.csv")
selected = [{"feature": f, "iv": iv[f], "family": family(f), "direction": direction(f), "vif": float(V[f])} for f in kept]
json.dump(selected, open(f"{OUT}/selected_features.json", "w"), indent=1)

n_low = sum(1 for f in decision if decision[f][0] == "dropped" and decision[f][1].startswith("IV"))
n_dup = sum(1 for f in decision if decision[f][0] == "dropped" and decision[f][1].startswith("overlaps"))
print(f"{len(rank)} features -> dropped for low IV: {n_low} | dropped for overlap: {n_dup} | KEPT: {len(kept)}")
print(f"lowest correlation anywhere in the candidate matrix: {corr.values.min():.3f}")

print("\nDROPPED FOR OVERLAP")
print(table[table.reason.str.startswith("overlaps")].round(3).to_string(index=False))

print("\nKEPT")
print(pd.DataFrame(selected).round(3).to_string(index=False))
kc = W[kept].corr().abs().values.copy()
np.fill_diagonal(kc, 0)
print(f"\nlargest correlation among kept features: {kc.max():.2f} | largest VIF: {V.max():.2f}")

# ---------------------------------------------------------------- the chart
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]       # one hue, light -> dark
cmap = LinearSegmentedColormap.from_list("blue_seq", STEPS)
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"

fams = {}
for f in cands:
    fams.setdefault(family(f), []).append(f)
fam_order = sorted(fams, key=lambda k: -max(iv[f] for f in fams[k]))
order = [f for k in fam_order for f in fams[k]]
C = corr.loc[order, order].values
n = len(order)

fig = plt.figure(figsize=(10, 9.4), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax = fig.add_axes([0.22, 0.10, 0.58, 0.64])
ax.set_facecolor(SURFACE)
mesh = ax.pcolormesh(np.clip(C, 0, 1), cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=1.5)
ax.set_xlim(0, n); ax.set_ylim(n, 0)
ax.xaxis.tick_top()
ax.set_xticks(np.arange(n) + 0.5); ax.set_yticks(np.arange(n) + 0.5)
ax.set_xticklabels(order, rotation=60, ha="left", fontsize=8, color=INK, rotation_mode="anchor")
ax.set_yticklabels(order, fontsize=8, color=INK)
for lab, f in zip(ax.get_xticklabels(), order): lab.set_fontweight("bold" if f in kept else "normal")
for lab, f in zip(ax.get_yticklabels(), order): lab.set_fontweight("bold" if f in kept else "normal")
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)

# numbers only on the pairs that triggered a drop
for i in range(n):
    for j in range(n):
        v = C[i, j]
        if i != j and abs(v) >= MAX_CORR:
            r_, g_, b_ = cmap(min(max(v, 0), 1))[:3]
            lum = 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if lum < 0.45 else INK)

# family separators and labels
pos = 0
for k in fam_order:
    size = len(fams[k])
    if pos > 0:
        ax.axhline(pos, color=SURFACE, linewidth=4); ax.axvline(pos, color=SURFACE, linewidth=4)
    ax.text(n + 0.4, pos + size / 2, k, va="center", ha="left", fontsize=8.5, color=INK2, clip_on=False)
    pos += size

fig.text(0.02, 0.972, "Features in the same family carry overlapping information", fontsize=13, fontweight="bold", color=INK)
fig.text(0.02, 0.948, f"Correlation between WOE values, training set. Bold names were kept; numbers mark pairs at {MAX_CORR:.2f} or above.",
         fontsize=9, color=INK2)
cax = fig.add_axes([0.22, 0.045, 0.30, 0.014])
cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[0, 0.35, 0.70, 1.0])
cb.outline.set_visible(False)
cb.ax.tick_params(length=0, labelsize=8, labelcolor=INK2)
cb.set_label("Correlation (0 = unrelated, 1 = identical)", fontsize=8, color=INK2)
fig.savefig(f"{OUT}/correlation_heatmap.png", facecolor=SURFACE)
print(f"\nchart saved: {OUT}/correlation_heatmap.png")
