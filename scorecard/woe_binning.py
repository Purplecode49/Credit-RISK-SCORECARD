"""
WOE binning toolkit (Stage 3.2).

fit_binning(train_df, features, target)  -> a "spec": the bins and WOE of every feature, learned from TRAIN only
transform_woe(df, spec)                  -> replaces every feature by the WOE of the bin it falls in
assign_bins(series, feature_spec)        -> which bin each value falls in (as readable labels)

Binning rules (what "sensible bins" means here):
  * numbers are cut into at most MAX_BINS groups, each holding at least 5% of the training people
  * WOE must move in ONE direction as the feature goes up (higher -> always riskier, or always safer)
  * empty values (NULL) get their own "Missing" bin
  * text categories get one bin each; very small categories are pooled into "Other"
"""
import numpy as np
import pandas as pd

MAX_BINS = 8
MIN_FRAC_NUMERIC = 0.05
MIN_FRAC_CATEGORICAL = 0.02
SMOOTH = 0.5          # tiny add-on so a bin with 0 defaults does not give log(0)


# ------------------------------------------------------------------ basic maths
def _woe(goods, bads, G, B):
    """WOE = ln( share of all goods in the bin / share of all bads in the bin )"""
    pg = (goods + SMOOTH) / G
    pb = (bads + SMOOTH) / B
    return np.log(pg / pb), pg, pb


def _counts(idx, y, k):
    n = np.bincount(idx, minlength=k).astype(float)
    bads = np.bincount(idx, weights=y, minlength=k)
    return n, n - bads, bads


# ------------------------------------------------------------------ numeric features
def _fit_numeric(x, y, G, B, N):
    miss = x.isna().values
    xv = x[~miss].values.astype(float)
    yv = y[~miss]
    uniq = np.unique(xv)
    if len(uniq) <= 12:                                   # few distinct values: start with one bin per value
        edges = list(uniq[1:])
    else:                                                 # otherwise start with 20 equal-sized (5%) slices
        edges = list(np.unique(np.quantile(xv, np.linspace(0, 1, 21)[1:-1])))
    min_n = MIN_FRAC_NUMERIC * N

    while True:
        k = len(edges) + 1
        idx = np.searchsorted(edges, xv, side="right")
        n, g, b = _counts(idx, yv, k)
        w, _, _ = _woe(g, b, G, B)
        if k == 1:
            break
        # 1) a bin with too few people: merge it into the neighbour with the closest WOE
        small = np.where(n < min_n)[0]
        if len(small):
            i = small[np.argmin(n[small])]
            if i == 0:
                j = 0
            elif i == k - 1:
                j = k - 2
            else:
                j = i - 1 if abs(w[i] - w[i - 1]) <= abs(w[i] - w[i + 1]) else i
            edges.pop(j)
            continue
        # 2) WOE must move in one direction: merge the first pair that breaks the trend
        if k >= 3:
            ix = np.arange(k)
            slope = np.sum(n * (ix - np.average(ix, weights=n)) * (w - np.average(w, weights=n)))
            direction = 1.0 if slope >= 0 else -1.0
            bad_pairs = np.where(direction * np.diff(w) < -1e-12)[0]
            if len(bad_pairs):
                edges.pop(bad_pairs[0])
                continue
        # 3) too many bins: merge the two neighbours whose WOE is closest
        if k > MAX_BINS:
            edges.pop(int(np.argmin(np.abs(np.diff(w)))))
            continue
        break

    int_like = len(uniq) <= 60 and np.allclose(uniq, np.round(uniq))
    return {"type": "numeric", "edges": [float(e) for e in edges], "int_like": bool(int_like)}


def _fmt(v):
    return f"{v:,.4g}"


def _numeric_labels(spec):
    e, k = spec["edges"], len(spec["edges"]) + 1
    if k == 1:
        return ["all values"]
    labels = []
    for i in range(k):
        if spec["int_like"]:
            lo, hi = (None if i == 0 else int(e[i - 1])), (None if i == k - 1 else int(e[i]) - 1)
            if lo is None:
                labels.append(f"<= {hi}")
            elif hi is None:
                labels.append(f">= {lo}")
            else:
                labels.append(f"{lo}" if lo == hi else f"{lo} to {hi}")
        else:
            if i == 0:
                labels.append(f"< {_fmt(e[0])}")
            elif i == k - 1:
                labels.append(f">= {_fmt(e[-1])}")
            else:
                labels.append(f"{_fmt(e[i - 1])} to < {_fmt(e[i])}")
    return labels


# ------------------------------------------------------------------ categorical features
def _fit_categorical(x, y, G, B, N):
    s = pd.Series(y, index=x.index)
    counts = x.value_counts()
    keep = [c for c, v in counts.items() if v >= MIN_FRAC_CATEGORICAL * N]
    return {"type": "categorical", "categories": [str(c) for c in keep], "has_other": len(keep) < len(counts)}


# ------------------------------------------------------------------ assign bins / stats
def _bin_index(x, fs):
    """integer bin index per row; -1 = Missing"""
    if fs["type"] == "numeric":
        v = pd.to_numeric(x, errors="coerce").astype("float64").values
        idx = np.searchsorted(fs["edges"], v, side="right")
        return np.where(np.isnan(v), -1, idx)
    cats = {c: i for i, c in enumerate(fs["categories"])}
    other = len(fs["categories"])
    return np.array([-1 if pd.isna(v) else cats.get(str(v), other) for v in x])


def _bin_labels(fs):
    if fs["type"] == "numeric":
        return _numeric_labels(fs)
    return fs["categories"] + (["Other"] if fs["has_other"] else [])


def _stats(fs, x, y, G, B):
    idx = _bin_index(x, fs)
    labels = _bin_labels(fs)
    rows = []
    order = list(range(len(labels))) + ([-1] if (idx == -1).any() else [])
    for i in order:
        m = idx == i
        n, bads = int(m.sum()), float(y[m].sum())
        w, pg, pb = _woe(n - bads, bads, G, B)
        rows.append({"bin": "Missing" if i == -1 else labels[i], "bin_index": i, "people": n,
                     "goods": int(n - bads), "bads": int(bads), "bad_rate_pct": 100 * bads / n if n else np.nan,
                     "pct_goods": 100 * pg, "pct_bads": 100 * pb, "woe": float(w), "iv_part": float((pg - pb) * w)})
    return pd.DataFrame(rows)


def fit_binning(df, features, target="TARGET"):
    y = df[target].values.astype(float)
    G, B, N = (y == 0).sum(), (y == 1).sum(), len(y)
    spec = {}
    for f in features:
        x = df[f]
        is_text = not pd.api.types.is_numeric_dtype(x)
        fs = _fit_categorical(x, y, G, B, N) if is_text else _fit_numeric(x, y, G, B, N)
        st = _stats(fs, x, y, G, B)
        fs["bins"] = st["bin"].tolist()
        fs["woe"] = st["woe"].tolist()
        fs["bin_index"] = st["bin_index"].tolist()
        fs["iv"] = float(st["iv_part"].sum())
        spec[f] = fs
    return spec


def bin_table(df, feature, spec, target="TARGET"):
    y = df[target].values.astype(float)
    return _stats(spec[feature], df[feature], y, (y == 0).sum(), (y == 1).sum())


# ------------------------------------------------------------------ apply to any data (train, test, future)
def transform_woe(df, spec):
    out = {}
    for f, fs in spec.items():
        idx = _bin_index(df[f], fs)
        lookup = dict(zip(fs["bin_index"], fs["woe"]))
        out[f] = [lookup.get(int(i), 0.0) for i in idx]     # a bin never seen in training gets a neutral WOE of 0
    return pd.DataFrame(out, index=df.index)


def assign_bins(series, fs):
    idx = _bin_index(series, fs)
    labels = _bin_labels(fs)
    return pd.Series(["Missing" if i == -1 else labels[i] for i in idx], index=series.index)
