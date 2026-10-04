
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

warnings.filterwarnings("ignore")
RUN_EXPERIMENT = True

raw = pd.read_csv("bank-additional-full-comma.csv")      # all 41,188 rows
df = raw.copy()
yb = (df["y"] == "yes").astype(int)
n = len(df)

CAT = ["job", "marital", "education", "default", "housing", "loan",
       "contact", "month", "day_of_week", "poutcome"]
UNK = ["job", "marital", "education", "default", "housing", "loan"]
MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
NUM = [c for c in df.columns if c not in CAT + ["y"]]


def head(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


def show(label, value):
    print(f"{label:<58}{value}")


# ===================================================== 1. Task 1 (dataset)
head("1. TASK 1 - DATASET AND TARGET")
show("Records", f"{n:,}")
show("Input attributes (numeric / categorical)", f"{len(NUM)} / {len(CAT)}")
show("Empty cells", int(df.isna().sum().sum()))
vc = df["y"].value_counts()
show("Subscribed (yes)", f"{vc['yes']:,} ({vc['yes'] / n * 100:.1f}%)")
show("Did not subscribe (no)", f"{vc['no']:,} ({vc['no'] / n * 100:.1f}%)")
show("Imbalance ratio (no : yes)", f"{vc['no'] / vc['yes']:.1f} to 1")
show("Accuracy of always predicting no", f"{vc['no'] / n * 100:.1f}%")
med = df.groupby("y")["duration"].median()
show("Median call duration, yes vs no (seconds)", f"{med['yes']:.0f} vs {med['no']:.0f}")
has_unk = (df[UNK] == "unknown").any(axis=1)
show("Records with at least one unknown", f"{has_unk.sum():,} ({has_unk.mean() * 100:.1f}%)")
show("Subscription rate, any unknown vs none",
     f"{yb[has_unk].mean() * 100:.1f}% vs {yb[~has_unk].mean() * 100:.1f}%")
show("Default = unknown", f"{(df.default == 'unknown').mean() * 100:.1f}%")
show("Correlation emp.var.rate vs euribor3m", f"{df['emp.var.rate'].corr(df['euribor3m']):.2f}")

# ===================================================== 2. Task 3 (quality)
head("2. TASK 3 - DATA QUALITY")
print("Unknown values by attribute")
for c in UNK:
    k = int((df[c] == "unknown").sum())
    show(f"  {c}", f"{k:,} ({k / n * 100:.1f}%)")
rate = df.groupby("default")["y"].apply(lambda s: (s == "yes").mean() * 100)
show("Subscription rate by default (no / unknown / yes)",
     f"{rate['no']:.1f}% / {rate['unknown']:.1f}% / {rate['yes']:.1f}%")
show("Records with default = yes", int((df.default == "yes").sum()))

feat = [c for c in df.columns if c != "y"]
dups = int(df.duplicated().sum())
g = df[df.duplicated(subset=feat, keep=False)].groupby(feat)["y"].nunique()
show("Exact duplicates", f"{dups} ({dups / n * 100:.2f}%)")
show("Duplicate groups with conflicting labels", int((g > 1).sum()))

print("Outliers (1.5 x IQR rule)")
for c in ["duration", "campaign", "age"]:
    q1, q3 = df[c].quantile([0.25, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    k = int(((df[c] < lo) | (df[c] > hi)).sum())
    show(f"  {c}", f"{k:,} ({k / n * 100:.1f}%)")
show("Longest call (seconds)", int(df.duration.max()))
show("Most contacts of one client", int(df.campaign.max()))

print("Skewness of the raw columns")
for c in ["duration", "campaign", "previous", "pdays"]:
    show(f"  {c}", f"{df[c].skew():.2f}")
show("Distinct values of the economic indicators",
     ", ".join(f"{c} {df[c].nunique()}" for c in MACRO))

cm = df[NUM].corr()
show("Correlation emp.var.rate vs euribor3m", f"{cm.loc['emp.var.rate', 'euribor3m']:.2f}")
show("Correlation euribor3m vs nr.employed", f"{cm.loc['euribor3m', 'nr.employed']:.2f}")
show("Correlation emp.var.rate vs nr.employed", f"{cm.loc['emp.var.rate', 'nr.employed']:.2f}")
show("Correlation pdays vs previous", f"{cm.loc['pdays', 'previous']:.2f}")
vif = np.diag(np.linalg.inv(cm.values))
print("Variance inflation factors (highest three)")
for c, v in sorted(zip(NUM, vif), key=lambda t: -t[1])[:3]:
    show(f"  {c}", f"{v:.1f}")


def cramers_v(a, b):
    t = pd.crosstab(a, b)
    return float(np.sqrt(stats.chi2_contingency(t)[0] / t.values.sum() / (min(t.shape) - 1)))


print("Cramer's V with the target")
for c in ["poutcome", "month", "job", "loan", "housing"]:
    show(f"  {c}", f"{cramers_v(df[c], df['y']):.3f}")

print("Inconsistencies")
show("pdays = 999", f"{(df.pdays == 999).mean() * 100:.1f}%")
inc = int(((df.pdays == 999) & (df.previous > 0)).sum())
show("pdays = 999 but previous > 0", f"{inc:,} ({inc / n * 100:.1f}% of the file)")
show("Retired and younger than 45", int(((df.job == "retired") & (df.age < 45)).sum()))
show("Students older than 40", int(((df.job == "student") & (df.age > 40)).sum()))
show("Education = illiterate", int((df.education == "illiterate").sum()))
show("Aged under 18", int((df.age < 18).sum()))
show("Months in the file", df.month.nunique())
show("Share of contacts in May", f"{(df.month == 'may').mean() * 100:.1f}%")
short = df.duration < 5
show("Calls under 5 seconds (and how many subscribed)", f"{short.sum()} ({int(yb[short].sum())})")

# ===================================================== 3. Task 4 (transforms, PCA)
head("3. TASK 4 - TRANSFORMATIONS AND PCA")
print("Skewness after each transformation")
for c in ["duration", "campaign", "previous"]:
    yj = pd.Series(PowerTransformer("yeo-johnson").fit_transform(df[[c]]).ravel()).skew()
    show(f"  {c}: raw / log1p / Yeo-Johnson",
         f"{df[c].skew():.2f} / {np.log1p(df[c]).skew():.2f} / {yj:.2f}")

ev = PCA().fit(StandardScaler().fit_transform(df[MACRO])).explained_variance_ratio_
show("Economic block, variance in 1st component", f"{ev[0] * 100:.0f}%")
show("Economic block, variance in first 3 components", f"{ev[:3].sum() * 100:.0f}%")
nine = ["age", "campaign", "pdays", "previous"] + MACRO
ev2 = PCA().fit(StandardScaler().fit_transform(df[nine])).explained_variance_ratio_
show("Nine numeric inputs, components needed for 95%", int(np.searchsorted(np.cumsum(ev2), 0.95) + 1))
minority = int(vc["yes"])
show("Majority rows discarded by 1:1 under-sampling", f"{(vc['no'] - minority) / vc['no'] * 100:.0f}%")

# ===================================================== 4. Task 2 (clustering, noise)
head("4. TASK 2 AND TASK 3 - CLUSTERING AND NOISE")
d = df.copy()
d["cb"] = (d.pdays != 999).astype(int)
d["pdays"] = np.where(d.pdays == 999, 0, d.pdays)
d[["age", "campaign", "previous"]] = PowerTransformer().fit_transform(d[["age", "campaign", "previous"]])
nums = ["age", "campaign", "pdays", "previous", "cb"] + MACRO
X = np.hstack([StandardScaler().fit_transform(d[nums]),
               OneHotEncoder(sparse_output=False).fit_transform(d[CAT])])
show("Encoded features (duration and y excluded)", X.shape[1])

y = yb.values
sample = np.random.default_rng(0).choice(len(X), 6000, replace=False)
print("k-means: k, silhouette, NMI with y, NMI with month, NMI with poutcome")
for k in range(2, 9):
    lab = KMeans(k, n_init=5, random_state=0).fit_predict(X)
    print(f"  k={k}  {silhouette_score(X[sample], lab[sample]):.3f}  {nmi(y, lab):.3f}  "
          f"{nmi(df.month, lab):.3f}  {nmi(df.poutcome, lab):.3f}")

idx = NearestNeighbors(n_neighbors=6).fit(X[sample]).kneighbors(X[sample], return_distance=False)[:, 1:]
ys = y[sample]
all_major = (ys[idx] == 0).all(axis=1)
show("Subscribers whose 5 nearest neighbours are all 'no'", f"{all_major[ys == 1].mean() * 100:.1f}%")
oof = cross_val_predict(HistGradientBoostingClassifier(max_iter=150, random_state=0), X, y,
                        cv=StratifiedKFold(5, shuffle=True, random_state=42),
                        method="predict_proba")[:, 1]
low = int(((oof < 0.05) & (y == 1)).sum())
show("Subscribers with out-of-fold probability below 5%", f"{low} of {y.sum():,} ({low / y.sum() * 100:.1f}%)")

# ===================================================== 5. Task 4 (experiment)
if RUN_EXPERIMENT:
    head("5. TASK 4 - CROSS-VALIDATED EXPERIMENT (TABLE 6)")
    data = raw.drop_duplicates().reset_index(drop=True)      # the 12 duplicates are removed
    yy_all = (data.y == "yes").astype(int).values

    def prep(tr, te, cfg):
        tr, te = tr.copy(), te.copy()
        if cfg.get("impute") == "mode":
            for c in UNK:
                m = tr.loc[tr[c] != "unknown", c].mode()[0]
                tr.loc[tr[c] == "unknown", c] = m
                te.loc[te[c] == "unknown", c] = m
        num = ["age", "campaign", "previous", "pdays"] + MACRO
        if cfg.get("duration"):
            num = ["duration"] + num
        if cfg.get("tailored"):
            for part in (tr, te):
                part["contacted_before"] = (part.pdays != 999).astype(int)
                part["pdays"] = np.where(part.pdays == 999, 0, part.pdays)
            yj = ["age", "campaign", "previous"] + (["duration"] if cfg.get("duration") else [])
            pt = PowerTransformer("yeo-johnson")
            tr[yj] = pt.fit_transform(tr[yj])
            te[yj] = pt.transform(te[yj])
            num = num + ["contacted_before"]
        if cfg.get("macro") == "drop":
            num = [c for c in num if c not in ("emp.var.rate", "euribor3m")]
        sc = StandardScaler()
        xn_tr, xn_te = sc.fit_transform(tr[num]), sc.transform(te[num])
        if cfg.get("macro") == "pca":
            keep = [i for i, c in enumerate(num) if c not in MACRO]
            mi = [i for i, c in enumerate(num) if c in MACRO]
            p = PCA(n_components=2).fit(xn_tr[:, mi])
            xn_tr = np.hstack([xn_tr[:, keep], p.transform(xn_tr[:, mi])])
            xn_te = np.hstack([xn_te[:, keep], p.transform(xn_te[:, mi])])
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        xc_tr, xc_te = oh.fit_transform(tr[CAT]), oh.transform(te[CAT])
        return np.hstack([xn_tr, xc_tr]), np.hstack([xn_te, xc_te]), xn_tr.shape[1]

    def smote_nc(Xa, ya, nnum, k=5, seed=0):
        """SMOTE-style sampler written by hand. Numeric columns are interpolated
        and the one-hot columns take a vote among the neighbours."""
        rng = np.random.default_rng(seed)
        mi = np.where(ya == 1)[0]
        xm = Xa[mi]
        need = (ya == 0).sum() - len(mi)
        nb_idx = NearestNeighbors(n_neighbors=k + 1).fit(xm).kneighbors(xm, return_distance=False)[:, 1:]
        base = rng.integers(0, len(xm), need)
        nb = nb_idx[base, rng.integers(0, k, need)]
        syn = np.empty((need, Xa.shape[1]))
        u = rng.random((need, 1))
        syn[:, :nnum] = xm[base, :nnum] + u * (xm[nb, :nnum] - xm[base, :nnum])
        syn[:, nnum:] = (xm[nb_idx[base]][:, :, nnum:].mean(axis=1) >= 0.5).astype(float)
        return np.vstack([Xa, syn]), np.concatenate([ya, np.ones(need, int)])

    def balance(Xa, ya, mode, nnum, seed=0):
        rng = np.random.default_rng(seed)
        if mode == "ros":
            mi = np.where(ya == 1)[0]
            extra = rng.choice(mi, (ya == 0).sum() - len(mi))
            return np.vstack([Xa, Xa[extra]]), np.concatenate([ya, ya[extra]])
        if mode == "rus":
            mi = np.where(ya == 1)[0]
            ma = rng.choice(np.where(ya == 0)[0], len(mi), replace=False)
            s = np.concatenate([mi, ma])
            return Xa[s], ya[s]
        return smote_nc(Xa, ya, nnum, seed=seed)

    def make_model(name, weighted):
        cw = "balanced" if weighted else None
        if name == "LR":
            return LogisticRegression(max_iter=2000, class_weight=cw)
        return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                              random_state=0, class_weight=cw)

    # C0 to C6 are the seven rows of Table 6. X1 to X3 are extra checks that are
    # not shown in the table but support statements in the text (Table 5 and Data Balancing).
    CONFIGS = {
        "C0 Baseline": {},
        "C1 Mode imputation": dict(impute="mode"),
        "C2 Tailored transforms": dict(tailored=True),
        "C3 PCA on economic block": dict(tailored=True, macro="pca"),
        "C4 Class weights": dict(tailored=True, bal="cw"),
        "C5 SMOTE (NC variant)": dict(tailored=True, bal="smote"),
        "C6 Duration included (leakage)": dict(tailored=True, duration=True),
        "X1 Redundant indicators dropped (extra)": dict(tailored=True, macro="drop"),
        "X2 Random over-sampling (extra)": dict(tailored=True, bal="ros"),
        "X3 Random under-sampling (extra)": dict(tailored=True, bal="rus"),
    }
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    rows, t0 = [], time.time()
    for cname, cfg in CONFIGS.items():
        for mname in ["LR", "HGB"]:
            res = []
            for f, (a, b) in enumerate(skf.split(data, yy_all)):
                xtr, xte, nnum = prep(data.iloc[a], data.iloc[b], cfg)
                ytr, yte = yy_all[a], yy_all[b]
                bal = cfg.get("bal")
                if bal in ("ros", "rus", "smote"):
                    xtr, ytr = balance(xtr, ytr, bal, nnum, seed=f)
                p = make_model(mname, bal == "cw").fit(xtr, ytr).predict_proba(xte)[:, 1]
                pr, rc, f1, _ = precision_recall_fscore_support(
                    yte, (p >= 0.5).astype(int), average="binary", zero_division=0)
                top = np.argsort(-p)[:int(0.1 * len(p))]
                res.append([roc_auc_score(yte, p), average_precision_score(yte, p),
                            pr, rc, f1, yte[top].sum() / yte.sum()])
            r = np.array(res)
            rows.append(dict(config=cname, model=mname, roc_auc=r[:, 0].mean(), roc_sd=r[:, 0].std(),
                             pr_auc=r[:, 1].mean(), pr_sd=r[:, 1].std(), precision=r[:, 2].mean(),
                             recall=r[:, 3].mean(), f1=r[:, 4].mean(), top10_capture=r[:, 5].mean()))
            print(f"  {cname} / {mname} done ({time.time() - t0:.0f}s)", flush=True)
    table6 = pd.DataFrame(rows)
    table6.to_csv("table6_results.csv", index=False)
    print("\nTable 6 (also saved as table6_results.csv)")
    print(table6.round(3).to_string(index=False))

    # Lowering the threshold of the unweighted models until recall reaches 0.63
    print("\nUnweighted models with the threshold lowered to reach recall 0.63")
    cfg = dict(tailored=True)
    for mname in ["LR", "HGB"]:
        out = []
        for a, b in skf.split(data, yy_all):
            xtr, xte, _ = prep(data.iloc[a], data.iloc[b], cfg)
            p = make_model(mname, False).fit(xtr, yy_all[a]).predict_proba(xte)[:, 1]
            pos = yy_all[b]
            order = np.argsort(-p)
            k = np.searchsorted(np.cumsum(pos[order]) / pos.sum(), 0.63)
            pred = (p >= p[order][k]).astype(int)
            pr, rc, f1, _ = precision_recall_fscore_support(pos, pred, average="binary")
            out.append([pr, rc, f1])
        pr, rc, f1 = np.mean(out, axis=0)
        show(f"  {mname} precision / recall / F1", f"{pr:.3f} / {rc:.3f} / {f1:.3f}")

print("\nDone.")
