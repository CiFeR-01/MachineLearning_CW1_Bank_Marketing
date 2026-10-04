import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
 
df = pd.read_csv("bank-additional-full-comma.csv").drop_duplicates()
y = (df.pop("y") == "yes").astype(int)
df = df.drop(columns="duration")                         # leakage
df["contacted_before"] = (df.pdays != 999).astype(int)   # explicit 999 code
df["pdays"] = df.pdays.where(df.pdays != 999, 0)
df["default_unknown"] = (df.default == "unknown").astype(int)
 
cat = df.select_dtypes(exclude="number").columns.tolist()  # "unknown" stays a level
skewed = ["age", "campaign", "previous"]
macro = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
other = ["pdays", "contacted_before", "default_unknown"]
onehot = OneHotEncoder(handle_unknown="ignore")
 
linear_prep = ColumnTransformer([
    ("power", Pipeline([("yj", PowerTransformer("yeo-johnson"))]), skewed),
    ("macro", Pipeline([("z", StandardScaler()), ("pca", PCA(3))]), macro),
    ("flags", StandardScaler(), other),
    ("cat", onehot, cat)])
tree_prep = ColumnTransformer([("num", "passthrough", skewed + macro + other),
                               ("cat", onehot, cat)])
 
lr = LogisticRegression(class_weight="balanced", max_iter=2000)
gb = HistGradientBoostingClassifier(class_weight="balanced", random_state=0)
models = {"Logistic regression": Pipeline([("prep", linear_prep), ("clf", lr)]),
          "Gradient boosting": Pipeline([("prep", tree_prep), ("clf", gb)])}
# All transformers sit inside the pipelines, so they are fitted on training folds only.
# For a time-based test, hold out the last 20% of rows (file is date ordered).
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for name, model in models.items():
    s = cross_validate(model, df, y, cv=cv, scoring=["roc_auc", "average_precision"])
    print(name, s["test_roc_auc"].mean().round(3))
    print(name, s["test_average_precision"].mean().round(3))
 
