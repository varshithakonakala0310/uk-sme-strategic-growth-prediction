"""
Dissertation recreation — synthetic data only.
Original dissertation: 248 UK SMEs, Jan 2018-Feb 2023, Kaggle secondary dataset.
This script generates a SYNTHETIC dataset shaped like that study (248 SMEs,
14 financial indicators) and recreates the methodology: descriptive analytics,
correlation, regression, Random Forest, Gradient Boosting, XGBoost comparison.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE,"data"), exist_ok=True)
os.makedirs(os.path.join(BASE,"results"), exist_ok=True)

rng = np.random.default_rng(42)
n = 248
employees = rng.integers(5, 250, n)
avg_turnover = rng.lognormal(mean=13.5, sigma=0.7, size=n)  # GBP
profit_margin = rng.normal(0.12, 0.08, n).clip(-0.05, 0.35)
avg_profit = avg_turnover * profit_margin
tax_rate = rng.normal(0.19, 0.03, n).clip(0.05, 0.30)
revenue_per_employee = avg_turnover / employees
profit_per_employee = avg_profit / employees
# latent growth driven mainly by profit_per_employee and profit_margin (mirrors dissertation narrative, synthetic)
latent = 0.55*(profit_per_employee/ profit_per_employee.max()) + 0.45*(profit_margin/0.35) + rng.normal(0,0.12,n)
turnover_growth = (latent*0.30 - 0.05).clip(-0.20, 0.60)
turnover_volatility = rng.uniform(0.05, 0.40, n)
cost_ratio = (1-profit_margin).clip(0.5,1.05)
liquidity_ratio = rng.normal(1.4,0.5,n).clip(0.3,3.0)
debt_to_equity = rng.normal(0.8,0.4,n).clip(0.05,2.5)
operating_efficiency = rng.normal(0.65,0.15,n).clip(0.2,0.95)
investment_intensity = rng.uniform(0.02,0.25,n)
export_share = rng.uniform(0,0.6,n)

df = pd.DataFrame({
    "company_id": [f"SME_{i+1:03d}" for i in range(n)],
    "employees": employees,
    "avg_turnover_gbp": avg_turnover.round(2),
    "avg_profit_gbp": avg_profit.round(2),
    "profit_margin": profit_margin.round(4),
    "tax_rate": tax_rate.round(4),
    "revenue_per_employee_gbp": revenue_per_employee.round(2),
    "profit_per_employee_gbp": profit_per_employee.round(2),
    "turnover_growth": turnover_growth.round(4),
    "turnover_volatility": turnover_volatility.round(4),
    "cost_ratio": cost_ratio.round(4),
    "liquidity_ratio": liquidity_ratio.round(4),
    "debt_to_equity": debt_to_equity.round(4),
    "operating_efficiency": operating_efficiency.round(4),
    "investment_intensity": investment_intensity.round(4),
    "export_share": export_share.round(4),
})
# 14 indicators = employees + 13 financial columns above (excluding id/target)
median_growth = df["turnover_growth"].median()
df["high_growth"] = (df["turnover_growth"] >= median_growth).astype(int)
df.to_csv(os.path.join(BASE,"data","synthetic_sme_data.csv"), index=False)

features = [c for c in df.columns if c not in ("company_id","high_growth","turnover_growth")]
# 14 indicators: employees + 13 financial indicators; turnover_growth is the
# continuous source of the target and is excluded from classification features
# to avoid target leakage.
print(f"Features ({len(features)}): {features}")
desc = df[features].describe().T
corr = df[features+["turnover_growth","high_growth"]].corr(numeric_only=True)

X = df[features]; y = df["high_growth"]
Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
scaler = StandardScaler(); Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)

# Regression on continuous growth
lin = LinearRegression().fit(Xtr_s, df.loc[Xtr.index,"turnover_growth"])
r2 = r2_score(df.loc[Xte.index,"turnover_growth"], lin.predict(Xte_s))

try:
    from xgboost import XGBClassifier
    xgb_model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42)
    xgb_label = "XGBoost"
except Exception as e:
    xgb_model = HistGradientBoostingClassifier(random_state=42)
    xgb_label = "HistGradientBoosting (gradient boosting variant — XGBoost not available)"

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    xgb_label: xgb_model,
}
rows=[]
importances={}
for name, m in models.items():
    use_scaled = "Logistic" in name
    m.fit(Xtr_s if use_scaled else Xtr, ytr)
    pred = m.predict(Xte_s if use_scaled else Xte)
    rows.append({"model":name,"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred,zero_division=0),"recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0)})
    if hasattr(m,"feature_importances_"):
        importances[name]=sorted(zip(features,m.feature_importances_), key=lambda x:-x[1])

comp = pd.DataFrame(rows).sort_values("f1",ascending=False)
comp.to_csv(os.path.join(BASE,"results","model_comparison.csv"), index=False)

best = comp.iloc[0]
top_corr = corr["turnover_growth"].drop(["turnover_growth","high_growth"]).abs().sort_values(ascending=False)
with open(os.path.join(BASE,"results","results_summary.txt"),"w") as f:
    f.write("UK SME Strategic Growth Prediction — SYNTHETIC RECREATION RESULTS\n")
    f.write("Dataset: synthetic, 248 SMEs, seed 42, 14 indicators, target high/low growth by median turnover growth\n")
    f.write(f"Class balance high_growth: {df['high_growth'].mean():.3f} (n_high={int(df['high_growth'].sum())}, n_low={int((1-df['high_growth']).sum())})\n")
    f.write(f"Linear regression R2 (turnover_growth, test): {r2:.4f}\n")
    f.write(f"Median turnover growth: {median_growth:.4f}\n")
    f.write("Top correlations with turnover_growth (abs):\n")
    for k,v in top_corr.head(5).items(): f.write(f"  {k}: {corr.loc[k,'turnover_growth']:.4f}\n")
    f.write("Model comparison (test, 25% holdout):\n")
    for _,r in comp.iterrows(): f.write(f"  {r['model']}: accuracy={r['accuracy']:.4f} precision={r['precision']:.4f} recall={r['recall']:.4f} f1={r['f1']:.4f}\n")
    f.write(f"Best model by F1: {best['model']} (F1={best['f1']:.4f}, accuracy={best['accuracy']:.4f})\n")
    for name, imp in importances.items():
        f.write(f"Top features — {name}: " + ", ".join(f"{k}={v:.3f}" for k,v in imp[:5]) + "\n")
print(comp.to_string(index=False)); print(f"R2={r2:.4f} best={best['model']}")
