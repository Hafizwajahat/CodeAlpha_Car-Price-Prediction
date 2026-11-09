"""CodeAlpha Task 3: Car Price Prediction
Dataset: CarPrice_Assignment.csv (Kaggle 'Car Price Prediction' / Geely Auto).
Usage: python car_price_prediction.py CarPrice_Assignment.csv
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

path = sys.argv[1] if len(sys.argv) > 1 else "CarPrice_Assignment.csv"
df = pd.read_csv(path)
print(df.shape); print(df.info()); print(df.isnull().sum().sum(), "missing values")

# --- Cleaning & feature engineering ---
df = df.drop_duplicates()
df = df.drop(columns=[c for c in ["car_ID"] if c in df.columns])
# Brand goodwill: extract brand from CarName and fix typos
df["brand"] = df["CarName"].str.split().str[0].str.lower().replace({
    "maxda": "mazda", "porcshce": "porsche", "toyouta": "toyota",
    "vokswagen": "volkswagen", "vw": "volkswagen"})
df = df.drop(columns="CarName")
df["power_to_weight"] = df["horsepower"] / df["curbweight"]
df["avg_mpg"] = (df["citympg"] + df["highwaympg"]) / 2

# --- EDA ---
plt.figure(figsize=(7, 4)); sns.histplot(df["price"], kde=True)
plt.title("Price distribution"); plt.savefig("price_dist.png", dpi=120); plt.close()
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes("number").corr(), cmap="coolwarm", center=0)
plt.savefig("correlation.png", dpi=120); plt.close()
plt.figure(figsize=(10, 4)); df.groupby("brand")["price"].mean().sort_values().plot.bar()
plt.title("Average price by brand"); plt.tight_layout()
plt.savefig("brand_price.png", dpi=120); plt.close()

# --- Model ---
X, y = df.drop(columns="price"), df["price"]
cat = X.select_dtypes(exclude="number").columns.tolist()
num = X.select_dtypes("number").columns.tolist()
prep = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}
fitted = {}
for name, reg in models.items():
    pipe = Pipeline([("prep", prep), ("model", reg)])
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()
    pipe.fit(X_train, y_train); p = pipe.predict(X_test)
    fitted[name] = (pipe, r2_score(y_test, p))
    print(f"{name:18s} CV R2={cv:.3f} | Test R2={r2_score(y_test, p):.3f} | "
          f"MAE={mean_absolute_error(y_test, p):,.0f} | RMSE={np.sqrt(mean_squared_error(y_test, p)):,.0f}")

best = max(fitted, key=lambda k: fitted[k][1]); pipe = fitted[best][0]
print("Best model:", best)
p = pipe.predict(X_test)
plt.figure(figsize=(5, 5)); plt.scatter(y_test, p, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual price"); plt.ylabel("Predicted price"); plt.title(best)
plt.savefig("actual_vs_pred.png", dpi=120); plt.close()

if hasattr(pipe["model"], "feature_importances_"):
    names = pipe["prep"].get_feature_names_out()
    imp = pd.Series(pipe["model"].feature_importances_, index=names).nlargest(12)
    plt.figure(figsize=(8, 5)); imp[::-1].plot.barh(); plt.title("Top features")
    plt.tight_layout(); plt.savefig("feature_importance.png", dpi=120); plt.close()
    print(imp)
