#!/usr/bin/env python3
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
import argparse

# -------------------------
# Command-line arguments
# -------------------------
p = argparse.ArgumentParser(description="Evaluate model on holdout dataset using optuna_scalar saved best parameters.")
p.add_argument("--train", type=str, required=True, help="Path to training CSV file (to refit full model)")
p.add_argument("--holdout", type=str, required=True, help="Path to holdout CSV file")
p.add_argument("--indir", type=str, default="results", help="Directory containing best_params.csv")
args = p.parse_args()

# -------------------------
# Load data
# -------------------------
df_train = pd.read_csv(args.train)
X = df_train.drop(columns=["COMPOUND_ID", "target"])
y = df_train["target"]

df_holdout = pd.read_csv(args.holdout)
X_holdout = df_holdout.drop(columns=["COMPOUND_ID", "target"])
y_holdout = df_holdout["target"]

# -------------------------
# Load best params
# -------------------------
params_path = os.path.join(args.indir, "best_params.csv")
if not os.path.exists(params_path):
    raise FileNotFoundError(f"Cannot find {params_path}. Run train_optuna_study.py first.")

best_params = pd.read_csv(params_path).iloc[0].to_dict()
print("Loaded best parameters from:", params_path)
for k, v in best_params.items():
    print(f"  {k}: {v}")

# -------------------------
# Refit and evaluate
# -------------------------
model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
model.fit(X, y)
y_pred = model.predict(X_holdout)

r2 = r2_score(y_holdout, y_pred)
rmse = root_mean_squared_error(y_holdout, y_pred)
print(f"\nHoldout Evaluation — R²: {r2:.3f}, RMSE: {rmse:.3f}")

# -------------------------
# Save model and plot
# -------------------------
os.makedirs(args.indir, exist_ok=True)
model_path = os.path.join(args.indir, "best_random_forest_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")

# Plot predicted vs actual
plt.figure(figsize=(6,6))
plt.scatter(y_holdout, y_pred, alpha=0.6, edgecolor='k')
plt.plot([min(y_holdout), max(y_holdout)], [min(y_holdout), max(y_holdout)], 'r--')
plt.title(f"Holdout Predictions\nR²={r2:.3f} | RMSE={rmse:.3f}")
plt.xlabel("Actual pIC₅₀")
plt.ylabel("Predicted pIC₅₀")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plot_path = os.path.join(args.indir, "holdout_predictions.png")
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"Plot saved to: {plot_path}")
