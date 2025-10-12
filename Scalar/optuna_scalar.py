#!/usr/bin/env python3
"""
optuna_scalar.py

Usage examples:
    python optuna_scalar.py path/to/Konda_pIC50_train.csv
    python optuna_scalar.py data.csv --target-col target --id-col COMPOUND_ID --n-trials 50 --log-file runs.csv
"""

import os
import time
import argparse
import datetime
import json
import sys

import optuna # type: ignore
import pandas as pd # type: ignore
from sklearn.model_selection import cross_val_score # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from molfeat.trans.fp import FPVecTransformer  # type: ignore



def parse_args():
    p = argparse.ArgumentParser(description="Run timed Optuna study on a CSV dataset.")
    p.add_argument("csv", help="Path to training CSV file")
    p.add_argument("--target-col", default="target", help="Name of the target column (default: 'target')")
    p.add_argument("--id-col", default="COMPOUND_ID", help="Name of ID column to drop (default: 'COMPOUND_ID')")
    p.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials (default: 20)")
    p.add_argument("--direction", choices=["maximize", "minimize"], default="maximize", help="Optimization direction")
    p.add_argument("--cv", type=int, default=5, help="Cross-validation folds for scoring (default: 5)")
    p.add_argument("--log-file", default=None, help="Optional CSV file to append run summary (timestamp, runtime, best value, best params)")
    p.add_argument("--outdir", type=str, default="results", help="Output directory for results")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    p.add_argument("--smiles-col", default=None,
               help="If provided, compute ECFP fingerprints from this SMILES column instead of using raw features")
    p.add_argument("--fp-kind", default="ecfp", help="Fingerprint type (ecfp, maccs, rdkit, etc.)")
    p.add_argument("--fp-length", type=int, default=2048, help="Fingerprint length (default: 2048)")
    return p.parse_args()

def load_data(path, id_col="COMPOUND_ID", target_col="target", smiles_col=None):
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in CSV columns: {list(df.columns)}")

    if smiles_col and smiles_col in df.columns:
        print(f"Using SMILES column '{smiles_col}' to generate ECFP fingerprints...", flush=True)
        fp = FPVecTransformer(kind="ecfp", length=2048, n_jobs=4)
        X = fp(df[smiles_col].tolist())
    else:
        drop_cols = [c for c in [id_col, target_col, smiles_col] if c in df.columns and c]
        X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col].values
    return X, y


def make_objective(X, y, cv=5, random_state=42):
    """
    Return an objective function that closes over X and y, so we don't reload the CSV each trial.
    Modify the model construction and hyperparameter search space as needed.
    """
    def objective(trial):
        # Define hyperparameter search space
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        max_depth = trial.suggest_int("max_depth", 3, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        # Use cross_val_score on X,y that are in scope
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
        return float(scores.mean())
    return objective

def timed_optuna_study(objective, n_trials=50, direction="maximize"):
    start = time.time()
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    duration = time.time() - start

    # Print and store runtime
    print(f"\nStudy took {duration/60:.2f} minutes ({duration:.1f} seconds)", flush=True)
    study.set_user_attr("runtime_sec", duration)
    return study, duration

def append_run_log(log_file, csv_path, duration, study):
    """Append a one-line summary to a CSV file (create if missing)."""
    timestamp = datetime.datetime.now().isoformat()
    best_value = float(study.best_value) if study.best_trial is not None else None
    best_params = json.dumps(study.best_params) if study.best_trial is not None else ""
    df_row = pd.DataFrame([{
        "timestamp": timestamp,
        "csv": csv_path,
        "runtime_sec": duration,
        "best_value": best_value,
        "best_params": best_params
    }])
    header = not pd.io.common.file_exists(log_file)
    df_row.to_csv(log_file, mode="a", index=False, header=header)

def main():
    args = parse_args()
    try:
        X, y = load_data(args.csv,
                 id_col=args.id_col,
                 target_col=args.target_col,
                 smiles_col=args.smiles_col)
    except Exception as e:
        print(f"Failed to load data: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"Loaded data: X shape = {X.shape}, y length = {len(y)}", flush=True)

    objective = make_objective(X, y, cv=args.cv, random_state=args.random_state)

    study, duration = timed_optuna_study(objective, n_trials=args.n_trials, direction=args.direction)

    # Print best result summary
    print("\nBest Trial:")
    if study.best_trial is not None:
        print(f"  Value: {study.best_trial.value:.6f}")
        print(f"  Params: {study.best_trial.params}")
        best_params = study.best_params
        pd.DataFrame([best_params]).to_csv(os.path.join(args.outdir, "best_params.csv"), index=False)
        print(f"Saved best parameters to {os.path.join(args.outdir, 'best_params.csv')}")
    else:
        print("  No completed trials found.")

    print(f"Recorded runtime (sec): {study.user_attrs.get('runtime_sec')}", flush=True)

    if args.log_file:
        try:
            append_run_log(args.log_file, args.csv, duration, study)
            print(f"Appended run summary to {args.log_file}", flush=True)
        except Exception as e:
            print(f"Failed to write log file: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
