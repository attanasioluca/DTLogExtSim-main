import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import optuna

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
TARGET_COLS = ["kpi_total_cost", "kpi_cycle_time", "kpi_waiting_time"]
LOG_TRANSFORM_COLS = ["kpi_cycle_time", "kpi_waiting_time"]
LOG_COL_IDX = [TARGET_COLS.index(c) for c in LOG_TRANSFORM_COLS]
NON_FEATURE_COLS = [
    "scenario_id",
    "kpi_total_cost", 
    "kpi_cycle_time", 
    "kpi_waiting_time", 
    "n_reps_used",
    "converged", "converged_wait", "converged_cost", "converged_duration",
]
CONVERGENCE_FLAGS = ["converged", "converged_wait", "converged_cost", "converged_duration"]

def main(SOURCE="synthetic", train_num=40000, n_trials=100):
    DATA_FILE = f"data/{SOURCE}/sim_data_waiting_times.csv" 
    print(f"Loading dataset from {DATA_FILE}...")

    df = pd.read_csv(DATA_FILE)
    n_before = len(df)
    df = df[df[CONVERGENCE_FLAGS].all(axis=1)].reset_index(drop=True)
    print(f"Filtered unconverged runs: {len(df)} / {n_before} rows remain.")
    
    X_df = df.drop(columns=NON_FEATURE_COLS)
    y_df = df[TARGET_COLS]

    y_raw = y_df.values.astype(np.float64)
    y_log = y_raw.copy()
    y_log[:, LOG_COL_IDX] = np.log1p(y_log[:, LOG_COL_IDX])

    input_size = X_df.shape[1]
    output_size = y_df.shape[1]
    print(f"Features: {input_size} | Targets: {output_size}")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_df.values, y_log, test_size=0.20, random_state=42
    )

    X_train = X_train_full[:train_num]
    y_train = y_train_full[:train_num]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }

        base_lgbm = lgb.LGBMRegressor(**params)
        model = MultiOutputRegressor(base_lgbm)
        model.fit(X_train_scaled, y_train_scaled)

        preds = model.predict(X_test_scaled)
        val_mse = np.mean((preds - y_test_scaled) ** 2)
        return val_mse

    print(f"\nStarting Optuna Hyperparameter Optimization ({n_trials} trials)...")
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials)

    print(f"\nBest Trial Validation Scaled MSE: {study.best_trial.value:.5f}")
    best_params = study.best_trial.params
    hyperparams = {
        "n_estimators": int(best_params["n_estimators"]),
        "learning_rate": float(best_params["learning_rate"]),
        "num_leaves": int(best_params["num_leaves"]),
        "max_depth": int(best_params["max_depth"]),
        "subsample": float(best_params["subsample"]),
        "colsample_bytree": float(best_params["colsample_bytree"]),
        "reg_alpha": float(best_params["reg_alpha"]),
        "reg_lambda": float(best_params["reg_lambda"])
    }

    base_dir = Path(__file__).parent
    source_out_dir = base_dir / f"output/{SOURCE}"
    source_out_dir.mkdir(parents=True, exist_ok=True)
    
    source_json = source_out_dir / "best_params.json"
    generic_json = base_dir / "best_params.json"
    
    with open(source_json, "w") as f:
        json.dump(hyperparams, f, indent=4)
    with open(generic_json, "w") as f:
        json.dump(hyperparams, f, indent=4)

    print("\n" + "="*50)
    print("LIGHTGBM HYPERPARAMETERS TUNED & SAVED")
    print("="*50)
    for k, v in hyperparams.items():
        print(f"{k.upper()} = {v}")
    print("-" * 50)
    print(f"Saved config automatically to:")
    print(f"  - {source_json}")
    print(f"  - {generic_json}")
    print("="*50 + "\n")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="synthetic", help="Dataset source (default: synthetic)")
    parser.add_argument("--train_num", type=int, default=40000, help="Number of training samples")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of Optuna trials")
    args = parser.parse_args()
    main(args.source, train_num=args.train_num, n_trials=args.n_trials)
