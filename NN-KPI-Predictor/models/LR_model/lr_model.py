import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import time
import json

# ==========================================
# CONFIGURATION & CONSTANTS
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

def inverse_transform_targets(y_scaled, y_scaler):
    y_unscaled = y_scaler.inverse_transform(y_scaled)
    y_real = y_unscaled.copy()
    y_real[:, LOG_COL_IDX] = np.expm1(y_real[:, LOG_COL_IDX])
    return y_real

# ==========================================
# LINEAR REGRESSION BASELINE
# ==========================================
def main(SOURCE="synthetic", train_num=40000):
    DATA_FILE = f"data/{SOURCE}/sim_data_waiting_times.csv"
    
    print("Loading dataset from source:", SOURCE)
    df = pd.read_csv(DATA_FILE) 

    # 1b. Drop simulation runs that didn't converge
    n_before = len(df)
    df = df[df[CONVERGENCE_FLAGS].all(axis=1)].reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} unconverged rows ({len(df)} remain).")

    X_df = df.drop(columns=NON_FEATURE_COLS)
    y_df = df[TARGET_COLS]

    input_size = X_df.shape[1]
    output_size = y_df.shape[1]
    print(f"Features: {input_size} | Targets: {output_size}")

    # 1c. Log1p the heavy-tailed KPIs
    y_raw = y_df.values.astype(np.float64)
    y_log = y_raw.copy()
    y_log[:, LOG_COL_IDX] = np.log1p(y_log[:, LOG_COL_IDX])

    # 2. Split the FULL dataset once to create a universal, locked test set
    X_train_full, X_test, y_train_full, y_test, _, y_test_raw = train_test_split(
        X_df.values, y_log, y_raw, test_size=0.2, random_state=42
    )

    TRAIN_SAMPLES = train_num
    
    # 4. Slice the training arrays down to the desired size
    X_train = X_train_full[:TRAIN_SAMPLES]
    y_train = y_train_full[:TRAIN_SAMPLES]
    
    print(f"--> Training on {len(X_train)} samples.")
    print(f"--> Testing on {len(X_test)} consistent samples.")

    print("Scaling data...")
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    # 5. Fit scalers ONLY on the active training subset
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    
    # Ensure directories exist and save scalers
    output_dir = Path(f"models/LR_model/output/{SOURCE}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(x_scaler, output_dir / 'lr_x_scaler.pkl')
    joblib.dump(y_scaler, output_dir / 'lr_y_scaler.pkl')
    print("Saved lr_x_scaler.pkl and lr_y_scaler.pkl")

    print("\nTraining Linear Regression Model...")
    start_time = time.time()
    
    # Scikit-learn's LinearRegression handles multiple outputs natively
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)
    
    # Save the model
    joblib.dump(model, output_dir / 'lr_model.pkl')
    
    train_time = time.time() - start_time
    print(f"Training complete in {train_time:.4f} seconds.")
    print("Saved trained model to lr_model.pkl")

    # ==========================================
    # EVALUATION & METRICS
    # ==========================================
    print("\nEvaluating on Test Set...")
    
    # Predict in scaled space
    predictions_scaled = model.predict(X_test_scaled)
    
    # Scaled test loss (MSE over all scaled targets)
    test_mse_scaled = np.mean((predictions_scaled - y_test_scaled) ** 2)
    
    # Undo StandardScaler + log1p to get back to real-world units
    preds_real = inverse_transform_targets(predictions_scaled, y_scaler)
    
    # Calculate Absolute Errors
    absolute_errors = np.abs(preds_real - y_test_raw)
    
    # Calculate MedAE and MAE
    medae_raw = np.median(absolute_errors, axis=0)
    mae_raw = np.mean(absolute_errors, axis=0)
    
    # Calculate percentage based on MedAE vs the Median True Value
    median_true_values = np.median(y_test_raw, axis=0)
    medae_pct = (medae_raw / np.abs(median_true_values)) * 100
    
    print("\n=====================================================================")
    print("                    LINEAR REGRESSION RESULTS")
    print("=====================================================================")
    print(f"Test Loss (Scaled MSE): {test_mse_scaled:.4f}")
    print(f"   ↳ MedAE -> Cost: ${medae_raw[0]:.2f} (±{medae_pct[0]:.1f}%) | "
          f"Cycle Time: {medae_raw[1]:.1f}s (±{medae_pct[1]:.1f}%) | "
          f"Wait Time: {medae_raw[2]:.1f}s (±{medae_pct[2]:.1f}%)")
    print(f"   ↳ MAE   -> Cost: ${mae_raw[0]:.2f} | Cycle Time: {mae_raw[1]:.1f}s | Wait Time: {mae_raw[2]:.1f}s")
    print("=====================================================================")

    # ==========================================
    # JSON EXPORT
    # ==========================================
    metrics = {
        "Data Source": SOURCE,
        "Model Name": "Linear Regression Baseline",
        "Best Test Loss": float(test_mse_scaled),
        "MedAE": {
            "cost": float(medae_raw[0]), 
            "cycle_time": float(medae_raw[1]), 
            "waiting_time": float(medae_raw[2])
        },
        "MAE": {
            "cost": float(mae_raw[0]), 
            "cycle_time": float(mae_raw[1]), 
            "waiting_time": float(mae_raw[2])
        },
        "MedAE_Percentage": {
            "cost": float(medae_pct[0]), 
            "cycle_time": float(medae_pct[1]), 
            "waiting_time": float(medae_pct[2])
        }
    }

    metrics_file = Path("models/output/metrics_lr_model.json")
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing metrics if they exist
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = []

    # Remove any previous entry for this source + model
    all_metrics = [
        m for m in all_metrics
        if not (
            m["Data Source"] == SOURCE
            and m["Model Name"] == metrics["Model Name"]
        )
    ]

    # Add the updated metrics
    all_metrics.append(metrics)
    all_metrics.sort(key=lambda x: x["Data Source"])

    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Updated metrics saved to {metrics_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        nargs="?",
        default="synthetic",
        help="Dataset source (default: synthetic)"
    )
    parser.add_argument(
        "--train_num",
        type=int,
        default=40000,
        help="Number of training samples to use (default: 40000)"
    )

    args = parser.parse_args()
    main(args.source, train_num=args.train_num)