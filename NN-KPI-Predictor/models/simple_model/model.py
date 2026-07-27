from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import time
import argparse

# ==========================================
# 1. PYTORCH DATASET DEFINITION
# ==========================================
class SimulationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

# ==========================================
# 2. NEURAL NETWORK ARCHITECTURE
# ==========================================
class PharmacySurrogate(nn.Module):
    def __init__(self, input_size):
        super(PharmacySurrogate, self).__init__()
        
        self.shared_entry = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Dropout(0.2)
        )
        self.shared_out = nn.Sequential(nn.Linear(64, 32), nn.Mish())

        # Branch 1: Total Cost
        self.cost_branch = nn.Sequential(nn.Linear(32, 16), nn.Mish(), nn.Linear(16, 1))
        # Branch 2: Cycle Time
        self.cycle_time_branch = nn.Sequential(nn.Linear(32, 16), nn.Mish(), nn.Linear(16, 1))
        # Branch 3: Waiting Time
        self.waiting_time_branch = nn.Sequential(nn.Linear(32, 16), nn.Mish(), nn.Linear(16, 1))

    def forward(self, x):
        x = self.shared_entry(x)
        shared_features = self.shared_out(x)
        
        t_cost = self.cost_branch(shared_features)
        cycle = self.cycle_time_branch(shared_features)
        wait = self.waiting_time_branch(shared_features)

        return torch.cat((t_cost, cycle, wait), dim=1)


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
def inverse_transform_targets_torch(y_scaled, y_mean_tensor, y_scale_tensor):
    y_unscaled = (y_scaled * y_scale_tensor) + y_mean_tensor
    cols = []
    for i in range(y_unscaled.shape[1]):
        col = y_unscaled[:, i]
        cols.append(torch.expm1(col) if i in LOG_COL_IDX else col)
    return torch.stack(cols, dim=1)

# ==========================================
# 3. MAIN TRAINING PIPELINE
# ==========================================
def main(SOURCE="synthetic", train_num=40000):
    DATA_FILE = f"data/{SOURCE}/sim_data_waiting_times.csv"
    BATCH_SIZE = 64
    EPOCHS = 2500
    LEARNING_RATE = 0.5e-3
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available(): 
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    print("Loading dataset from source:", SOURCE)
    df = pd.read_csv(DATA_FILE) 

    # 1b. Drop simulation runs that didn't converge -- their KPI estimates
    # aren't reliable ground truth, so they shouldn't be training targets.
    n_before = len(df)
    df = df[df[CONVERGENCE_FLAGS].all(axis=1)].reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} unconverged rows ({len(df)} remain).")

    X_df = df.drop(columns=NON_FEATURE_COLS)
    y_df = df[TARGET_COLS]

    input_size = X_df.shape[1]
    output_size = y_df.shape[1]
    print(f"Features: {input_size} | Targets: {output_size}")

    # 1c. Log1p the heavy-tailed KPIs so extreme (but real) outlier scenarios
    # don't dominate the MSE loss once everything is standardized.
    y_raw = y_df.values.astype(np.float64)
    y_log = y_raw.copy()
    y_log[:, LOG_COL_IDX] = np.log1p(y_log[:, LOG_COL_IDX])

    # 2. Split the FULL dataset once to create a universal, locked test set.
    # Split X, the log-transformed y (used for training), and the raw y
    # (used later to report real-world-unit metrics) together so indices
    # stay aligned.
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

    output_dir = Path(f"models/simple_model/output/{SOURCE}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the scalers
    joblib.dump(x_scaler, output_dir / "x_scaler.pkl")
    joblib.dump(y_scaler, output_dir / "y_scaler.pkl")
    print("Saved x_scaler.pkl and y_scaler.pkl")

    # --- B. DataLoader Setup ---
    train_dataset = SimulationDataset(X_train_scaled, y_train_scaled)
    test_dataset = SimulationDataset(X_test_scaled, y_test_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- C. Model Initialization ---
    model = PharmacySurrogate(input_size).to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    print(f"Starting training for up to {EPOCHS} epochs...")
    start_time = time.time()
    
    best_test_loss = float('inf')
    patience = 200
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # Backward pass & Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        scheduler.step()  # Updates learning rate
        
        # --- E. Validation/Testing Loop ---
        model.eval()
        test_loss = 0.0
        test_preds_scaled = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item() * batch_X.size(0)

                test_preds_scaled.append(predictions.cpu().numpy())
                
        test_loss /= len(test_loader.dataset)
        
        # Print progress every 5 epochs using MedAE and MAE instead of RMSE
        if (epoch + 1) % 5 == 0 or epoch == 0:
            preds_scaled = np.vstack(test_preds_scaled)

            # Undo StandardScaler + log1p to get back to real-world units
            preds_real = inverse_transform_targets(preds_scaled, y_scaler)
            
            # Calculate Absolute Errors
            absolute_errors = np.abs(preds_real - y_test_raw)
            
            # Calculate MedAE and MAE
            medae_raw = np.median(absolute_errors, axis=0)
            mae_raw = np.mean(absolute_errors, axis=0)
            
            # Calculate percentage based on MedAE vs the Median True Value
            median_true_values = np.median(y_test_raw, axis=0)
            medae_pct = (medae_raw / np.abs(median_true_values)) * 100
            
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
            print(f"   ↳ MedAE -> Cost: ${medae_raw[0]:.2f} (±{medae_pct[0]:.1f}%) | "
                  f"Cycle Time: {medae_raw[1]:.1f}s (±{medae_pct[1]:.1f}%) | "
                  f"Wait Time: {medae_raw[2]:.1f}s (±{medae_pct[2]:.1f}%)")
            print(f"   ↳ MAE   -> Cost: ${mae_raw[0]:.2f} | Cycle Time: {mae_raw[1]:.1f}s | Wait Time: {mae_raw[2]:.1f}s")

        # --- EARLY STOPPING LOGIC ---
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            # Overwrite the saved model ONLY when test loss reaches a new low
            torch.save(model.state_dict(), f"models/simple_model/output/{SOURCE}/surrogate_model.pth")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at Epoch {epoch+1}!")
            print(f"Best Test Loss achieved: {best_test_loss:.4f}")
            break 

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f} seconds.")
    print("Saved the BEST trained model weights to surrogate_model.pth")

    # ==========================================
    # FINAL EVALUATION & JSON EXPORT
    # ==========================================
    print("\nEvaluating BEST model for JSON export...")
    
    # Loads the best weights
    model.load_state_dict(torch.load(f"models/simple_model/output/{SOURCE}/surrogate_model.pth"))
    model.eval()
    
    all_preds = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X)
            all_preds.append(preds.cpu().numpy())
            
    predictions_scaled = np.vstack(all_preds)

    # Undo StandardScaler + log1p to get real-world-unit predictions
    predictions_real = inverse_transform_targets(predictions_scaled, y_scaler)

    # Calculate Absolute Errors for final JSON export
    absolute_errors = np.abs(predictions_real - y_test_raw)
    medae_kpi = np.median(absolute_errors, axis=0)
    mae_kpi = np.mean(absolute_errors, axis=0)
    
    median_true_values = np.median(y_test_raw, axis=0)
    percentage_medae = (medae_kpi / np.abs(median_true_values)) * 100

    import json
    metrics = {
        "Data Source": SOURCE,
        "Model Name": "Simple NN",
        "Best Test Loss": float(best_test_loss),
        "MedAE": {
            "cost": float(medae_kpi[0]), 
            "cycle_time": float(medae_kpi[1]), 
            "waiting_time": float(medae_kpi[2])
        },
        "MAE": {
            "cost": float(mae_kpi[0]), 
            "cycle_time": float(mae_kpi[1]), 
            "waiting_time": float(mae_kpi[2])
        },
        "MedAE_Percentage": {
            "cost": float(percentage_medae[0]), 
            "cycle_time": float(percentage_medae[1]), 
            "waiting_time": float(percentage_medae[2])
        }
    }

    metrics_file = Path("models/output/metrics_simple_nn.json")

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