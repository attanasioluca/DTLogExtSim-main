import argparse
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
import os
import json

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

def load_hyperparameters(source=None, base_dir=None):
    """
    Loads hyperparameter settings using precedence:
    1. Environment variables (BATCH_SIZE, LEARNING_RATE/LR, WEIGHT_DECAY, HIDDEN_DIM, NUM_BLOCKS, DROPOUT_RATE)
    2. Source-specific JSON config (output/{source}/best_params.json)
    3. Global model JSON config (best_params.json)
    4. Default fallbacks
    """
    if base_dir is None:
        base_dir = Path(__file__).parent

    defaults = {
        "batch_size": 128,
        "learning_rate": 0.0004942350005629194,
        "weight_decay": 0.00969342764145206,
        "hidden_dim": 512,
        "num_blocks": 6,
        "dropout_rate": 0.32274561809704083
    }
    params = defaults.copy()

    # 1. Source-specific JSON file
    if source:
        source_json = base_dir / f"output/{source}/best_params.json"
        if source_json.exists():
            try:
                with open(source_json, "r") as f:
                    params.update(json.load(f))
            except Exception:
                pass

    # 2. Global model JSON file (if source JSON did not exist)
    generic_json = base_dir / "best_params.json"
    if generic_json.exists():
        try:
            with open(generic_json, "r") as f:
                loaded = json.load(f)
                for k, v in loaded.items():
                    if source and (base_dir / f"output/{source}/best_params.json").exists():
                        continue
                    params[k] = v
        except Exception:
            pass

    # 3. Environment Variables Override
    if os.environ.get("BATCH_SIZE"):
        params["batch_size"] = int(os.environ["BATCH_SIZE"])
    if os.environ.get("LEARNING_RATE"):
        params["learning_rate"] = float(os.environ["LEARNING_RATE"])
    elif os.environ.get("LR"):
        params["learning_rate"] = float(os.environ["LR"])
    if os.environ.get("WEIGHT_DECAY"):
        params["weight_decay"] = float(os.environ["WEIGHT_DECAY"])
    if os.environ.get("HIDDEN_DIM"):
        params["hidden_dim"] = int(os.environ["HIDDEN_DIM"])
    if os.environ.get("NUM_BLOCKS"):
        params["num_blocks"] = int(os.environ["NUM_BLOCKS"])
    if os.environ.get("DROPOUT_RATE"):
        params["dropout_rate"] = float(os.environ["DROPOUT_RATE"])

    return params

DEFAULT_PARAMS = load_hyperparameters()
BATCH_SIZE = int(DEFAULT_PARAMS["batch_size"])
LEARNING_RATE = float(DEFAULT_PARAMS.get("learning_rate", DEFAULT_PARAMS.get("lr", 0.00049)))
WEIGHT_DECAY = float(DEFAULT_PARAMS["weight_decay"])
HIDDEN_DIM = int(DEFAULT_PARAMS["hidden_dim"])
NUM_BLOCKS = int(DEFAULT_PARAMS["num_blocks"])
DROPOUT_RATE = float(DEFAULT_PARAMS["dropout_rate"])

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

class SimulationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class ResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate)
        )
    def forward(self, x): return x + self.net(x) 

class DeepSurrogateModel(nn.Module):
    def __init__(self, input_size, hidden_dim=256, num_blocks=4, dropout_rate=0.1):
        super(DeepSurrogateModel, self).__init__()
        
        self.entry = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish()
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)])
        self.shared_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish()
        )
        def make_branch():
            return nn.Sequential(
                nn.Linear(hidden_dim, 64), 
                nn.LayerNorm(64), 
                nn.Mish(),
                nn.Linear(64, 32), 
                nn.Mish(),
                nn.Linear(32, 1)
            )
            
        self.total_cost_branch   = make_branch()
        self.cycle_time_branch   = make_branch()
        self.waiting_time_branch = make_branch()

    def forward(self, x):
        x = self.entry(x)
        x = self.blocks(x)
        shared = self.shared_out(x)
        return torch.cat((
            self.total_cost_branch(shared), 
            self.cycle_time_branch(shared), 
            self.waiting_time_branch(shared)
        ), dim=1) 

def main(SOURCE="real", train_num=5000):
    DATA_FILE = f"data/{SOURCE}/sim_data_waiting_times.csv" 
    EPOCHS = int(os.environ.get("EPOCHS", 2500))
    PATIENCE = int(os.environ.get("PATIENCE", 200))
    TEST_SIZE = float(os.environ.get("TEST_SIZE", 0.20))

    params = load_hyperparameters(source=SOURCE)
    batch_size = int(params["batch_size"])
    learning_rate = float(params.get("learning_rate", params.get("lr")))
    weight_decay = float(params["weight_decay"])
    hidden_dim = int(params["hidden_dim"])
    num_blocks = int(params["num_blocks"])
    dropout_rate = float(params["dropout_rate"])

    print(f"Hyperparameters loaded for source '{SOURCE}': BATCH_SIZE={batch_size}, LEARNING_RATE={learning_rate}, WEIGHT_DECAY={weight_decay}, HIDDEN_DIM={hidden_dim}, NUM_BLOCKS={num_blocks}, DROPOUT_RATE={dropout_rate}")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Loading dataset from source:", SOURCE)
    df = pd.read_csv(DATA_FILE)

    n_before = len(df)
    df = df[df[CONVERGENCE_FLAGS].all(axis=1)].reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} unconverged rows ({len(df)} remain).")

    X_df = df.drop(columns=NON_FEATURE_COLS)
    y_df = df[TARGET_COLS]
    
    input_size = X_df.shape[1]
    output_size = y_df.shape[1]
    print(f"Features: {input_size} | Targets: {output_size}")

    y_raw = y_df.values.astype(np.float64)
    y_log = y_raw.copy()
    y_log[:, LOG_COL_IDX] = np.log1p(y_log[:, LOG_COL_IDX])
    
    X_train_full, X_test, y_train_full, y_test, _, y_test_raw = train_test_split(
        X_df.values, y_log, y_raw, test_size=TEST_SIZE, random_state=42
    )

    X_train = X_train_full[:train_num]
    y_train = y_train_full[:train_num]
    
    print(f"--> Training on {len(X_train)} samples.")
    print(f"--> Testing on {len(X_test)} consistent samples.")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    output_dir = Path(f"models/deep_network/output/{SOURCE}")
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(x_scaler, output_dir / 'x_scaler.pkl')
    joblib.dump(y_scaler, output_dir / 'y_scaler.pkl')

    train_dataset = SimulationDataset(X_train_scaled, y_train_scaled)
    test_dataset = SimulationDataset(X_test_scaled, y_test_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = DeepSurrogateModel(input_size, hidden_dim, num_blocks, dropout_rate).to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    best_test_loss = float('inf')
    patience = PATIENCE
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        
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
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            preds_scaled = np.vstack(test_preds_scaled)
            preds_real = inverse_transform_targets(preds_scaled, y_scaler)
            absolute_errors = np.abs(preds_real - y_test_raw)
            medae_raw = np.median(absolute_errors, axis=0)
            mae_raw = np.mean(absolute_errors, axis=0)
            
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
            print(f"   ↳ MedAE -> Cost: ${medae_raw[0]:.2f} | Cycle Time: {medae_raw[1]:.1f}s | Wait Time: {medae_raw[2]:.1f}s")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "surrogate_model.pth")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at Epoch {epoch+1}!")
            break 

    print(f"Training finished in {time.time() - start_time:.2f}s.")

    # ==========================================
    # FINAL EVALUATION & JSON EXPORT
    # ==========================================
    print("\nEvaluating BEST model for JSON export...")
    model.load_state_dict(torch.load(output_dir / "surrogate_model.pth"))
    model.eval()
    
    all_preds = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X)
            all_preds.append(preds.cpu().numpy())
            
    predictions_scaled = np.vstack(all_preds)
    predictions_real = inverse_transform_targets(predictions_scaled, y_scaler)
    
    absolute_errors = np.abs(predictions_real - y_test_raw)
    medae_kpi = np.median(absolute_errors, axis=0)
    mae_kpi = np.mean(absolute_errors, axis=0)
    median_true_values = np.median(y_test_raw, axis=0)
    percentage_medae = (medae_kpi / np.abs(median_true_values)) * 100
    
    metrics = {
        "Data Source": SOURCE,
        "Model Name": "Deep Network - Hypertuned",
        "Train Samples": len(X_train),
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

    metrics_file = Path("models/output/metrics_deep_nn.json")
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = []

    all_metrics = [
        m for m in all_metrics
        if not (
            m.get("Data Source") == SOURCE
            and m.get("Model Name") == metrics["Model Name"]
            and m.get("Train Samples") == metrics["Train Samples"]
        )
    ]
    all_metrics.append(metrics)
    all_metrics.sort(key=lambda x: (x.get("Data Source", ""), x.get("Model Name", ""), x.get("Train Samples", 0)))

    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Exported metrics to {metrics_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="synthetic", help="Dataset source")
    parser.add_argument("--train_num", type=int, default=100000, help="Number of training samples")
    args = parser.parse_args()
    main(args.source, train_num=args.train_num)