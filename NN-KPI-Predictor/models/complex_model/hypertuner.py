import json
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

class SimulationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class SurrogateModel(nn.Module):
    def __init__(self, input_size, dropout_rate=0.1):
        super(SurrogateModel, self).__init__()
        self.shared_entry = nn.Sequential(nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.Mish(), nn.Dropout(dropout_rate))
        self.shared_h1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.Mish()
        self.drop1 = nn.Dropout(dropout_rate)
        self.shared_h2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.Mish()
        self.drop2 = nn.Dropout(dropout_rate)
        self.shared_out = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.Mish())

        self.total_cost_branch   = nn.Sequential(nn.Linear(64, 32), nn.Mish(), nn.Linear(32, 1))
        self.cycle_time_branch   = nn.Sequential(nn.Linear(64, 32), nn.Mish(), nn.Linear(32, 1))
        self.waiting_time_branch = nn.Sequential(nn.Linear(64, 32), nn.Mish(), nn.Linear(32, 1))

    def forward(self, x):
        x = self.shared_entry(x)
        h1 = self.drop1(self.act1(self.bn1(self.shared_h1(x))))
        h2 = self.drop2(self.act2(self.bn2(self.shared_h2(h1))))
        h2 = h2 + h1 
        shared_features = self.shared_out(h2)
        return torch.cat((self.total_cost_branch(shared_features), self.cycle_time_branch(shared_features), self.waiting_time_branch(shared_features)), dim=1) 

def main(SOURCE="BIMP", train_num=40000):
    DATA_FILE = f"data/{SOURCE}/sim_data_waiting_times.csv" 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(DATA_FILE)
    
    # 1. Filter unconverged runs
    df = df[df[CONVERGENCE_FLAGS].all(axis=1)].reset_index(drop=True)

    X_df = df.drop(columns=NON_FEATURE_COLS)
    y_df = df[TARGET_COLS]

    # 2. Selectively Log1p heavy-tailed KPIs
    y_raw = y_df.values.astype(np.float64)
    y_log = y_raw.copy()
    y_log[:, LOG_COL_IDX] = np.log1p(y_log[:, LOG_COL_IDX])

    input_size = X_df.shape[1]
    X_train_full, X_test, y_train_full, y_test = train_test_split(X_df.values, y_log, test_size=0.20, random_state=42)

    X_train = X_train_full[:train_num]
    y_train = y_train_full[:train_num]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        
        train_loader = DataLoader(SimulationDataset(X_train_scaled, y_train_scaled), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(SimulationDataset(X_test_scaled, y_test_scaled), batch_size=batch_size, shuffle=False)

        model = SurrogateModel(input_size, dropout_rate=dropout_rate).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss() 
        
        for epoch in range(100):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                predictions = model(batch_X)
                
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_kpi_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    val_kpi_loss += criterion(model(batch_X), batch_y).item() * batch_X.size(0)
                    
            val_kpi_loss /= len(test_loader.dataset)
            trial.report(val_kpi_loss, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()

        return val_kpi_loss

    study = optuna.create_study(
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=15), 
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=100, timeout=36000)
    
    print(f"\nBest Trial Validation MSE: {study.best_trial.value:.5f}")

    best_params = study.best_trial.params
    hyperparams = {
        "batch_size": int(best_params["batch_size"]),
        "learning_rate": float(best_params["lr"]),
        "weight_decay": float(best_params["weight_decay"]),
        "dropout_rate": float(best_params["dropout_rate"])
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

    print(f"BATCH_SIZE = {hyperparams['batch_size']}")
    print(f"LEARNING_RATE = {hyperparams['learning_rate']}")
    print(f"WEIGHT_DECAY = {hyperparams['weight_decay']}")
    print(f"DROPOUT_RATE = {hyperparams['dropout_rate']}")
    print("-" * 50)
    print(f"Saved config automatically to:")
    print(f"  - {source_json}")
    print(f"  - {generic_json}")
    print("\nOr run with environment variables:")
    print(f"export BATCH_SIZE={hyperparams['batch_size']} LEARNING_RATE={hyperparams['learning_rate']} WEIGHT_DECAY={hyperparams['weight_decay']} DROPOUT_RATE={hyperparams['dropout_rate']}")
    print("="*50 + "\n")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="synthetic", help="Dataset source (default: synthetic)")
    parser.add_argument("--train_num", type=int, default=100000, help="Number of training samples")
    args = parser.parse_args()
    main(args.source, train_num=args.train_num)