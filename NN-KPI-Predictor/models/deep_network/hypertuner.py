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
    def forward(self, x):
        return x + self.net(x)

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
                nn.Linear(hidden_dim, 64), nn.LayerNorm(64), nn.Mish(),
                nn.Linear(64, 32), nn.Mish(),
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

def main(SOURCE="real", train_num=40000):
    DATA_FILE = f"data/{SOURCE}/sim_data_waiting_times.csv" 
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    df = pd.read_csv(DATA_FILE)
    df = df[df[CONVERGENCE_FLAGS].all(axis=1)].reset_index(drop=True)
    
    X_df = df.drop(columns=NON_FEATURE_COLS)
    y_df = df[TARGET_COLS]

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
        lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.45)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
        hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512])
        num_blocks = trial.suggest_int("num_blocks", 4, 8)
        
        train_loader = DataLoader(SimulationDataset(X_train_scaled, y_train_scaled), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(SimulationDataset(X_test_scaled, y_test_scaled), batch_size=batch_size, shuffle=False)

        model = DeepSurrogateModel(input_size, hidden_dim, num_blocks, dropout_rate).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
        criterion = nn.MSELoss() 
        
        for epoch in range(300):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                predictions = model(batch_X)
                
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
                
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
        "hidden_dim": int(best_params["hidden_dim"]),
        "num_blocks": int(best_params["num_blocks"]),
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

    print("\n" + "="*50)
    print("🚀 HYPERPARAMETERS TUNED & AUTOMATICALLY LINKED!")
    print("="*50)
    print(f"BATCH_SIZE = {hyperparams['batch_size']}")
    print(f"LEARNING_RATE = {hyperparams['learning_rate']}")
    print(f"WEIGHT_DECAY = {hyperparams['weight_decay']}")
    print(f"HIDDEN_DIM = {hyperparams['hidden_dim']}")
    print(f"NUM_BLOCKS = {hyperparams['num_blocks']}")
    print(f"DROPOUT_RATE = {hyperparams['dropout_rate']}")
    print("-" * 50)
    print(f"Saved config automatically to:")
    print(f"  - {source_json}")
    print(f"  - {generic_json}")
    print("\nOr run with environment variables:")
    print(f"export BATCH_SIZE={hyperparams['batch_size']} LEARNING_RATE={hyperparams['learning_rate']} WEIGHT_DECAY={hyperparams['weight_decay']} HIDDEN_DIM={hyperparams['hidden_dim']} NUM_BLOCKS={hyperparams['num_blocks']} DROPOUT_RATE={hyperparams['dropout_rate']}")
    print("="*50 + "\n")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="synthetic", help="Dataset source")
    parser.add_argument("--train_num", type=int, default=100000, help="Number of training samples")
    args = parser.parse_args()
    main(args.source, train_num=args.train_num)