import argparse
import json
import copy
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_dir not in sys.path: sys.path.append(parent_dir)

from hypertuned_model import (
    DeepPharmacySurrogate,
    NON_FEATURE_COLS,
    CONVERGENCE_FLAGS,
    inverse_transform_targets,
    inverse_transform_targets_torch
)
from helpers.simulator import ScenarioSimulator

def evaluate_scenario(scenario_data, full_model, process_details, num_reps=15):
    simulator = ScenarioSimulator(scenario_data, full_model, process_details, seed=42)
    result = simulator.run_scenario(replications=num_reps, until=86400 * 90)
    
    avg_cost = result.get("total_cost", 0.0)
    avg_dur  = result.get("avg_cycle_time", 0.0)
    avg_wait = result.get("avg_wait_time", 0.0)
    
    return avg_cost, avg_dur, avg_wait

def main(SOURCE="synthetic"):
    # --- CONFIGURATION ---
    MAX_Z_SCORE = 3.0           
    
    BASE_FILE = f"data/{SOURCE}/model/scenario.json"
    MODEL_FILE = f"data/{SOURCE}/model/model.json"
    DATA_FILE = f"data/{SOURCE}/sim_data_waiting_times.csv" 
    
    # STEP 0: LOAD FILES & PREP DYNAMIC TARGETS
    print("[0/4] Loading dataset to determine realistic targets...")
    df_all = pd.read_csv(DATA_FILE)
    
    df = df_all[df_all[CONVERGENCE_FLAGS].all(axis=1)].reset_index(drop=True)
    
    TARGET_COST = df['kpi_total_cost'].quantile(0.05)
    TARGET_DURATION = df['kpi_cycle_time'].quantile(0.05)
    TARGET_WAIT_TIME = df['kpi_waiting_time'].quantile(0.05)
    
    print(f"--- DYNAMIC TARGETS (5th Percentile) ---")
    print(f"Goal Cost:       ${TARGET_COST:.2f}")
    print(f"Goal Cycle Time: {TARGET_DURATION:.1f} seconds")
    print(f"Goal Wait Time:  {TARGET_WAIT_TIME:.1f} seconds\n")

    with open(BASE_FILE, 'r') as f: base_json = json.load(f)
    with open(MODEL_FILE, 'r') as f: full_model = json.load(f)
    
    baseline_scenario = base_json["0"]
    
    process_details = {}
    for p_id, p_data in full_model.get("process_elements", {}).items():
        if p_data.get("node_details"): 
            process_details = p_data
            break
            
    valid_node_ids = set(process_details["node_details"].keys())
    for node_id, node in process_details["node_details"].items():
        if "previous" in node and node["previous"]:
            node["previous"] = [p for p in node["previous"] if p in valid_node_ids]

    print("[1/4] Running Ground-Truth SimPy Evaluation on BASELINE...")
    base_true_cost, base_true_duration, base_true_wait = evaluate_scenario(
        baseline_scenario, full_model, process_details, num_reps=15
    )

    print("\n[2/4] Running Deep Neural Network Optimizer...")
    x_scaler = joblib.load(f'models/deep_network/output/{SOURCE}/x_scaler.pkl')
    y_scaler = joblib.load(f'models/deep_network/output/{SOURCE}/y_scaler.pkl')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Instantiate the Deep model with optimized hyperparameters
    model = DeepPharmacySurrogate(
        input_size=x_scaler.n_features_in_, 
        hidden_dim=512, 
        num_blocks=6, 
        dropout_rate=0.3811733811859317
    ).to(device)
    
    model.load_state_dict(torch.load(f'models/deep_network/output/{SOURCE}/surrogate_model.pth', map_location=device, weights_only=True))
    model.eval()
    for param in model.parameters(): param.requires_grad = False

    X_df = df.drop(columns=NON_FEATURE_COLS)
    X_cols = X_df.columns.tolist()
    
    raw_min_array = X_df.min().values.reshape(1, -1)
    raw_max_array = X_df.max().values.reshape(1, -1)
    
    min_scaled_bounds = torch.tensor(x_scaler.transform(raw_min_array), dtype=torch.float32, device=device)
    max_scaled_bounds = torch.tensor(x_scaler.transform(raw_max_array), dtype=torch.float32, device=device)
    
    NUM_STARTS = 500
    rand_starts = min_scaled_bounds + torch.rand((NUM_STARTS - 1, len(X_cols)), device=device) * (max_scaled_bounds - min_scaled_bounds)
    base_tensor = torch.tensor(x_scaler.transform(X_df.iloc[0].values.reshape(1, -1)), dtype=torch.float32, device=device)
    
    x_optim = nn.Parameter(torch.cat([rand_starts, base_tensor], dim=0), requires_grad=True)

    res_amount_indices = [i for i, col in enumerate(X_cols) if col.startswith("res_") and col.endswith("_amount")]
    x_mean_tensor = torch.tensor(x_scaler.mean_, dtype=torch.float32, device=device)
    x_scale_tensor = torch.tensor(x_scaler.scale_, dtype=torch.float32, device=device)
    y_mean_tensor = torch.tensor(y_scaler.mean_, dtype=torch.float32, device=device)
    y_scale_tensor = torch.tensor(y_scaler.scale_, dtype=torch.float32, device=device)

    optimizer = optim.Adam([x_optim], lr=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-4)
    
    best_global_loss = float('inf')
    best_x_optimal = None

    for epoch in range(10000):
        optimizer.zero_grad()
        predictions = model(x_optim)
        
        raw_preds = inverse_transform_targets_torch(predictions, y_mean_tensor, y_scale_tensor)
        
        pred_avg_cost = raw_preds[:, 0]
        pred_dur      = raw_preds[:, 1]
        pred_wait     = raw_preds[:, 2]
        
        loss_avg_cost = ((pred_avg_cost - TARGET_COST) / TARGET_COST) ** 2
        loss_avg_dur  = ((pred_dur - TARGET_DURATION) / TARGET_DURATION) ** 2
        loss_avg_wait = ((pred_wait - TARGET_WAIT_TIME) / TARGET_WAIT_TIME) ** 2
        
        kpi_loss = loss_avg_cost + loss_avg_dur + loss_avg_wait
        
        x_raw_diff = (x_optim * x_scale_tensor) + x_mean_tensor
        res_amounts = x_raw_diff[:, res_amount_indices]
        fractional_penalty = torch.sum(torch.sin(np.pi * res_amounts) ** 2, dim=1)
        z_score_penalty = torch.sum(torch.relu(torch.abs(x_optim) - MAX_Z_SCORE) ** 2, dim=1)
        
        integer_weight = max(0.0, (epoch - 5000) / 5000.0) * 1000.0 
        loss = (10000.0 * kpi_loss) + (integer_weight * fractional_penalty) + (0.5 * z_score_penalty)
        
        min_loss_in_batch, min_idx = torch.min(loss), torch.argmin(loss)
        if min_loss_in_batch.item() < best_global_loss:
            best_global_loss = min_loss_in_batch.item()
            best_x_optimal = x_optim[min_idx].detach().clone()
        
        loss.sum().backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad(): x_optim.clamp_(min_scaled_bounds, max_scaled_bounds)

    optimized_x_raw = x_scaler.inverse_transform(best_x_optimal.cpu().numpy().reshape(1, -1))[0]

    print("[3/4] Injecting Optimized Parameters...")
    opt_scenario = copy.deepcopy(baseline_scenario)
    discretized_x_raw = np.copy(optimized_x_raw)
    
    for i, col in enumerate(X_cols):
        val = optimized_x_raw[i]
        if col == "arrival_rate_mean":
            val = max(120.0, round(val, 2))
            opt_scenario["arrivalRateDistribution"]["mean"] = str(val)
        elif col.startswith("res_") and col.endswith("_amount"):
            res_name = col.replace("res_", "").replace("_amount", "").replace("_", " ")
            val = max(1, min(15, int(round(val))))  
            for r in opt_scenario.get("resources", []):
                if r["name"] == res_name: r["totalAmount"] = str(val)
        elif col.startswith("res_") and col.endswith("_cost"):
            res_name = col.replace("res_", "").replace("_cost", "").replace("_", " ")
            val = max(10.00, round(val, 2))
            for r in opt_scenario.get("resources", []):
                if r["name"] == res_name: r["costPerHour"] = str(val)
        elif col.startswith("el_") and col.endswith("_duration"):
            el_id = col.replace("el_", "").replace("_duration", "")
            val = max(1.0, round(val, 2))
            for el in opt_scenario.get("elements", []):
                if el["elementId"] == el_id:
                    el["durationDistribution"]["mean"] = str(val)
                    if "standardDeviation" in el["durationDistribution"]:
                        el["durationDistribution"]["standardDeviation"] = "0.0"
        discretized_x_raw[i] = val

    discretized_tensor = torch.tensor(x_scaler.transform(discretized_x_raw.reshape(1, -1)), dtype=torch.float32, device=device)
    with torch.no_grad(): 
        final_pred_scaled = model(discretized_tensor).cpu().numpy()
        
    final_pred = inverse_transform_targets(final_pred_scaled, y_scaler)[0]

    nn_pred_avg_cost = final_pred[0]
    nn_pred_dur_mean = final_pred[1]
    nn_pred_wait_mean = final_pred[2]

    print("[4/4] Running Ground-Truth SimPy Evaluation on OPTIMIZED...")
    opt_true_cost, opt_true_duration, opt_true_wait = evaluate_scenario(
        opt_scenario, full_model, process_details, num_reps=15
    )

    print("\n=====================================================================")
    print("                    VALIDATION & ROI REPORT")
    print("=====================================================================")
    print(f"                | COST (Total)         | CYCLE TIME (Avg)     | WAITING TIME (Avg)")
    print("---------------------------------------------------------------------")
    print(f"TARGET GOAL     | ${TARGET_COST:<19.2f} | {TARGET_DURATION:<19.1f}s | {TARGET_WAIT_TIME:.1f}s")
    print(f"BASELINE (True) | ${base_true_cost:<19.2f} | {base_true_duration:<19.1f}s | {base_true_wait:.1f}s")
    print(f"NN PREDICTED    | ${nn_pred_avg_cost:<19.2f} | {nn_pred_dur_mean:<19.1f}s | {nn_pred_wait_mean:.1f}s")
    print(f"OPTIMIZED (True)| ${opt_true_cost:<19.2f} | {opt_true_duration:<19.1f}s | {opt_true_wait:.1f}s")
    print("---------------------------------------------------------------------")

    print("\n=====================================================================")
    print("                 RECOMMENDED CONFIGURATION CHANGES")
    print("=====================================================================")
    
    base_x_raw = X_df.iloc[0].values
    changes_found = False
    
    for i, col in enumerate(X_cols):
        b_val = base_x_raw[i]
        o_val = discretized_x_raw[i]
        
        if abs(b_val - o_val) > 0.01:
            changes_found = True
            if col.endswith("_amount"):
                print(f"{col:<30} | Baseline: {int(b_val):<4} -> Optimized: {int(o_val)}")
            else:
                print(f"{col:<30} | Baseline: {b_val:<7.2f} -> Optimized: {o_val:.2f}")
                
    if not changes_found:
        print("No changes required. The baseline scenario already hits your targets.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="synthetic", help="Dataset source")
    args = parser.parse_args()
    main(args.source)