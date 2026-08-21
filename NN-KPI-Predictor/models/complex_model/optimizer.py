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
from pathlib import Path

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_dir not in sys.path: sys.path.append(parent_dir)

from hypertuned_model import (
    SurrogateModel,
    NON_FEATURE_COLS,
    CONVERGENCE_FLAGS,
    DROPOUT_RATE,
    load_hyperparameters,
    inverse_transform_targets,
    inverse_transform_targets_torch
)
from helpers.simulator import ScenarioSimulator

def evaluate_scenario(scenario_data, full_model, process_details, num_reps=50):
    simulator = ScenarioSimulator(scenario_data, full_model, process_details, seed=42)
    result = simulator.run_scenario(replications=num_reps, until=86400 * 90)
    
    avg_cost = result.get("total_cost", 0.0)
    avg_dur  = result.get("avg_cycle_time", 0.0)
    avg_wait = result.get("avg_wait_time", 0.0)
    
    return avg_cost, avg_dur, avg_wait

# ==========================================
# MAIN PIPELINE
# ==========================================
def main(
    SOURCE="synthetic",
    num_reps=50,
    cost_pct=-10.0,
    cycle_pct=-10.0,
    wait_pct=-10.0,
    max_z_score=3.0,
    num_starts=500,
    epochs=10000,
    lr=0.1
):
    MAX_Z_SCORE = max_z_score           
    
    BASE_FILE = f"data/{SOURCE}/model/scenario.json"
    MODEL_FILE = f"data/{SOURCE}/model/model.json"
    DATA_FILE = f"data/{SOURCE}/sim_data_waiting_times.csv" 
    
    # STEP 0: LOAD FILES & INITIALIZE DATASET
    print("[0/4] Loading scenario files and dataset...")
    df_all = pd.read_csv(DATA_FILE)
    df = df_all[df_all[CONVERGENCE_FLAGS].all(axis=1)].reset_index(drop=True)

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

    # STEP 1: EVALUATE BASELINE SCENARIO & COMPUTE TARGETS
    print("[1/4] Running Ground-Truth SimPy Evaluation on BASELINE...")
    base_true_cost, base_true_duration, base_true_wait = evaluate_scenario(
        baseline_scenario, full_model, process_details, num_reps=num_reps
    )

    TARGET_COST = base_true_cost * (1.0 + cost_pct / 100.0)
    TARGET_DURATION = base_true_duration * (1.0 + cycle_pct / 100.0)
    TARGET_WAIT_TIME = base_true_wait * (1.0 + wait_pct / 100.0)

    print(f"\n--- TARGET KPI GOALS (Relative to Baseline) ---")
    print(f"Baseline Cost:       ${base_true_cost:.2f} -> Target ({'+' if cost_pct >= 0 else ''}{cost_pct:.1f}%): ${TARGET_COST:.2f}")
    print(f"Baseline Cycle Time: {base_true_duration:.1f}s -> Target ({'+' if cycle_pct >= 0 else ''}{cycle_pct:.1f}%): {TARGET_DURATION:.1f}s")
    print(f"Baseline Wait Time:  {base_true_wait:.1f}s -> Target ({'+' if wait_pct >= 0 else ''}{wait_pct:.1f}%): {TARGET_WAIT_TIME:.1f}s\n")

    # STEP 2: NEURAL NETWORK OPTIMIZATION
    print("[2/4] Running High-Speed Neural Network Optimizer...")
    x_scaler = joblib.load(f'models/complex_model/output/{SOURCE}/x_scaler.pkl')
    y_scaler = joblib.load(f'models/complex_model/output/{SOURCE}/y_scaler.pkl')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    params = load_hyperparameters(source=SOURCE)
    model = SurrogateModel(x_scaler.n_features_in_, DROPOUT_RATE=params["dropout_rate"]).to(device)
    model.load_state_dict(torch.load(f'models/complex_model/output/{SOURCE}/surrogate_model.pth', map_location=device, weights_only=True))
    model.eval()
    for param in model.parameters(): param.requires_grad = False

    X_df = df.drop(columns=NON_FEATURE_COLS)
    X_cols = X_df.columns.tolist()
    
    raw_min_array = X_df.min().values.reshape(1, -1)
    raw_max_array = X_df.max().values.reshape(1, -1)
    
    min_scaled_bounds = torch.tensor(x_scaler.transform(raw_min_array), dtype=torch.float32, device=device)
    max_scaled_bounds = torch.tensor(x_scaler.transform(raw_max_array), dtype=torch.float32, device=device)
    
    NUM_STARTS = num_starts
    rand_starts = min_scaled_bounds + torch.rand((NUM_STARTS - 1, len(X_cols)), device=device) * (max_scaled_bounds - min_scaled_bounds)
    base_tensor = torch.tensor(x_scaler.transform(X_df.iloc[0].values.reshape(1, -1)), dtype=torch.float32, device=device)
    
    x_optim = nn.Parameter(torch.cat([rand_starts, base_tensor], dim=0), requires_grad=True)

    res_amount_indices = [i for i, col in enumerate(X_cols) if col.startswith("res_") and col.endswith("_amount")]
    x_mean_tensor = torch.tensor(x_scaler.mean_, dtype=torch.float32, device=device)
    x_scale_tensor = torch.tensor(x_scaler.scale_, dtype=torch.float32, device=device)
    y_mean_tensor = torch.tensor(y_scaler.mean_, dtype=torch.float32, device=device)
    y_scale_tensor = torch.tensor(y_scaler.scale_, dtype=torch.float32, device=device)

    optimizer = optim.Adam([x_optim], lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    best_global_loss = float('inf')
    best_x_optimal = None

    for epoch in range(epochs):
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
        
        integer_weight = max(0.0, (epoch - (epochs // 2)) / (epochs // 2)) * 1000.0 
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

    # STEP 3: INJECT OPTIMIZED PARAMETERS
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

    # STEP 4: GROUND-TRUTH SIMULATION VALIDATION
    print("[4/4] Running Ground-Truth SimPy Evaluation on OPTIMIZED...")
    opt_true_cost, opt_true_duration, opt_true_wait = evaluate_scenario(
        opt_scenario, full_model, process_details, num_reps=num_reps
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
    recommendations = []
    
    for i, col in enumerate(X_cols):
        b_val = float(base_x_raw[i])
        o_val = float(discretized_x_raw[i])
        
        if abs(b_val - o_val) > 0.01:
            changes_found = True
            if col.endswith("_amount"):
                b_val_out = int(round(b_val))
                o_val_out = int(round(o_val))
                print(f"{col:<30} | Baseline: {b_val_out:<4} -> Optimized: {o_val_out}")
            else:
                b_val_out = round(b_val, 2)
                o_val_out = round(o_val, 2)
                print(f"{col:<30} | Baseline: {b_val_out:<7.2f} -> Optimized: {o_val_out:.2f}")
            recommendations.append({
                "parameter": col,
                "baseline": b_val_out,
                "optimized": o_val_out
            })
                
    if not changes_found:
        print("No changes required. The baseline scenario already hits your targets.")

    results_out = {
        "settings": {
            "max_param_change_pct": 50.0,
            "weight_cost": 33.0,
            "weight_cycle": 33.0,
            "weight_waiting": 34.0
        },
        "target_goal": {
            "cost": float(TARGET_COST),
            "cycle_time": float(TARGET_DURATION),
            "waiting_time": float(TARGET_WAIT_TIME)
        },
        "baseline": {
            "cost": float(base_true_cost),
            "cycle_time": float(base_true_duration),
            "waiting_time": float(base_true_wait)
        },
        "nn_predicted": {
            "cost": float(nn_pred_avg_cost),
            "cycle_time": float(nn_pred_dur_mean),
            "waiting_time": float(nn_pred_wait_mean)
        },
        "optimized": {
            "cost": float(opt_true_cost),
            "cycle_time": float(opt_true_duration),
            "waiting_time": float(opt_true_wait)
        },
        "deltas": {
            "end_cost": float(opt_true_cost - base_true_cost),
            "end_cycle": float(opt_true_duration - base_true_duration),
            "end_wait": float(opt_true_wait - base_true_wait)
        },
        "recommendations": recommendations
    }

    out_file = Path("models/output/optimizer_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results_out, f, indent=4)
    print(f"\nOptimizer results exported to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Complex Model Inverse Optimizer")
    parser.add_argument("source", nargs="?", default="synthetic", help="Dataset source (default: synthetic)")
    parser.add_argument("--num_reps", type=int, default=50, help="SimPy simulation replications (default: 50)")
    parser.add_argument("--cost_pct", type=float, default=-10.0, help="Target cost %% change relative to baseline (+10 increases, -10 decreases, default: -10.0)")
    parser.add_argument("--cycle_pct", type=float, default=-10.0, help="Target cycle time %% change relative to baseline (+10 increases, -10 decreases, default: -10.0)")
    parser.add_argument("--wait_pct", type=float, default=-10.0, help="Target waiting time %% change relative to baseline (+10 increases, -10 decreases, default: -10.0)")
    parser.add_argument("--max_z_score", type=float, default=3.0, help="Max Z-score bound penalty threshold (default: 3.0)")
    parser.add_argument("--num_starts", type=int, default=500, help="Multi-start candidate initializations (default: 500)")
    parser.add_argument("--epochs", type=int, default=10000, help="Max optimization epochs (default: 10000)")
    parser.add_argument("--lr", type=float, default=0.1, help="Optimizer learning rate (default: 0.1)")
    args = parser.parse_args()

    main(
        SOURCE=args.source,
        num_reps=args.num_reps,
        cost_pct=args.cost_pct,
        cycle_pct=args.cycle_pct,
        wait_pct=args.wait_pct,
        max_z_score=args.max_z_score,
        num_starts=args.num_starts,
        epochs=args.epochs,
        lr=args.lr
    )