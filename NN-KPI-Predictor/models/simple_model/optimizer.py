import argparse
import json
import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
import pandas as pd
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from model import (
    PharmacySurrogate,
    NON_FEATURE_COLS,
    CONVERGENCE_FLAGS,
    inverse_transform_targets,
    inverse_transform_targets_torch,
)
from helpers.simulator import ScenarioSimulator

def evaluate_scenario(scenario_data, full_model, process_details, num_reps=15):
    simulator = ScenarioSimulator(scenario_data, full_model, process_details, seed=42)
    result = simulator.run_scenario(replications=num_reps, until = 86400 * 90)
    
    return result.get("total_cost", 0.0), result.get("avg_cycle_time", 0.0), result.get("avg_wait_time", 0.0)

# ==========================================
# MAIN PIPELINE
# ==========================================
def main(SOURCE="synthetic", max_iterations=10000, max_param_change_pct=50.0, weight_cost=33.0, weight_cycle=33.0, weight_waiting=34.0, target_cost=None, target_cycle=None, target_waiting=None, model_type="simple"):
    # --- CONFIGURATION ---
    BASE_FILE = f"data/{SOURCE}/model/scenario.json"
    MODEL_FILE = f"data/{SOURCE}/model/model.json"
    DATA_FILE = f"data/{SOURCE}/sim_data_waiting_times.csv" 
    
    # STEP 0: LOAD FILES & PREP DYNAMIC TARGETS
    print("[0/4] Loading dataset to determine realistic targets...")
    df_all = pd.read_csv(DATA_FILE)
    
    # Only keep fully-converged rows for both target-setting and bounds
    df = df_all[df_all[CONVERGENCE_FLAGS].all(axis=1)].reset_index(drop=True)

    # Use user-specified target goals if provided, otherwise fallback to dataset percentiles
    if target_cost is not None and float(target_cost) > 0:
        TARGET_COST = float(target_cost)
    else:
        TARGET_COST = float(df['kpi_total_cost'].quantile(0.05))

    if target_cycle is not None and float(target_cycle) > 0:
        TARGET_DURATION = float(target_cycle)
    else:
        TARGET_DURATION = float(df['kpi_cycle_time'].quantile(0.010))

    if target_waiting is not None and float(target_waiting) > 0:
        TARGET_WAITING_TIME = float(target_waiting)
    else:
        TARGET_WAITING_TIME = float(df['kpi_waiting_time'].quantile(0.10))
    
    print(f"--- TARGET GOALS ---")
    print(f"Goal Cost:         ${TARGET_COST:.2f}")
    print(f"Goal Duration:     {TARGET_DURATION:.1f} seconds")
    print(f"Goal Waiting Time: {TARGET_WAITING_TIME:.1f} seconds")
    print(f"Max Allowed Param Variation: ±{max_param_change_pct:.1f}%")
    print(f"KPI Priorities -> Cost: {weight_cost:.1f}%, Cycle Time: {weight_cycle:.1f}%, Wait Time: {weight_waiting:.1f}%\n")

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

    # STEP 1: EVALUATE BASELINE SCENARIO
    print("[1/4] Running Ground-Truth SimPy Evaluation on BASELINE...")
    base_true_cost, base_true_cycle_time, base_true_wait_time = evaluate_scenario(
        baseline_scenario, full_model, process_details
    )

    # STEP 2: NEURAL NETWORK OPTIMIZATION
    print("\n[2/4] Running Targeted Neural Network Optimizer...")

    # Feature Alignment
    X_df = df.drop(columns=NON_FEATURE_COLS)
    X_cols = X_df.columns.tolist()

    model_dirs = {
        "simple": "simple_model",
        "complex": "complex_model",
        "deep": "deep_network",
        "lr": "simple_model"
    }
    target_dir = model_dirs.get(model_type, "simple_model")
    scaler_path = f"models/{target_dir}/output/{SOURCE}/x_scaler.pkl"
    yscaler_path = f"models/{target_dir}/output/{SOURCE}/y_scaler.pkl"
    model_weight_path = f"models/{target_dir}/output/{SOURCE}/surrogate_model.pth"

    if os.path.exists(scaler_path) and os.path.exists(model_weight_path):
        x_scaler = joblib.load(scaler_path)
        y_scaler = joblib.load(yscaler_path)
    else:
        scaler_path = f"models/simple_model/output/{SOURCE}/x_scaler.pkl"
        x_scaler = joblib.load(scaler_path)
        y_scaler = joblib.load(f"models/simple_model/output/{SOURCE}/y_scaler.pkl")
        model_weight_path = f"models/simple_model/output/{SOURCE}/surrogate_model.pth"

    if getattr(x_scaler, 'n_features_in_', None) != X_df.shape[1]:
        raise ValueError(
            f"Feature count mismatch: current dataset has {X_df.shape[1]} features, "
            f"but the loaded surrogate model scaler expects {getattr(x_scaler, 'n_features_in_', 'unknown')} features as input. "
            f"Please run Step 3 (Train Surrogate Model) to train the model on your current configuration before running optimization."
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    if model_type == "complex" and os.path.exists(f"models/complex_model/output/{SOURCE}/surrogate_model.pth"):
        from models.complex_model.hypertuned_model import PharmacySurrogate as ComplexPharmacySurrogate, DROPOUT_RATE
        model = ComplexPharmacySurrogate(x_scaler.n_features_in_, DROPOUT_RATE=DROPOUT_RATE).to(device)
    elif model_type == "deep" and os.path.exists(f"models/deep_network/output/{SOURCE}/surrogate_model.pth"):
        from models.deep_network.hypertuned_model import DeepPharmacySurrogate
        model = DeepPharmacySurrogate(input_size=x_scaler.n_features_in_, hidden_dim=512, num_blocks=6, dropout_rate=0.381).to(device)
    else:
        model = PharmacySurrogate(x_scaler.n_features_in_).to(device)

    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
    model.eval()
    for param in model.parameters(): param.requires_grad = False
    
    raw_min_array = X_df.min().values.reshape(1, -1)
    raw_max_array = X_df.max().values.reshape(1, -1)
    
    # Apply user-defined percentage variation bounds relative to baseline parameters
    base_x_raw_arr = X_df.iloc[0].values
    delta_vec = np.maximum(np.abs(base_x_raw_arr) * (float(max_param_change_pct) / 100.0), 1.0)
    param_min_vec = np.maximum(raw_min_array[0], base_x_raw_arr - delta_vec)
    param_max_vec = np.minimum(raw_max_array[0], base_x_raw_arr + delta_vec)
    
    for idx in range(len(param_min_vec)):
        if param_min_vec[idx] > param_max_vec[idx]:
            param_min_vec[idx] = param_max_vec[idx]

    min_scaled_bounds = torch.tensor(x_scaler.transform(param_min_vec.reshape(1, -1)), dtype=torch.float32, device=device)
    max_scaled_bounds = torch.tensor(x_scaler.transform(param_max_vec.reshape(1, -1)), dtype=torch.float32, device=device)
    
    # --- MULTI-START INITIALIZATION ---
    NUM_STARTS = 500
    
    # Spawn 499 random starts all over the parameter space within percentage bounds
    rand_starts = min_scaled_bounds + torch.rand((NUM_STARTS - 1, len(X_cols)), device=device) * (max_scaled_bounds - min_scaled_bounds)
    
    # Add baseline as the 500th start
    base_x_raw = X_df.iloc[0].values.reshape(1, -1)
    base_tensor = torch.tensor(x_scaler.transform(base_x_raw), dtype=torch.float32, device=device)
    
    batch_x_init = torch.cat([rand_starts, base_tensor], dim=0)
    x_optim = nn.Parameter(batch_x_init, requires_grad=True)

    res_amount_indices = [i for i, col in enumerate(X_cols) if col.startswith("res_") and col.endswith("_amount")]
    x_mean_tensor = torch.tensor(x_scaler.mean_, dtype=torch.float32, device=device)
    x_scale_tensor = torch.tensor(x_scaler.scale_, dtype=torch.float32, device=device)

    y_mean_tensor = torch.tensor(y_scaler.mean_, dtype=torch.float32, device=device)
    y_scale_tensor = torch.tensor(y_scaler.scale_, dtype=torch.float32, device=device)

    optimizer = optim.Adam([x_optim], lr=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations, eta_min=1e-4)
    best_global_loss = float('inf')
    best_x_optimal = None

    # Calculate normalized KPI priority weights
    total_w = max(1e-6, float(weight_cost) + float(weight_cycle) + float(weight_waiting))
    w_c = float(weight_cost) / total_w
    w_cy = float(weight_cycle) / total_w
    w_w = float(weight_waiting) / total_w

    for epoch in range(max_iterations):
        optimizer.zero_grad()
        
        predictions = model(x_optim) 
        
        raw_preds = inverse_transform_targets_torch(predictions, y_mean_tensor, y_scale_tensor)
        
        pred_total_cost  = raw_preds[:, 0]
        pred_cycle_time  = raw_preds[:, 1]
        pred_waiting_time = raw_preds[:, 2]
        
        loss_cost    = ((pred_total_cost - TARGET_COST) / TARGET_COST) ** 2
        loss_cycle   = ((pred_cycle_time - TARGET_DURATION) / TARGET_DURATION) ** 2
        loss_waiting = ((pred_waiting_time - TARGET_WAITING_TIME) / TARGET_WAITING_TIME) ** 2
        kpi_loss = (w_c * loss_cost) + (w_cy * loss_cycle) + (w_w * loss_waiting)
        
        # GUARDRAILS
        x_raw_differentiable = (x_optim * x_scale_tensor) + x_mean_tensor
        res_amounts = x_raw_differentiable[:, res_amount_indices]
        fractional_penalty = torch.sum(torch.sin(np.pi * res_amounts) ** 2, dim=1)
        
        half_iters = max(1.0, max_iterations / 2.0)
        integer_weight = max(0.0, (epoch - half_iters) / half_iters) * 1000.0 
        
        loss = (10000.0 * kpi_loss) + (integer_weight * fractional_penalty)
        min_loss_in_batch, min_idx = torch.min(loss), torch.argmin(loss)
        if min_loss_in_batch.item() < best_global_loss:
            best_global_loss = min_loss_in_batch.item()
            best_x_optimal = x_optim[min_idx].detach().clone()
        batch_loss = loss.sum()
        batch_loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
            x_optim.clamp_(min_scaled_bounds, max_scaled_bounds)

        if (epoch + 1) % 1000 == 0 or epoch == max_iterations - 1:
            print(f"Optimizing Iteration [{epoch+1}/{max_iterations}] - Loss: {best_global_loss:.4f}", flush=True)

    if best_x_optimal is None:
        best_x_optimal = x_optim[0].detach().clone()
    optimized_x_raw = x_scaler.inverse_transform(best_x_optimal.cpu().numpy().reshape(1, -1))[0]

    # STEP 3: INJECT OPTIMIZED PARAMETERS WITH PHYSICAL CONSTRAINTS
    print("[3/4] Injecting Optimized Parameters and Formatting...")
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
                if r["name"] == res_name:
                    r["totalAmount"] = str(val)
                    
        elif col.startswith("res_") and col.endswith("_cost"):
            res_name = col.replace("res_", "").replace("_cost", "").replace("_", " ")
            val = max(10.00, round(val, 2))
            for r in opt_scenario.get("resources", []):
                if r["name"] == res_name:
                    r["costPerHour"] = str(val)
                    
        elif col.startswith("el_") and col.endswith("_duration"):
            el_id = col.replace("el_", "").replace("_duration", "")
            val = max(1.0, round(val, 2))
            for el in opt_scenario.get("elements", []):
                if el["elementId"] == el_id:
                    el["durationDistribution"]["mean"] = str(val)
                    
                    if "standardDeviation" in el["durationDistribution"]:
                        el["durationDistribution"]["standardDeviation"] = "0.0"
        
        discretized_x_raw[i] = val

    discretized_x_scaled = x_scaler.transform(discretized_x_raw.reshape(1, -1))
    discretized_tensor = torch.tensor(discretized_x_scaled, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        final_pred_scaled = model(discretized_tensor).cpu().numpy()

    final_pred = inverse_transform_targets(final_pred_scaled, y_scaler)[0]

    # STEP 4: EVALUATE OPTIMIZED SCENARIO
    print("[4/4] Running Ground-Truth SimPy Evaluation on OPTIMIZED...")
    opt_true_cost, opt_true_duration, opt_true_completed = evaluate_scenario(
        opt_scenario, full_model, process_details
    )

    print("\n=====================================================================")
    print("                    VALIDATION & ROI REPORT")
    print("=====================================================================")
    print(f"                | COST (Total)         | CYCLE TIME (Avg)     | WAITING TIME (Avg)")
    print("---------------------------------------------------------------------")
    print(f"TARGET GOAL     | ${TARGET_COST:<19.2f} | {TARGET_DURATION:<19.1f}s | {TARGET_WAITING_TIME:.1f}s")
    print(f"BASELINE (True) | ${base_true_cost:<19.2f} | {base_true_cycle_time:<19.1f}s | {base_true_wait_time:.1f}s")
    print(f"NN PREDICTED    | ${final_pred[0]:<19.2f} | {final_pred[1]:<19.1f}s | {final_pred[2]:.1f}s")
    print(f"OPTIMIZED (True)| ${opt_true_cost:<19.2f} | {opt_true_duration:<19.1f}s | {opt_true_completed:.1f}s")
    print("---------------------------------------------------------------------")

    start_cost_diff = round(base_true_cost, 2) - TARGET_COST
    start_cycle_diff = base_true_cycle_time - TARGET_DURATION
    start_wait_diff = base_true_wait_time - TARGET_WAITING_TIME
    end_cost_diff = round(opt_true_cost, 2) - TARGET_COST
    end_cycle_diff = opt_true_duration - TARGET_DURATION
    end_wait_diff = opt_true_completed - TARGET_WAITING_TIME

    print(f"STARTING DELTA      | {('+' if start_cost_diff > 0 else '')}${start_cost_diff:<19.2f} | {('+' if start_cycle_diff > 0 else '')}{start_cycle_diff:.1f}s | {('+' if start_wait_diff > 0 else '')}{start_wait_diff:.1f}s")
    print("=====================================================================")
    print(f"FINISHING DELTA      | {('+' if end_cost_diff > 0 else '')}${end_cost_diff:<19.2f} | {('+' if end_cycle_diff > 0 else '')}{end_cycle_diff:.1f}s | {('+' if end_wait_diff > 0 else '')}{end_wait_diff:.1f}s")
    print("=====================================================================")
    print("\n=====================================================================")
    print("                 RECOMMENDED CONFIGURATION CHANGES")
    print("=====================================================================")
    
    base_x_raw_arr = X_df.iloc[0].values
    changes_found = False
    recommendations_list = []
    
    for i, col in enumerate(X_cols):
        b_val = float(base_x_raw_arr[i])
        o_val = float(discretized_x_raw[i])
        
        if abs(b_val - o_val) > 0.01:
            changes_found = True
            rec_item = {
                "parameter": col,
                "baseline": int(b_val) if col.endswith("_amount") else round(b_val, 2),
                "optimized": int(o_val) if col.endswith("_amount") else round(o_val, 2)
            }
            recommendations_list.append(rec_item)
            if col.endswith("_amount"):
                print(f"{col:<30} | Baseline: {int(b_val):<4} -> Optimized: {int(o_val)}")
            else:
                print(f"{col:<30} | Baseline: {b_val:<7.2f} -> Optimized: {o_val:.2f}")
                
    if not changes_found:
        print("No changes required. The baseline scenario already hits your targets.")

    # SAVE JSON RESULTS FOR WEB INTERFACE
    opt_results = {
        "settings": {
            "max_param_change_pct": float(max_param_change_pct),
            "weight_cost": float(weight_cost),
            "weight_cycle": float(weight_cycle),
            "weight_waiting": float(weight_waiting)
        },
        "target_goal": {
            "cost": round(TARGET_COST, 2),
            "cycle_time": round(TARGET_DURATION, 1),
            "waiting_time": round(TARGET_WAITING_TIME, 1)
        },
        "baseline": {
            "cost": float(round(base_true_cost, 2)),
            "cycle_time": float(round(base_true_cycle_time, 1)),
            "waiting_time": float(round(base_true_wait_time, 1))
        },
        "nn_predicted": {
            "cost": float(round(final_pred[0], 2)),
            "cycle_time": float(round(final_pred[1], 1)),
            "waiting_time": float(round(final_pred[2], 1))
        },
        "optimized": {
            "cost": float(round(opt_true_cost, 2)),
            "cycle_time": float(round(opt_true_duration, 1)),
            "waiting_time": float(round(opt_true_completed, 1))
        },
        "deltas": {
            "start_cost": float(round(start_cost_diff, 2)),
            "start_cycle": float(round(start_cycle_diff, 1)),
            "start_wait": float(round(start_wait_diff, 1)),
            "end_cost": float(round(end_cost_diff, 2)),
            "end_cycle": float(round(end_cycle_diff, 1)),
            "end_wait": float(round(end_wait_diff, 1))
        },
        "recommendations": recommendations_list
    }

    output_dir = Path("models/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "optimizer_results.json", "w") as f:
        json.dump(opt_results, f, indent=4)
    print("Exported optimizer results to models/output/optimizer_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        nargs="?",
        default="synthetic",
        help="Dataset source (default: synthetic)"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=10000,
        help="Max optimizer iterations (default: 10000)"
    )
    parser.add_argument(
        "--max_param_change_pct",
        type=float,
        default=50.0,
        help="Max allowed parameter variation percentage (default: 50.0)"
    )
    parser.add_argument(
        "--weight_cost",
        type=float,
        default=33.0,
        help="Cost KPI priority weight percentage (default: 33.0)"
    )
    parser.add_argument(
        "--weight_cycle",
        type=float,
        default=33.0,
        help="Cycle time KPI priority weight percentage (default: 33.0)"
    )
    parser.add_argument(
        "--weight_waiting",
        type=float,
        default=34.0,
        help="Waiting time KPI priority weight percentage (default: 34.0)"
    )
    parser.add_argument(
        "--target_cost",
        type=float,
        default=None,
        help="Target Cost Goal"
    )
    parser.add_argument(
        "--target_cycle",
        type=float,
        default=None,
        help="Target Cycle Time Goal"
    )
    parser.add_argument(
        "--target_waiting",
        type=float,
        default=None,
        help="Target Waiting Time Goal"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="simple",
        help="Model architecture type (default: simple)"
    )

    args = parser.parse_args()
    main(
        args.source,
        max_iterations=args.max_iterations,
        max_param_change_pct=args.max_param_change_pct,
        weight_cost=args.weight_cost,
        weight_cycle=args.weight_cycle,
        weight_waiting=args.weight_waiting,
        target_cost=args.target_cost,
        target_cycle=args.target_cycle,
        target_waiting=args.target_waiting,
        model_type=args.model_type
    )