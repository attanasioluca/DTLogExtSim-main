import argparse
import json
import copy
import sys
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models.lightgbm_model.lgbm_model import (
    NON_FEATURE_COLS,
    CONVERGENCE_FLAGS,
    inverse_transform_targets
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
    num_starts=5000,
    maxiter=1000
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

    # STEP 2: SURROGATE MODEL OPTIMIZATION
    print("[2/4] Running LightGBM Surrogate Model Optimizer...")
    x_scaler = joblib.load(f'models/lightgbm_model/output/{SOURCE}/lgbm_x_scaler.pkl')
    y_scaler = joblib.load(f'models/lightgbm_model/output/{SOURCE}/lgbm_y_scaler.pkl')
    model = joblib.load(f'models/lightgbm_model/output/{SOURCE}/lgbm_model.pkl')

    X_df = df.drop(columns=NON_FEATURE_COLS)
    X_cols = X_df.columns.tolist()
    
    raw_min_array = X_df.min().values.reshape(1, -1)
    raw_max_array = X_df.max().values.reshape(1, -1)
    
    min_scaled_bounds = x_scaler.transform(raw_min_array)[0]
    max_scaled_bounds = x_scaler.transform(raw_max_array)[0]
    
    res_amount_indices = [i for i, col in enumerate(X_cols) if col.startswith("res_") and col.endswith("_amount")]

    def loss_func(x_scaled):
        x_scaled_2d = x_scaled.reshape(1, -1)
        preds_scaled = model.predict(x_scaled_2d)
        preds_real = inverse_transform_targets(preds_scaled, y_scaler)[0]
        
        pred_cost, pred_dur, pred_wait = preds_real[0], preds_real[1], preds_real[2]
        
        loss_cost = ((pred_cost - TARGET_COST) / TARGET_COST) ** 2
        loss_dur  = ((pred_dur - TARGET_DURATION) / TARGET_DURATION) ** 2
        loss_wait = ((pred_wait - TARGET_WAIT_TIME) / TARGET_WAIT_TIME) ** 2
        
        kpi_loss = loss_cost + loss_dur + loss_wait
        
        x_raw = x_scaler.inverse_transform(x_scaled_2d)[0]
        res_amounts = x_raw[res_amount_indices]
        fractional_penalty = np.sum(np.sin(np.pi * res_amounts) ** 2)
        
        z_scores = np.abs(x_scaled)
        z_score_penalty = np.sum(np.maximum(0.0, z_scores - MAX_Z_SCORE) ** 2)
        
        return (10000.0 * kpi_loss) + (1000.0 * fractional_penalty) + (0.5 * z_score_penalty)

    NUM_STARTS = num_starts
    rand_starts = min_scaled_bounds + np.random.rand(NUM_STARTS - 1, len(X_cols)) * (max_scaled_bounds - min_scaled_bounds)
    base_scaled = x_scaler.transform(X_df.iloc[0].values.reshape(1, -1))
    
    candidate_starts = np.vstack([rand_starts, base_scaled])
    
    preds_scaled_all = model.predict(candidate_starts)
    preds_real_all = inverse_transform_targets(preds_scaled_all, y_scaler)
    
    costs_all = preds_real_all[:, 0]
    durs_all  = preds_real_all[:, 1]
    waits_all = preds_real_all[:, 2]
    
    kpi_losses_all = ((costs_all - TARGET_COST) / TARGET_COST) ** 2 + \
                     ((durs_all - TARGET_DURATION) / TARGET_DURATION) ** 2 + \
                     ((waits_all - TARGET_WAIT_TIME) / TARGET_WAIT_TIME) ** 2
                     
    best_candidate_idx = np.argmin(kpi_losses_all)
    initial_x = candidate_starts[best_candidate_idx]
    
    bounds = list(zip(min_scaled_bounds, max_scaled_bounds))
    opt_res = minimize(
        loss_func,
        initial_x,
        method="Powell",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-5}
    )
    
    best_x_optimal = opt_res.x
    optimized_x_raw = x_scaler.inverse_transform(best_x_optimal.reshape(1, -1))[0]

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

    discretized_scaled = x_scaler.transform(discretized_x_raw.reshape(1, -1))
    final_pred_scaled = model.predict(discretized_scaled)
    final_pred = inverse_transform_targets(final_pred_scaled, y_scaler)[0]

    lgbm_pred_avg_cost = final_pred[0]
    lgbm_pred_dur_mean = final_pred[1]
    lgbm_pred_wait_mean = final_pred[2]

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
    print(f"LGBM PREDICTED  | ${lgbm_pred_avg_cost:<19.2f} | {lgbm_pred_dur_mean:<19.1f}s | {lgbm_pred_wait_mean:.1f}s")
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
            "cost": float(lgbm_pred_avg_cost),
            "cycle_time": float(lgbm_pred_dur_mean),
            "waiting_time": float(lgbm_pred_wait_mean)
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
    parser = argparse.ArgumentParser(description="Run LightGBM Inverse Optimizer")
    parser.add_argument("source", nargs="?", default="synthetic", help="Dataset source (default: synthetic)")
    parser.add_argument("--num_reps", type=int, default=50, help="SimPy simulation replications (default: 50)")
    parser.add_argument("--cost_pct", type=float, default=-10.0, help="Target cost %% change relative to baseline (+10 increases, -10 decreases, default: -10.0)")
    parser.add_argument("--cycle_pct", type=float, default=-10.0, help="Target cycle time %% change relative to baseline (+10 increases, -10 decreases, default: -10.0)")
    parser.add_argument("--wait_pct", type=float, default=-10.0, help="Target waiting time %% change relative to baseline (+10 increases, -10 decreases, default: -10.0)")
    parser.add_argument("--max_z_score", type=float, default=3.0, help="Max Z-score bound penalty threshold (default: 3.0)")
    parser.add_argument("--num_starts", type=int, default=5000, help="Multi-start candidate initializations (default: 5000)")
    parser.add_argument("--maxiter", type=int, default=1000, help="Max optimization iterations (default: 1000)")
    args = parser.parse_args()
    
    main(
        SOURCE=args.source,
        num_reps=args.num_reps,
        cost_pct=args.cost_pct,
        cycle_pct=args.cycle_pct,
        wait_pct=args.wait_pct,
        max_z_score=args.max_z_score,
        num_starts=args.num_starts,
        maxiter=args.maxiter
    )
