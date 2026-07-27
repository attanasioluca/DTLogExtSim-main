import os
import sys
import copy
import json
import optuna
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path: sys.path.append(parent_dir)

from helpers.simulator import ScenarioSimulator

# --- GLOBAL PREPARATION (Loads data & cleans graph exactly ONCE) ---
BASE_FILE = "data/model/scenario.json"
MODEL_FILE = "data/model/model.json"
DATA_FILE = "data/2_100reps_data.csv"

with open(BASE_FILE, 'r') as f: base_json = json.load(f)
with open(MODEL_FILE, 'r') as f: full_model = json.load(f)

process_details = {}
for p_id, p_data in full_model.get("process_elements", {}).items():
    if p_data.get("node_details"): 
        process_details = copy.deepcopy(p_data)
        break

valid_node_ids = set(process_details["node_details"].keys())
for node_id, node in process_details["node_details"].items():
    if "previous" in node and node["previous"]:
        node["previous"] = [p for p in node["previous"] if p in valid_node_ids]

df = pd.read_csv(DATA_FILE)
X_cols = [col for col in df.columns if col.startswith(("arrival_", "res_", "el_"))]
feature_min = df[X_cols].min().to_dict()
feature_max = df[X_cols].max().to_dict()


def evaluate_scenario(scenario_data, full_model, process_details, num_reps=100):
    simulator = ScenarioSimulator(scenario_data, full_model, process_details, seed=42)
    result = simulator.run_scenario(replications=num_reps)
    
    t_cost     = result.get("kpi_total_cost", result.get("total_cost", 0.0))
    std_t_cost = result.get("std_total_cost", result.get("std_cost", 0.0))
    comp       = result.get("kpi_completed", result.get("completed", 1.0))
    if comp == 0: comp = 1.0
    
    avg_cost = t_cost / comp
    std_cost = std_t_cost / comp
    avg_dur  = result.get("kpi_avg_duration", result.get("avg_duration", 0.0))
    std_dur  = result.get("std_duration", 0.0)
    
    return avg_cost, std_cost, avg_dur, std_dur


def objective(trial):
    # 1. DYNAMICALLY SUGGEST PARAMETERS FROM CSV BOUNDS
    suggested_params = {}
    for col in X_cols:
        c_min, c_max = feature_min[col], feature_max[col]
        
        if col == "arrival_rate_mean":
            suggested_params[col] = trial.suggest_float(col, max(120.0, c_min), c_max)
            
        elif col.startswith("res_") and col.endswith("_amount"):
            suggested_params[col] = trial.suggest_int(col, 1, 15)
            
        elif col.startswith("res_") and col.endswith("_cost"):
            suggested_params[col] = trial.suggest_float(col, max(10.0, c_min), c_max)
            
        elif col.startswith("el_") and col.endswith("_duration"):
            suggested_params[col] = trial.suggest_float(col, max(1.0, c_min), c_max)

    # 2. RECONSTRUCT & INJECT SCENARIO JSON PAYLOAD
    scenario_data = copy.deepcopy(base_json["0"])
    
    for col, val in suggested_params.items():
        if col == "arrival_rate_mean":
            scenario_data["arrivalRateDistribution"]["mean"] = str(round(val, 2))
            
        elif col.startswith("res_") and col.endswith("_amount"):
            res_name = col.replace("res_", "").replace("_amount", "").replace("_", " ")
            for r in scenario_data.get("resources", []):
                if r["name"] == res_name: r["totalAmount"] = str(int(val))
                
        elif col.startswith("res_") and col.endswith("_cost"):
            res_name = col.replace("res_", "").replace("_cost", "").replace("_", " ")
            for r in scenario_data.get("resources", []):
                if r["name"] == res_name: r["costPerHour"] = str(round(val, 2))
                
        elif col.startswith("el_") and col.endswith("_duration"):
            el_id = col.replace("el_", "").replace("_duration", "")
            for el in scenario_data.get("elements", []):
                if el["elementId"] == el_id:
                    orig_mean = float(el["durationDistribution"]["mean"])
                    orig_std = float(el.get("durationDistribution", {}).get("standardDeviation", 0))
                    el["durationDistribution"]["mean"] = str(round(val, 2))
                    if orig_mean > 0 and orig_std > 0:
                        el["durationDistribution"]["standardDeviation"] = str(round(val * (orig_std / orig_mean), 2))

    # 3. RUN TRUE SIMULATION (100 Reps for Variance Stability)
    avg_cost, std_cost, avg_dur, std_dur = evaluate_scenario(
        scenario_data, full_model, process_details, num_reps=100
    )
    
    # 4. CALCULATE MULTI-OBJECTIVE PENALTY SCORES
    TARGET_COST = 20.50
    TARGET_DURATION = 10500.0
    RISK_WEIGHT = 0.5  # Lambda: Heavily penalizes solutions with unstable standard deviations
    
    cost_score = abs(avg_cost - TARGET_COST) + (RISK_WEIGHT * std_cost)
    dur_score  = abs(avg_dur - TARGET_DURATION) + (RISK_WEIGHT * std_dur)
    
    return cost_score, dur_score


def main():
    print("Starting Direct SimPy Multi-Objective Bayesian Optimization...")
    print("Sampler: TPE (Tree-structured Parzen Estimator)")
    
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=42)
    )
    
    study.optimize(objective, n_trials=200) 
    
    print("\n=====================================================================")
    print("                 PARETO FRONT OPTIMIZATION RESULTS")
    print("=====================================================================")
    print("Optuna discovered the following optimal trade-offs (Cost vs. Duration):")
    
    for i, trial in enumerate(study.best_trials):
        print(f"\n[Pareto Solution #{i+1}]")
        print(f"  Score: Cost Penalty={trial.values[0]:.2f} | Duration Penalty={trial.values[1]:.2f}")
        
        # Display the most impactful resource configurations for this solution
        important_params = {k: v for k, v in trial.params.items() if "_amount" in k}
        print(f"  Resource Allocations: {important_params}")

if __name__ == "__main__":
    main()