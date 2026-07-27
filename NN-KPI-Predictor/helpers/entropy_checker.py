import os
import json
import numpy as np
import math
import time
from helpers.simulator import ScenarioSimulator

def calculate_optimal_replications(results, target_error_percentage=0.05, z_score=1.96):
    """
    Calculates the optimal number of replications using the relative error method.
    """
    costs = [r["total_cost"] for r in results]
    durations = [r["duration"] for r in results]
    wait_times = [r["wait_time"] for r in results]
    
    # Using sample standard deviation (ddof=1)
    mean_cost = np.mean(costs)
    std_cost = np.std(costs, ddof=1) if len(costs) > 1 else 0
    
    mean_dur = np.mean(durations)
    std_dur = np.std(durations, ddof=1) if len(durations) > 1 else 0

    mean_wait = np.mean(wait_times)
    std_wait = np.std(wait_times, ddof=1) if len(wait_times) > 1 else 0
    
    # Calculate n* for Cost
    if mean_cost > 0 and std_cost > 0:
        n_cost = math.ceil((z_score * std_cost / (target_error_percentage * mean_cost)) ** 2)
    else:
        n_cost = len(results)
        
    # Calculate n* for Duration
    if mean_dur > 0 and std_dur > 0:
        n_dur = math.ceil((z_score * std_dur / (target_error_percentage * mean_dur)) ** 2)
    else:
        n_dur = len(results)

    # Calculate n* for Wait Time
    if mean_wait > 0 and std_wait > 0:
        n_wait = math.ceil((z_score * std_wait / (target_error_percentage * mean_wait)) ** 2)
    else:
        n_wait = len(results)
        
    return max(n_cost, n_dur, n_wait), n_cost, n_dur, n_wait, mean_cost, std_cost, mean_dur, std_dur, mean_wait, std_wait

def main():
    BASE_FILE = "data/synthetic/model/scenario.json"
    MODEL_FILE = "data/synthetic/model/model.json"
    
    try:
        with open(BASE_FILE, 'r') as f:
            full_scenario = json.load(f)
            base_json = full_scenario.get("0", full_scenario)
    except FileNotFoundError:
        print(f"Error: Could not find {BASE_FILE}")
        return

    try:
        with open(MODEL_FILE, 'r') as f:
            full_model = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {MODEL_FILE}")
        return
        
    process_details = {}
    for p_id, p_data in full_model.get("process_elements", {}).items():
        if p_data.get("node_details"):
            process_details = p_data
            break
            
    # Clean up any broken nodes
    valid_node_ids = set(process_details.get("node_details", {}).keys())
    for node_id, node in process_details.get("node_details", {}).items():
        if "previous" in node and node["previous"]:
            node["previous"] = [p for p in node["previous"] if p in valid_node_ids]

    # --- Setup Parameters ---
    PILOT_REPS = 1000
    TARGET_ERROR = 0.05 # 5% acceptable margin
    CONFIDENCE_Z = 1.96 # Z-score for 95% confidence interval
    SIMULATION_TIME = 86400  # Matches the 30-day run in your gathering script

    print(f"Running initial pilot batch of {PILOT_REPS} replications...")
    start_time = time.time()
    
    # Initialize your actual ScenarioSimulator
    simulator = ScenarioSimulator(
        base_json,
        full_model,
        process_details,
        seed=42
    )

    results = []
    for rep in range(PILOT_REPS):
        print(f"Running pilot replication {rep+1}/{PILOT_REPS}...", end="\r")
        # Run replication natively and append the returned KPI dictionary
        results.append(simulator.run_replication(until=SIMULATION_TIME))
    
    print("\nCalculating optimal configuration...")

    optimal_n, n_cost, n_dur, n_wait, m_cost, s_cost, m_dur, s_dur, m_wait, s_wait = calculate_optimal_replications(
        results, TARGET_ERROR, CONFIDENCE_Z
    )

    print("\n" + "="*65)
    print("SIMULATION REPLICATION CALCULATOR")
    print("="*65)
    print(f"Target Accuracy:      {100 - (TARGET_ERROR * 100):.1f}%")
    print(f"Confidence Level:     95% (Z={CONFIDENCE_Z})")
    print("-" * 65)
    print(f"Cost KPI      -> Mean: {m_cost:.2f} | Std Dev: {s_cost:.2f} | Required Reps: {n_cost}")
    print(f"Duration KPI  -> Mean: {m_dur:.2f} | Std Dev: {s_dur:.2f} | Required Reps: {n_dur}")
    print(f"Wait Time KPI -> Mean: {m_wait:.2f} | Std Dev: {s_wait:.2f} | Required Reps: {n_wait}")
    print("-" * 65)
    print(f">>> RECOMMENDED SIMULATIONS PER SCENARIO: {optimal_n} <<<")
    print("="*65)
    print(f"Calculation completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()