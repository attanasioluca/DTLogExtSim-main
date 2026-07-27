import argparse
import os
import sys
import json
import copy
import math
import traceback
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from scipy.stats import t as t_dist
import time

# Go up two levels to reach the root directory so 'helpers' resolves correctly
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from helpers.simulator import ScenarioSimulator

# ==========================================
# GLOBAL VARIABLES FOR WORKERS
# ==========================================
_BASE_JSON = None
_FULL_MODEL = None
_PROCESS_DETAILS = None
_ABS_TOL_WAIT = 120.0
_ABS_TOL_DUR = 120.0
_ABS_TOL_COST = 0.5
_MAX_WAIT_THRESHOLD = 86400 * 7  # Fallback: 7 days

def init_worker(base_file, model_file, tol_wait, tol_dur, tol_cost, max_wait):
    """Runs ONCE per CPU core when the pool starts."""
    global _BASE_JSON, _FULL_MODEL, _PROCESS_DETAILS
    global _ABS_TOL_WAIT, _ABS_TOL_DUR, _ABS_TOL_COST, _MAX_WAIT_THRESHOLD

    _ABS_TOL_WAIT = tol_wait
    _ABS_TOL_DUR = tol_dur
    _ABS_TOL_COST = tol_cost
    _MAX_WAIT_THRESHOLD = max_wait

    with open(base_file, 'r') as f:
        _BASE_JSON = json.load(f)
    with open(model_file, 'r') as f:
        _FULL_MODEL = json.load(f)

    for p_id, p_data in _FULL_MODEL.get("process_elements", {}).items():
        if p_data.get("node_details"):
            _PROCESS_DETAILS = p_data
            break

    valid_node_ids = set(_PROCESS_DETAILS["node_details"].keys())
    for node_id, node in _PROCESS_DETAILS["node_details"].items():
        if "previous" in node and node["previous"]:
            node["previous"] = [p for p in node["previous"] if p in valid_node_ids]

# ==========================================
# 1. MUTATION LOGIC
# ==========================================

def mutate_float_broad(rng, base_val, min_multiplier=0.4, max_multiplier=2.5):
    val = float(base_val)
    new_val = rng.uniform(val * min_multiplier, val * max_multiplier)
    return max(0.01, round(new_val, 2))

def mutate_arrival_rate(rng, base_val):
    val = float(base_val)
    new_val = rng.uniform(val * 0.5, val * 2.0)
    return max(60.0, round(new_val, 2))

def mutate_int_broad(rng, base_val, min_multiplier=0.4, max_multiplier=2.5, floor=1, hard_cap=None):
    val = float(base_val)
    low = max(floor, int(round(val * min_multiplier)))
    high = max(low, int(round(val * max_multiplier)))
    if hard_cap is not None:
        high = min(high, hard_cap)
        low = min(low, high)
    if high <= low:
        high = low + 1
    return int(rng.integers(low, high + 1)) 

# ==========================================
# 2. PRECISION STOPPING RULE
# ==========================================

def precision_reached(values, rel_target, abs_tol):
    n = len(values)
    if n < 2:
        return False, float("inf")
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1))
    if std_val == 0.0:
        return True, 0.0
    half_width = t_dist.ppf(0.975, n - 1) * std_val / math.sqrt(n)
    tolerance = max(abs_tol, rel_target * abs(mean_val))
    return half_width <= tolerance, half_width

# ==========================================
# 3. SIMULATION LOGIC (WORKER NODE)
# ==========================================

MIN_REPS = 5
MAX_REPS = 75
TARGET_REL_ERROR = 0.02
SIMULATION_TIME = 86400 * 90
MAX_INSTANCES = None

def worker_task(scenario_id):
    try:
        rng = np.random.default_rng(seed=scenario_id)
        variant = copy.deepcopy(_BASE_JSON)
        base = variant["0"]

        # --- MUTATE CONFIGS ---
        arr_dist = base.get("arrivalRateDistribution", {})
        if arr_dist and "mean" in arr_dist and arr_dist["mean"]:
            arr_dist["mean"] = str(mutate_arrival_rate(rng, arr_dist["mean"]))

        for res in base.get("resources", []):
            if "totalAmount" in res and res["totalAmount"]:
                res["totalAmount"] = str(mutate_int_broad(rng, res["totalAmount"]))
            if "costPerHour" in res and res["costPerHour"]:
                res["costPerHour"] = str(mutate_float_broad(rng, res["costPerHour"]))

        for el in base.get("elements", []):
            dist = el.get("durationDistribution", {})
            if dist and "mean" in dist and dist["mean"]:
                dist["mean"] = str(mutate_float_broad(rng, dist["mean"]))
            if "fixedCost" in el and el["fixedCost"]:
                el["fixedCost"] = str(mutate_float_broad(rng, el["fixedCost"]))

        # --- Extract X Features ---
        features = {
            "scenario_id": scenario_id,
            "arrival_rate_mean": float(base["arrivalRateDistribution"]["mean"]),
        }
        for res in base.get("resources", []):
            features[f"res_{res['name'].replace(' ', '_')}_amount"] = int(res["totalAmount"])
            features[f"res_{res['name'].replace(' ', '_')}_cost"] = float(res["costPerHour"])

        for el in base.get("elements", []):
            dist = el.get("durationDistribution", {})
            if dist and "mean" in dist and dist["mean"]:
                features[f"el_{el['elementId']}_duration"] = float(dist["mean"])

        simulator = ScenarioSimulator(
            base,
            _FULL_MODEL,
            _PROCESS_DETAILS,
            seed=scenario_id,
        )
        rep_seeds = np.random.SeedSequence(scenario_id).spawn(MAX_REPS)

        results = []
        converged = False
        converged_wait = converged_cost = converged_duration = False

        for rep in range(MAX_REPS):
            simulator.seed = rep_seeds[rep]
            rep_result = simulator.run_replication(until=SIMULATION_TIME, max_instances=MAX_INSTANCES)
            
            # EARLY DROPPING: If the queue immediately explodes past our dynamic threshold, 
            # drop the simulation entirely to save time. It would be dropped by the model anyway.
            if rep_result.get("wait_time", 0) > _MAX_WAIT_THRESHOLD:
                print(f"[Worker] Scenario {scenario_id} hit saturated queue abort threshold on rep {rep}. Dropping.", flush=True)
                return None
                
            results.append(rep_result)

            if len(results) >= MIN_REPS:
                wait_times = [r["wait_time"] for r in results]
                costs = [r["total_cost"] for r in results]
                durations = [r["duration"] for r in results]
                
                converged_wait, _ = precision_reached(wait_times, TARGET_REL_ERROR, _ABS_TOL_WAIT)
                converged_cost, _ = precision_reached(costs, TARGET_REL_ERROR, _ABS_TOL_COST)
                converged_duration, _ = precision_reached(durations, TARGET_REL_ERROR, _ABS_TOL_DUR)

                if converged_wait and converged_cost and converged_duration:
                    converged = True
                    break

        # --- AGGREGATE FINAL KPIs ---
        total_costs = [r["total_cost"] for r in results]
        durations = [r["duration"] for r in results]
        wait_times = [r["wait_time"] for r in results]

        features["kpi_total_cost"] = float(np.mean(total_costs))
        features["kpi_cycle_time"] = float(np.mean(durations))
        features["kpi_waiting_time"] = float(np.mean(wait_times))

        features["n_reps_used"] = len(results)
        features["converged"] = converged
        features["converged_wait"] = converged_wait
        features["converged_cost"] = converged_cost
        features["converged_duration"] = converged_duration

        return features

    except Exception:
        print(f"Error in scenario {scenario_id}:\n{traceback.format_exc()}", flush=True)
        return None

# ==========================================
# 4. PARALLEL PIPELINE & BATCH WRITING
# ==========================================

def main(SOURCE="synthetic", START_ID=0, NUM_SCENARIOS=10_000, WORKERS_NUM=22):
    BATCH_SIZE = 100
    BASE_FILE = f"data/{SOURCE}/model/scenario.json"
    MODEL_FILE = f"data/{SOURCE}/model/model.json"
    OUTPUT_FILE = f"data/{SOURCE}/sim_data_waiting_times.csv" 
    
    completed_ids = set()
    header_written = False
    
    # Dynamic tolerance defaults
    tol_wait = 120.0
    tol_dur = 120.0
    tol_cost = 0.5
    max_wait_abort = 86400 * 7  # 7 days default
    
    if os.path.exists(OUTPUT_FILE):
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            if "scenario_id" in existing.columns:
                completed_ids = set(existing["scenario_id"].astype(int).tolist())
                header_written = True
                print(f"Resuming: {len(completed_ids)} scenarios already in {OUTPUT_FILE}, skipping them.", flush=True)
            
            # Calculate dynamic tolerances and thresholds from existing successful runs
            valid_existing = existing[existing["converged"] == True] if "converged" in existing.columns else existing
            if len(valid_existing) > 10:
                tol_cost = float(valid_existing["kpi_total_cost"].median() * 0.05)
                tol_dur = float(valid_existing["kpi_cycle_time"].median() * 0.05)
                tol_wait = float(valid_existing["kpi_waiting_time"].median() * 0.05)
                
                # 5x the 95th percentile is our abort threshold for hopelessly skewed runs
                max_wait_abort = float(min(_MAX_WAIT_THRESHOLD, valid_existing["kpi_waiting_time"].quantile(0.95) * 5))
                
        except Exception as e:
            print(f"Could not read existing {OUTPUT_FILE} ({e}); starting fresh.", flush=True)
            if os.path.exists(OUTPUT_FILE):
                os.remove(OUTPUT_FILE)

    print(f"--- DYNAMIC BOUNDS CALCULATION ---")
    print(f"Cost ABS_TOL:   ${tol_cost:.2f}")
    print(f"Dur ABS_TOL:    {tol_dur:.2f}s")
    print(f"Wait ABS_TOL:   {tol_wait:.2f}s")
    print(f"Abort if wait exceeds: {max_wait_abort:.2f}s\n")

    if NUM_SCENARIOS <= 0:
        print("NUM_SCENARIOS is 0 or less. Nothing to do.", flush=True)
        return

    start_time = time.time()
    results_batch = []
    
    successful_new = 0
    current_id = START_ID
    
    init_args = (BASE_FILE, MODEL_FILE, tol_wait, tol_dur, tol_cost, max_wait_abort)

    with ProcessPoolExecutor(max_workers=WORKERS_NUM, initializer=init_worker, initargs=init_args) as executor:
        active_futures = set()
        
        # Helper function to queue up the next unprocessed ID
        def submit_next():
            nonlocal current_id
            while current_id in completed_ids:
                current_id += 1
            fut = executor.submit(worker_task, current_id)
            active_futures.add(fut)
            current_id += 1
            
        # Initially fill the pool queue (double the worker count keeps cores fed)
        for _ in range(WORKERS_NUM * 2):
            submit_next()
            
        while active_futures and successful_new < NUM_SCENARIOS:
            # Wait for at least one scenario to finish
            done, active_futures = wait(active_futures, return_when=FIRST_COMPLETED)
            
            for future in done:
                result = future.result()
                
                # Only count towards the total if the worker didn't return None
                if result is not None:
                    results_batch.append(result)
                    successful_new += 1
                    
                    if successful_new % 5 == 0:
                        print(f"Successful new scenarios this run: {successful_new}/{NUM_SCENARIOS}", flush=True)
                        
                    if len(results_batch) >= BATCH_SIZE:
                        df = pd.DataFrame(results_batch)
                        df.to_csv(OUTPUT_FILE, mode='a', index=False, header=not header_written)
                        header_written = True
                        results_batch = []
                        elapsed = time.time() - start_time
                        print(f"--- SAVED BATCH --- | {successful_new:,}/{NUM_SCENARIOS:,} this run | Elapsed: {elapsed:.2f}s", flush=True)

                if successful_new >= NUM_SCENARIOS:
                    break
            
            # If we haven't hit the target, keep feeding the pool up to its capacity limit
            # This ensures we keep trying new IDs even if many fail
            if successful_new < NUM_SCENARIOS:
                while len(active_futures) < (WORKERS_NUM * 2):
                    submit_next()

        # Cancel any remaining pending futures so the script can exit cleanly
        for fut in active_futures:
            fut.cancel()

    # Save any remaining data in the final batch
    if results_batch:
        df = pd.DataFrame(results_batch)
        df.to_csv(OUTPUT_FILE, mode='a', index=False, header=not header_written)

    total_time = time.time() - start_time
    print(f"Pipeline complete! {successful_new:,} new scenarios successfully generated in {total_time:.2f} seconds.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="synthetic", help="Dataset source directory name")
    parser.add_argument("--start_id", type=int, default=0, help="Starting scenario ID")
    parser.add_argument("--num_scenarios", type=int, default=10000, help="Total number of scenarios to target")
    parser.add_argument("--workers", type=int, default=22, help="Number of CPU cores to allocate")
    
    args = parser.parse_args()
    main(args.source, args.start_id, args.num_scenarios, args.workers)