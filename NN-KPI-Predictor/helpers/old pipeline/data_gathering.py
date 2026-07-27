import os
import json
import copy
import math
import traceback
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import t as t_dist
import time
from simulator import ScenarioSimulator

# ==========================================
# GLOBAL VARIABLES FOR WORKERS
# ==========================================
_BASE_JSON = None
_FULL_MODEL = None
_PROCESS_DETAILS = None

def init_worker(base_file, model_file):
    """Runs ONCE per CPU core when the pool starts."""
    global _BASE_JSON, _FULL_MODEL, _PROCESS_DETAILS

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
    # Prevent the simulator from spawning millions of instances
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
    return int(rng.integers(low, high + 1))  # +1: rng.integers() upper bound is exclusive

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

WORKERS_NUM = 22
MIN_REPS = 5
MAX_REPS = 15
TARGET_REL_ERROR = 0.02
SIMULATION_TIME = 86400 * 90
ABS_TOL_WAIT_SECONDS = 120
ABS_TOL_DURATION_SECONDS = 120
ABS_TOL_COST = 0.5

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

        for rep in range(MAX_REPS):
            simulator.seed = rep_seeds[rep]
            print(f"[Worker] Scenario {scenario_id} is running rep {rep}/{MAX_REPS}...", flush=True)
            rep_result = simulator.run_replication(until=SIMULATION_TIME)
            results.append(rep_result)

            if len(results) >= MIN_REPS:
                costs = [r["total_cost"] for r in results]
                durations = [r["duration"] for r in results]
                converged_cost, _ = precision_reached(costs, TARGET_REL_ERROR, ABS_TOL_COST)
                converged_duration, _ = precision_reached(durations, TARGET_REL_ERROR, ABS_TOL_DURATION_SECONDS)

                if converged_cost and converged_duration:
                    converged = True
                    break

        # --- AGGREGATE FINAL KPIs ---
        total_costs = [r["total_cost"] for r in results]
        durations = [r["duration"] for r in results]

        features["kpi_total_cost"] = float(np.mean(total_costs))
        features["kpi_std_total_cost"] = float(np.std(total_costs, ddof=1)) if len(total_costs) > 1 else 0.0

        features["kpi_cycle_time"] = float(np.mean(durations))
        features["kpi_std_cycle_time"] = float(np.std(durations, ddof=1)) if len(durations) > 1 else 0.0

        # Diagnostics: lets you audit convergence behavior after the fact
        # instead of guessing from aggregate stats alone.
        features["n_reps_used"] = len(results)
        features["converged"] = converged
        features["converged_cost"] = converged_cost
        features["converged_duration"] = converged_duration

        return features

    except Exception:
        print(f"Error in scenario {scenario_id}:\n{traceback.format_exc()}", flush=True)
        return None

# ==========================================
# 4. PARALLEL PIPELINE & BATCH WRITING
# ==========================================

def main():
    BASE_FILE = "data/BIMP/model/scenario.json"
    MODEL_FILE = "data/BIMP/model/model.json"
    OUTPUT_FILE = "data/BIMP/no_wait_90_day_data.csv"
    START_ID = 0
    NUM_SCENARIOS = 10_000
    BATCH_SIZE = 1
    completed_ids = set()
    header_written = False
    if os.path.exists(OUTPUT_FILE):
        try:
            existing = pd.read_csv(OUTPUT_FILE, usecols=["scenario_id"])
            completed_ids = set(existing["scenario_id"].astype(int).tolist())
            header_written = True
            print(f"Resuming: {len(completed_ids)} scenarios already in {OUTPUT_FILE}, skipping them.", flush=True)
        except Exception as e:
            print(f"Could not read existing {OUTPUT_FILE} ({e}); starting fresh.", flush=True)
            os.remove(OUTPUT_FILE)

    scenario_ids = [i for i in range(START_ID, START_ID + NUM_SCENARIOS) if i not in completed_ids]
    if not scenario_ids:
        print("All scenarios already completed. Nothing to do.", flush=True)
        return

    start_time = time.time()
    results_batch = []
    total_processed = 0

    with ProcessPoolExecutor(max_workers=WORKERS_NUM, initializer=init_worker, initargs=(BASE_FILE, MODEL_FILE)) as executor:
        futures = [executor.submit(worker_task, i) for i in scenario_ids]

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results_batch.append(result)

            total_processed += 1
            if total_processed % 5 == 0:
                print(f"Scenarios completed this run: {total_processed}/{len(scenario_ids)}", flush=True)

            if len(results_batch) >= BATCH_SIZE:
                df = pd.DataFrame(results_batch)
                df.to_csv(OUTPUT_FILE, mode='a', index=False, header=not header_written)
                header_written = True
                results_batch = []
                elapsed = time.time() - start_time
                print(f"--- SAVED BATCH --- | {total_processed:,}/{len(scenario_ids):,} this run | Elapsed: {elapsed:.2f}s", flush=True)

    if results_batch:
        df = pd.DataFrame(results_batch)
        df.to_csv(OUTPUT_FILE, mode='a', index=False, header=not header_written)

    total_time = time.time() - start_time
    print(f"Pipeline complete! {len(scenario_ids):,} scenarios processed in {total_time:.2f} seconds.", flush=True)

if __name__ == "__main__":
    main()
