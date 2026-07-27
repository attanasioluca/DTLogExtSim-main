import os
import sys
import json
import subprocess
import threading
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

# Path to NN-KPI-Predictor
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
NN_KPI_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'NN-KPI-Predictor'))
if not os.path.exists(NN_KPI_DIR):
    NN_KPI_DIR = '/app/NN-KPI-Predictor'

STATUS: Dict[str, Dict[str, Any]] = {
    "simulation": {"running": False, "progress": "", "error": None, "completed": False},
    "training": {"running": False, "progress": "", "error": None, "completed": False},
    "optimization": {"running": False, "progress": "", "error": None, "completed": False}
}

def get_session_dir(upload_folder):
    session_dir = os.path.join(upload_folder, "nnkpi_session")
    os.makedirs(os.path.join(session_dir, "data", "synthetic", "model"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "models", "simple_model", "output", "synthetic"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "models", "LR_model", "output", "synthetic"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "models", "complex_model", "output", "synthetic"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "models", "deep_network", "output", "synthetic"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "models", "output"), exist_ok=True)
    return session_dir

def save_uploaded_files(model_file, scenario_file, upload_folder, data_file=None):
    session_dir = get_session_dir(upload_folder)
    if model_file:
        model_dst = os.path.join(session_dir, "data", "synthetic", "model", "model.json")
        model_file.save(model_dst)
    if scenario_file:
        scenario_dst = os.path.join(session_dir, "data", "synthetic", "model", "scenario.json")
        scenario_file.save(scenario_dst)
    if data_file and data_file.filename != '':
        data_dst = os.path.join(session_dir, "data", "synthetic", "sim_data_waiting_times.csv")
        data_file.save(data_dst)
    return {"status": "success", "session_dir": session_dir}

def _run_simulation_thread(session_dir, num_scenarios, workers):
    global STATUS
    STATUS["simulation"] = {"running": True, "progress": "Starting simulation data gathering...", "error": None, "completed": False}
    try:
        data_gathering_script = os.path.join(NN_KPI_DIR, "helpers", "data_gathering.py")
        if not os.path.exists(data_gathering_script):
            raise FileNotFoundError(f"Script not found: {data_gathering_script}")

        scenario_path = os.path.join(session_dir, "data", "synthetic", "model", "scenario.json")
        model_path = os.path.join(session_dir, "data", "synthetic", "model", "model.json")
        if not os.path.exists(scenario_path) or not os.path.exists(model_path):
            raise FileNotFoundError("Please upload model.json and scenario.json in Step 1 before running simulations.")

        cmd = [
            sys.executable,
            data_gathering_script,
            "--source", "synthetic",
            "--num_scenarios", str(num_scenarios),
            "--workers", str(workers)
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = NN_KPI_DIR + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.Popen(
            cmd,
            cwd=session_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        output_lines = []
        if proc.stdout:
            for line in proc.stdout:
                output_lines.append(line)
                if "SAVED BATCH" in line or "Successful new scenarios" in line:
                    STATUS["simulation"]["progress"] = line.strip()

        proc.wait()
        if proc.returncode == 0:
            STATUS["simulation"] = {"running": False, "progress": "Simulation completed successfully!", "error": None, "completed": True}
        else:
            err_msg = "".join(output_lines[-15:]).strip() or f"Process exited with code {proc.returncode}"
            STATUS["simulation"] = {"running": False, "progress": "", "error": err_msg, "completed": False}

    except Exception as e:
        STATUS["simulation"] = {"running": False, "progress": "", "error": str(e), "completed": False}

def start_simulation(upload_folder, num_scenarios=10000, workers=8):
    session_dir = get_session_dir(upload_folder)
    if STATUS["simulation"]["running"]:
        return {"status": "busy", "message": "Simulation is already running"}
    
    t = threading.Thread(target=_run_simulation_thread, args=(session_dir, num_scenarios, workers))
    t.daemon = True
    t.start()
    return {"status": "started", "message": "Simulation data gathering started"}

def _run_training_thread(session_dir, train_num, model_type="simple"):
    global STATUS
    STATUS["training"] = {"running": True, "progress": f"Starting {model_type.upper()} model training...", "error": None, "completed": False}
    try:
        model_scripts = {
            "simple": os.path.join(NN_KPI_DIR, "models", "simple_model", "model.py"),
            "lr": os.path.join(NN_KPI_DIR, "models", "LR_model", "lr_model.py"),
            "complex": os.path.join(NN_KPI_DIR, "models", "complex_model", "hypertuned_model.py"),
            "deep": os.path.join(NN_KPI_DIR, "models", "deep_network", "hypertuned_model.py")
        }
        model_script = model_scripts.get(model_type, model_scripts["simple"])
        if not os.path.exists(model_script):
            raise FileNotFoundError(f"Model script not found: {model_script}")

        cmd = [
            sys.executable,
            model_script,
            "synthetic",
            "--train_num", str(train_num)
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = NN_KPI_DIR + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.Popen(
            cmd,
            cwd=session_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        output_lines = []
        if proc.stdout:
            for line in proc.stdout:
                output_lines.append(line)
                if "Epoch" in line or "Loss" in line or "Evaluating" in line or "Metrics" in line or "MAE" in line or "Updated" in line or "Exported" in line:
                    STATUS["training"]["progress"] = line.strip()

        proc.wait()
        if proc.returncode == 0:
            STATUS["training"] = {"running": False, "progress": f"{model_type.upper()} Model training completed!", "error": None, "completed": True, "model_type": model_type}
        else:
            err_msg = "".join(output_lines[-15:]).strip() or f"Training failed with exit code {proc.returncode}"
            STATUS["training"] = {"running": False, "progress": "", "error": err_msg, "completed": False}
    except Exception as e:
        STATUS["training"] = {"running": False, "progress": "", "error": str(e), "completed": False}


def start_training(upload_folder, train_num=4000, model_type="simple"):
    session_dir = get_session_dir(upload_folder)
    if STATUS["training"]["running"]:
        return {"status": "busy", "message": "Training is already running"}

    t = threading.Thread(target=_run_training_thread, args=(session_dir, train_num, model_type))
    t.daemon = True
    t.start()
    return {"status": "started", "message": f"{model_type.upper()} training started"}

def get_training_results(upload_folder):
    session_dir = get_session_dir(upload_folder)
    metrics_files = [
        os.path.join(session_dir, "models", "output", "metrics_simple_nn.json"),
        os.path.join(session_dir, "models", "output", "metrics_lr_model.json"),
        os.path.join(session_dir, "models", "output", "metrics_complex_nn.json"),
        os.path.join(session_dir, "models", "output", "metrics_deep_nn.json")
    ]
    all_metrics = []
    for mf in metrics_files:
        if os.path.exists(mf):
            try:
                with open(mf, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_metrics.extend(data)
                    elif isinstance(data, dict):
                        all_metrics.append(data)
            except Exception:
                pass
    if all_metrics:
        return {"status": "success", "metrics": all_metrics}
    return {"status": "not_found", "metrics": None}

def compute_baseline_kpis(upload_folder, num_reps=10):
    session_dir = get_session_dir(upload_folder)
    scenario_path = os.path.join(session_dir, "data", "synthetic", "model", "scenario.json")
    model_path = os.path.join(session_dir, "data", "synthetic", "model", "model.json")
    if not os.path.exists(scenario_path) or not os.path.exists(model_path):
        return {"status": "error", "message": "Please upload model.json and scenario.json in Step 1 first."}

    try:
        if NN_KPI_DIR not in sys.path:
            sys.path.append(NN_KPI_DIR)
        from helpers.simulator import ScenarioSimulator

        with open(scenario_path, 'r') as f:
            base_json = json.load(f)
        with open(model_path, 'r') as f:
            full_model = json.load(f)

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

        simulator = ScenarioSimulator(baseline_scenario, full_model, process_details, seed=42)
        res = simulator.run_scenario(replications=num_reps, until=86400 * 90)

        cost = float(round(res.get("total_cost", 0.0), 2))
        cycle_time = float(round(res.get("avg_cycle_time", 0.0), 1))
        waiting_time = float(round(res.get("avg_wait_time", 0.0), 1))

        return {
            "status": "success",
            "baseline": {
                "cost": cost,
                "cycle_time": cycle_time,
                "waiting_time": waiting_time
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def _run_optimization_thread(session_dir, max_iterations, max_param_change_pct, weight_cost, weight_cycle, weight_waiting, target_cost=None, target_cycle=None, target_waiting=None, model_type="simple"):
    global STATUS
    STATUS["optimization"] = {"running": True, "progress": "Starting optimization...", "error": None, "completed": False}
    try:
        optimizer_script = os.path.join(NN_KPI_DIR, "models", "simple_model", "optimizer.py")
        if not os.path.exists(optimizer_script):
            raise FileNotFoundError(f"Optimizer script not found: {optimizer_script}")

        cmd = [
            sys.executable,
            optimizer_script,
            "synthetic",
            "--max_iterations", str(max_iterations),
            "--max_param_change_pct", str(max_param_change_pct),
            "--weight_cost", str(weight_cost),
            "--weight_cycle", str(weight_cycle),
            "--weight_waiting", str(weight_waiting),
            "--model_type", str(model_type)
        ]
        if target_cost is not None and float(target_cost) > 0:
            cmd.extend(["--target_cost", str(target_cost)])
        if target_cycle is not None and float(target_cycle) > 0:
            cmd.extend(["--target_cycle", str(target_cycle)])
        if target_waiting is not None and float(target_waiting) > 0:
            cmd.extend(["--target_waiting", str(target_waiting)])

        env = os.environ.copy()
        env["PYTHONPATH"] = NN_KPI_DIR + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.Popen(
            cmd,
            cwd=session_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        output_lines = []
        if proc.stdout:
            for line in proc.stdout:
                output_lines.append(line)
                if any(k in line for k in ["Optimizing", "Epoch", "TARGET", "VALIDATION", "OPTIMIZED", "RECOMMENDED", "Step", "[0/", "[1/", "[2/", "[3/", "[4/"]):
                    STATUS["optimization"]["progress"] = line.strip()

        proc.wait()
        if proc.returncode == 0:
            STATUS["optimization"] = {"running": False, "progress": "Optimization completed successfully!", "error": None, "completed": True}
        else:
            err_msg = "".join(output_lines[-15:]).strip() or f"Optimization failed with exit code {proc.returncode}"
            STATUS["optimization"] = {"running": False, "progress": "", "error": err_msg, "completed": False}
    except Exception as e:
        STATUS["optimization"] = {"running": False, "progress": "", "error": str(e), "completed": False}

def start_optimization(upload_folder, max_iterations=10000, max_param_change_pct=50.0, weight_cost=33.0, weight_cycle=33.0, weight_waiting=34.0, target_cost=None, target_cycle=None, target_waiting=None, model_type=None):
    session_dir = get_session_dir(upload_folder)
    if STATUS["optimization"]["running"]:
        return {"status": "busy", "message": "Optimization is already running"}

    if not model_type:
        model_type = STATUS["training"].get("model_type", "simple")

    t = threading.Thread(
        target=_run_optimization_thread,
        args=(session_dir, max_iterations, max_param_change_pct, weight_cost, weight_cycle, weight_waiting, target_cost, target_cycle, target_waiting, model_type)
    )
    t.daemon = True
    t.start()
    return {"status": "started", "message": "Targeted optimization started"}

def get_optimization_status():
    return STATUS["optimization"]

def get_optimization_results(upload_folder):
    session_dir = get_session_dir(upload_folder)
    results_path = os.path.join(session_dir, "models", "output", "optimizer_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        return {"status": "success", "results": results}
    return {"status": "not_found", "results": None}

def get_training_status():
    return STATUS["training"]

def get_session_status():
    return STATUS

