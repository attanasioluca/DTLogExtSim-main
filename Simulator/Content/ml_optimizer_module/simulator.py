import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from simulator_engine_module.simulator_engine2 import SimulatorEngine
import numpy as np
import simpy
import threading

def generalized_simulate(loader, n_instances=60):
    env = simpy.Environment()
    
    # 1. Setup Resource pools from the BPMN loader
    resources = loader.extra_data['resources']
    global_resources = {}
    for res in resources:
        resource_list = []
        for _ in range(int(res['totalAmount'])):
            simpy_res = simpy.Resource(env, capacity=1)
            # Structure: (Res, Cost, Timetable, LastType, Setup, MaxUsage, UsageContainer, Lock)
            resource_tuple = (
                simpy_res, 
                res['costPerHour'], 
                res['timetableName'], 
                "", 
                res['setupTime'], 
                res['maxUsage'], 
                simpy.Container(env, init=0, capacity=int(res['maxUsage'])) if res['maxUsage'] else None,
                threading.Lock()
            )
            resource_list.append(resource_tuple)
        global_resources[res['name']] = resource_list

    totalCost = {}
    timeUsedPerResource = {}
    rows = []
    
    # 2. Extract instance configuration
    instance_types = loader.extra_data['processInstances']
    
    # 3. Initialize the simulation for each instance
    for process_id, process_details in loader.process_data['process_elements'].items():
        for i in range(n_instances):
            # Assign instance types based on the BPMN configuration
            inst_type = instance_types[0]['type'] 
            SimulatorEngine(
                env, "Process", process_details, i, loader, global_resources, 
                False, totalCost, timeUsedPerResource, {}, rows, 
                start_delay=0, instance_type=inst_type
            )
    
    # 4. Execute the simulation
    env.run()
    
    # 5. Calculate Average KPI (in this case Mean Cost)
    mean_cost = np.mean(list(totalCost.values())) if totalCost else 0
    return {
        "mean_cost": float(mean_cost)
    }