import simpy
import numpy as np
from collections import defaultdict

# Make sure this import points to the correct location of your uploaded file
from simulator_engine import SimulatorEngine

class DataWrapperLoader:
    def __init__(self, base_data, full_model):
        self.extra_data = base_data
        self.process_data = full_model

def tracked_instance(
    env,
    instance_id,
    loader,
    process_details,
    global_resources,
    totalCost,
    timeUsedPerResource,
    durations,
    rows,       # <-- Added for SimulatorEngine
    rng,
    extraLog    # <-- Added for SimulatorEngine
):
    start_time = env.now

    # Initializing exactly to the simulator_engine_new.py signature
    engine = SimulatorEngine(
        env=env,
        name=f"Inst_{instance_id}",
        process_details=process_details,
        num=instance_id,
        loader=loader,
        global_resources=global_resources,
        logging_opt=0, 
        totalCost=totalCost,
        timeUsedPerResource=timeUsedPerResource,
        extraLog=extraLog,
        rows=rows,
        start_delay=0,
        rng=rng,
        instance_type="default",
    )

    yield engine.action

    # Only tracking cycle time (duration)
    cycle_time = env.now - start_time
    durations.append(cycle_time)


def arrival_generator(
    env,
    rng,
    loader,
    process_details,
    global_resources,
    totalCost,
    timeUsedPerResource,
    arrival_dist,
    durations,
    rows,
    extraLog
):
    instance_id = 0

    arr_type = arrival_dist.get("type", "exponential").lower()
    arr_mean = float(arrival_dist.get("mean", 10.0))

    while True:
        env.process(
            tracked_instance(
                env,
                instance_id,
                loader,
                process_details,
                global_resources,
                totalCost,
                timeUsedPerResource,
                durations,
                rows,
                rng,
                extraLog
            )
        )

        instance_id += 1

        if arr_type == "fixed":
            yield env.timeout(arr_mean)

        elif arr_type == "normal":
            std_dev = float(arrival_dist.get("arg1", arr_mean * 0.15))
            sample = rng.normal(arr_mean, std_dev)
            yield env.timeout(max(0.0, sample))

        else:
            yield env.timeout(rng.exponential(arr_mean))


class ScenarioSimulator:

    def __init__(
        self,
        base_json,
        full_model,
        process_details,
        seed=None,
    ):
        self.base = base_json
        self.full_model = full_model
        self.process_details = process_details
        self.seed = seed

    def _create_resources(self, env):

        global_resources = {}

        for res in self.base.get("resources", []):

            res_name = res["name"]
            res_amount = max(1, int(res["totalAmount"]))

            cost = float(res.get("costPerHour", 0.0))
            timetable = res.get("timetableName", "")
            setup_time = res.get("setupTime", {"type": ""})
            max_usage = res.get("maxUsage", "")

            res_list = []

            for _ in range(res_amount):
                res_list.append(
                    (
                        simpy.Resource(env, capacity=1),
                        cost,
                        timetable,
                        "default",
                        setup_time,
                        max_usage,
                        0,
                        simpy.Resource(env, capacity=1),
                    )
                )

            global_resources[res_name] = res_list

        return global_resources

    def run_replication(self, until=86400):

        rng = np.random.default_rng(self.seed)

        env = simpy.Environment()

        loader = DataWrapperLoader(
            self.base,
            self.full_model,
        )

        totalCost = {}
        timeUsedPerResource = defaultdict(float)
        durations = []
        rows = []        # Log array required by SimulatorEngine
        extraLog = {}    # Logging dict required by SimulatorEngine

        global_resources = self._create_resources(env)

        env.process(
            arrival_generator(
                env,
                rng,
                loader,
                self.process_details,
                global_resources,
                totalCost,
                timeUsedPerResource,
                self.base["arrivalRateDistribution"],
                durations,
                rows,
                extraLog
            )
        )

        env.run(until=until)

        completed = len(durations)

        total_fixed_cost = sum(totalCost.values())

        total_resource_cost = 0.0

        for resource_name, used_seconds in timeUsedPerResource.items():

            cost_per_hour = next(
                (
                    float(r.get("costPerHour", 0.0))
                    for r in self.base.get("resources", [])
                    if r["name"] == resource_name
                ),
                0.0,
            )

            total_resource_cost += (
                used_seconds / 3600.0
            ) * cost_per_hour

        total_cost = total_fixed_cost + total_resource_cost
        
        avg_duration = (
            float(np.mean(durations))
            if completed > 0
            else 0.0
        )

        return {
            "total_cost": total_cost,
            "duration": avg_duration,
            "wait_time": 0
        }

    def run_scenario(self, replications=50, until=86400):
        results = []
        for rep in range(replications):
            if self.seed is not None:
                self.seed += 1
                
            print(f"[Worker] Scenario {self.seed - rep} is running rep {rep}/{replications}...", flush=True)

            results.append(self.run_replication(until=until))

        total_costs = [r["total_cost"] for r in results]
        durations = [r["duration"] for r in results]

        # Returning 0 for wait times as explicitly requested
        return {
            "total_cost": float(np.mean(total_costs)),
            "std_cost": float(np.std(total_costs, ddof=1)) if len(total_costs) > 1 else 0.0,
            "avg_cycle_time": float(np.mean(durations)), 
            "std_cycle_time": float(np.std(durations, ddof=1)) if len(durations) > 1 else 0.0,
            "avg_wait_time": 0.0, 
            "std_wait_time": 0.0
        }