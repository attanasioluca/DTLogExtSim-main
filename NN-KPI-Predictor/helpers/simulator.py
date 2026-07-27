import simpy
import numpy as np
from collections import defaultdict
from .simulator_engine_waiting_times import SimulatorEngine
from datetime import timedelta


class DataWrapperLoader:
    def __init__(self, base_data, full_model):
        self.extra_data = base_data
        self.process_data = full_model


def get_business_cycle_time(start_sim_sec, end_sim_sec, start_datetime_obj):
    # 1. Convert simulation seconds to real-world datetime objects
    start_dt = start_datetime_obj + timedelta(seconds=start_sim_sec)
    end_dt = start_datetime_obj + timedelta(seconds=end_sim_sec)
    
    # 2. Get raw calendar duration
    calendar_seconds = (end_dt - start_dt).total_seconds()
    
    # 3. Calculate how many weekend days fell between the start and end dates
    start_date = start_dt.date()
    end_date = end_dt.date()
    
    total_days = (end_date - start_date).days
    business_days = np.busday_count(start_date, end_date) # Excludes Sat/Sun by default
    weekend_days = total_days - business_days
    
    # 4. Subtract the weekend seconds
    business_seconds = calendar_seconds - (weekend_days * 86400)
    
    return max(0.0, business_seconds)        

def tracked_instance(
    env,
    instance_id,
    loader,
    process_details,
    global_resources,
    totalCost,
    timeUsedPerResource,
    durations,
    wait_times,
    rng,
):
    start_time = env.now

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
        extraLog={},
        rows=[],
        rng=rng,
        start_delay=0,
        instance_type="default",
    )

    yield engine.action
    business_cycle_seconds = get_business_cycle_time(
        start_sim_sec=start_time, 
        end_sim_sec=env.now, 
        start_datetime_obj=engine.start_datetime_obj
    )
    durations.append(business_cycle_seconds)
    wait_times.append(engine.total_wait_time)

def arrival_generator(
    env,
    loader,
    process_details,
    global_resources,
    totalCost,
    timeUsedPerResource,
    arrival_dist,
    durations,
    wait_time,
    rng,
    max_instances=None  # <-- 1. Add this parameter
):
    instance_id = 0

    arr_type = arrival_dist.get("type", "exponential").lower()
    arr_mean = float(arrival_dist.get("mean", 10.0))

    while True:
        # <-- 2. Add this break condition
        if max_instances is not None and instance_id >= max_instances:
            break

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
                wait_time,
                rng,
            )
        )

        instance_id += 1

        if arr_type == "fixed":
            yield env.timeout(arr_mean)
        elif arr_type == "normal":
            std_dev = float(arrival_dist.get("arg1", arr_mean * 0.15))
            yield env.timeout(max(0, rng.normal(arr_mean, std_dev)))
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

    def run_replication(self, until=86400, max_instances=None):

        rng = np.random.default_rng(self.seed)

        env = simpy.Environment()

        loader = DataWrapperLoader(
            self.base,
            self.full_model,
        )

        totalCost = {}
        timeUsedPerResource = defaultdict(float)
        durations = []
        wait_times = []

        global_resources = self._create_resources(env)

        env.process(
            arrival_generator(
                env,
                loader,
                self.process_details,
                global_resources,
                totalCost,
                timeUsedPerResource,
                self.base["arrivalRateDistribution"],
                durations,
                wait_times,
                rng,
                max_instances=max_instances
            )
        )

        if max_instances is not None:
            env.run()
        else:
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
            "wait_time": float(np.mean(wait_times)) if completed > 0 else 0.0,
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
        wait_times = [r["wait_time"] for r in results]

        return {
            
            "total_cost": float(np.mean(total_costs)),
            "std_cost": float(np.std(total_costs, ddof=1)),
            "avg_cycle_time": float(np.mean(durations)), 
            "std_cycle_time": float(np.std(durations, ddof=1)),
            "avg_wait_time": float(np.mean(wait_times)), 
            "std_wait_time": float(np.std(wait_times, ddof=1))
        }