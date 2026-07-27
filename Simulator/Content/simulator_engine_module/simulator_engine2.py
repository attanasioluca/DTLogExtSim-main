from datetime import datetime, timedelta
from datetime import time as dt_time
import numpy as np
import math
import timeCalculator

class SimulatorEngine:
    def __init__(self, env, name, process_details, num, loader, global_resources, logging_opt, totalCost, timeUsedPerResource, extraLog, rows, start_delay=0, instance_type="default"):
        self.env = env
        self.name = name
        self.process_details = process_details
        self.num = num
        self.loader = loader
        self.global_resources = global_resources
        self.logging_opt = logging_opt
        self.totalCost = totalCost
        self.timeUsedPerResource = timeUsedPerResource
        self.extraLog = extraLog
        self.rows = rows
        self.start_delay = start_delay
        self.instance_type = instance_type

        # Initialize State Tracking
        self.stack = []
        self.stackInclusive = []
        self.executed_nodes = {self.num: set()}
        self.terminateEndEvent = {self.num: False}
        self.subprocessTerminate = {self.num: {}}
        self.subprocessExternalException = {}
        self.subprocessInternalError = {}
        self.worklist_resources = {self.num: {}}
        self.state_changed_event = {self.num: self.env.event()}
        self.totalCost[self.num] = 0.0
        
        self._initialize_configurations()
        
        # Start the engine
        self.action = env.process(self.run())

    def _initialize_configurations(self):
        """Extracts configuration mapping out of the constructor for readability."""
        extra = self.loader.extra_data
        
        # Thresholds & Limits
        self.costThresholds = {e['elementId']: float(e['costThreshold']) if e['costThreshold'] else None for e in extra['elements']}
        self.durationThresholds = {e['elementId']: float(e['durationThreshold']) if e['durationThreshold'] else None for e in extra['elements']}
        self.durationThresholdTimeUnits = {e['elementId']: e['durationThresholdTimeUnit'] for e in extra['elements']}
        
        # Normalize time thresholds to seconds
        for el_id, t_unit in self.durationThresholdTimeUnits.items():
            if self.durationThresholds[el_id] is not None:
                multiplier = {'minutes': 60, 'hours': 3600, 'days': 86400}.get(t_unit, 1)
                self.durationThresholds[el_id] *= multiplier

        # Task specific data
        self.task_durations = {e['elementId']: e['durationDistribution'] for e in extra['elements']}
        self.task_resources = {e['elementId']: e['resourceIds'] for e in extra['elements']}
        self.task_costs = {e['elementId']: e['fixedCost'] for e in extra['elements']}
        self.tasks_worklists = {e['elementId']: e['worklistId'] for e in extra['elements']}
        self.catchEvents = extra.get('catchEvents', {})

        # Group resources
        self.task_grouped_resources = {}
        for el_id, res_list in self.task_resources.items():
            grouped = {}
            for res in (res_list or []):
                grouped.setdefault(res['groupId'], []).append((res['resourceName'], int(res['amountNeeded'])))
            self.task_grouped_resources[el_id] = grouped

        # Time tracking
        self.start_datetime_obj = datetime.strptime(extra['startDateTime'], "%Y-%m-%dT%H:%M:%S")
        self.sequence_flows_dict = {item['elementId']: item for item in extra['sequenceFlows']}

    def run(self):
        """Main execution entry point."""
        yield self.env.timeout(self.start_delay)
        
        start_node_id = next((n_id for n_id, n in self.process_details['node_details'].items() if n['type'] == 'startEvent'), None)
        if start_node_id:
            yield from self.run_node(start_node_id)
            
        self._log_threshold_breaches()

    def _log_threshold_breaches(self):
        """Checks for threshold violations at the end of the instance."""
        for element_id, limit in self.durationThresholds.items():
            if limit is not None and limit < 0.0:
                self.extraLog[f"{element_id} exceeded DURATION threshold in instance {self.num} by:"] = abs(limit)
        for element_id, limit in self.costThresholds.items():
            if limit is not None and limit < 0.0:
                self.extraLog[f"{element_id} exceeded COST threshold in instance {self.num} by:"] = abs(limit)

    # --- DISPATCHER ---
    def run_node(self, node_id, subprocess_node=None):
        """Routes execution to specific node handlers based on type."""
        if self.terminateEndEvent[self.num]:
            return

        node = self._get_node(node_id, subprocess_node)
        node_type = node['type'].replace('_', '') # e.g. parallelGateway_close -> parallelGatewayclose
        
        handler_method = getattr(self, f"_handle_{node_type}", self._handle_unsupported)
        
        # Execute the handler
        result = handler_method(node_id, node, subprocess_node)
        
        # Only 'yield from' if the handler actually returned a generator
        # (Methods without 'yield' will return None and safely skip this)
        if result is not None:
            yield from result

    def _get_node(self, node_id, subprocess_node):
        if subprocess_node is None:
            return self.process_details['node_details'][node_id]
        return self.process_details['node_details'][subprocess_node]['subprocess_details'][node_id]

    # --- SPECIFIC NODE HANDLERS ---
    
    def _handle_startEvent(self, node_id, node, subprocess_node):
        if node.get('subtype') == "timerEventDefinition":
            wait_time = timeCalculator.convert_to_seconds(self.catchEvents[node_id])
            self.xeslog(node_id, "start", node['type'])
            yield self.env.timeout(wait_time)
            
        if node['previous']:
            while not all(prev in self.executed_nodes[self.num] for prev in node['previous']):
                yield self.state_changed_event[self.num]

        self.xeslog(node_id, "complete", node['type'])
        self.mark_node_executed(node_id)
        yield from self.run_node(node['next'][0], subprocess_node)

    def _handle_task(self, node_id, node, subprocess_node):
        if subprocess_node is not None and self._check_subprocess_exceptions(subprocess_node):
            return # Abort if external exception caught

        # Wait for sequence flows
        if node['previous']:
            while not all(prev in self.executed_nodes[self.num] for prev in node['previous']):
                yield self.state_changed_event[self.num]

        self.xeslog(node_id, "assign", node['type'])
        task_time = timeCalculator.convert_to_seconds(self.task_durations[node_id])
        if self.durationThresholds[node_id] is not None:
            self.durationThresholds[node_id] -= task_time

        # --- Modularized Resource Allocation ---
        acquired_requests = []
        if self.task_resources[node_id]:
            acquired_requests = yield self.env.process(self._allocate_task_resources(node_id))
            if self.terminateEndEvent[self.num]: return

        self.xeslog(node_id, "start", node['type'])
        yield self.env.timeout(task_time if math.isfinite(task_time) else 0)

        # Check exceptions mid-task
        if self.terminateEndEvent[self.num] or (subprocess_node and self._check_subprocess_exceptions(subprocess_node)):
            return

        # Complete Task
        self.mark_node_executed(node_id)
        task_cost = self.task_costs[node_id]
        if task_cost:
            self.totalCost[self.num] += float(task_cost)
            if self.costThresholds[node_id] is not None:
                self.costThresholds[node_id] -= float(task_cost)

        self.xeslog(node_id, "complete", node['type'], resourceid=[r[0] for r in acquired_requests], taskCost=task_cost)

        # Release Resources
        for name, req, cost_per_hour, res_simpy, mode in acquired_requests:
            self.timeUsedPerResource[name] += task_time
            if req.triggered:
                req.cancel()
                req.resource.release(req)
                self.trigger_resource_release()

        yield from self.run_node(node['next'][0], subprocess_node)

    def _handle_exclusiveGateway(self, node_id, node, subprocess_node):
        flows = [(fid, f) for fid, f in self.loader.process_data['sequence_flows'].items() if f['sourceRef'] == node_id]
        probs = {}
        forced_target = None
        
        for fid, f in flows:
            diagbp_flow = self.sequence_flows_dict.get(fid)
            if diagbp_flow:
                # Check for forced instance types
                for type_dict in diagbp_flow.get('types', []):
                    if type_dict['type'] == self.instance_type:
                        forced_target = f['targetRef']
                        break
                if forced_target: break
                probs[f['targetRef']] = float(diagbp_flow['executionProbability'])

        if forced_target:
            next_node = forced_target
        elif probs:
            # Normalize probabilities if they don't sum to 1
            total = sum(probs.values())
            normalized = [p/total for p in probs.values()]
            next_node = np.random.choice(list(probs.keys()), p=normalized)
        else:
            next_node = node['next'][0]

        self.xeslog(node_id, "complete", node['type'])
        yield from self.run_node(next_node, subprocess_node)

    def _handle_parallelGateway(self, node_id, node, subprocess_node):
        self.xeslog(node_id, "complete", node['type'])
        events = [self.env.process(self.run_node(next_id, subprocess_node)) for next_id in node['next']]
        yield self.env.all_of(events)
        
        if self.stack:
            close_id, next_node = self.stack.pop()
            self.xeslog(close_id, "complete", node['type'])
            yield from self.run_node(next_node, subprocess_node)

    def _handle_parallelGatewayclose(self, node_id, node, subprocess_node):
        if (node_id, node['next'][0]) not in self.stack:
            self.stack.append((node_id, node['next'][0]))

    def _handle_endEvent(self, node_id, node, subprocess_node):
        self.mark_node_executed(node_id)
        self.xeslog(node_id, "complete", node['type'])
        
        subtype = node.get('subtype')
        if subtype == 'terminateEventDefinition':
            if subprocess_node is None:
                self.terminateEndEvent[self.num] = True
            else:
                self.subprocessTerminate[self.num][subprocess_node] = True
        elif subtype == 'errorEventDefinition' and subprocess_node is not None:
            self.subprocessInternalError[subprocess_node] = True

    def _handle_unsupported(self, node_id, node, subprocess_node):
        """Fallback for nodes not yet implemented in the dispatcher."""
        # Add handling for missing node types like eventBasedGateway here
        pass

    # --- CORE UTILITIES ---
    def _allocate_task_resources(self, node_id):
        """Extracted logic for polling and requesting resources."""
        grouped_resources = self.task_grouped_resources.get(node_id, {})
        # Simplified resource allocation loop here (replace with your existing complex logic, 
        # but isolated to keep the main task handler clean).
        # Return a list of acquired_requests: [(name, request, cost, simpy_resource, mode), ...]
        yield self.env.timeout(0) # Placeholder
        return []

    def _check_subprocess_exceptions(self, subprocess_node):
        """Handles boundary errors and terminations cleanly."""
        if self.subprocessTerminate[self.num].get(subprocess_node):
            return True
        # add boundary event logic here
        return False

    def mark_node_executed(self, node_id):
        self.executed_nodes[self.num].add(node_id)
        if not self.state_changed_event[self.num].triggered:
            self.state_changed_event[self.num].succeed()
            self.state_changed_event[self.num] = self.env.event()

    def trigger_resource_release(self):
        if not hasattr(self.env, 'resource_released_event'):
            self.env.resource_released_event = self.env.event()
        if not self.env.resource_released_event.triggered:
            self.env.resource_released_event.succeed()
            self.env.resource_released_event = self.env.event()

    def xeslog(self, node_id, status, nodeType, resourceid=None, taskCost=None, resource_cost_hour=None):
        if not math.isfinite(self.env.now): return
        if self.logging_opt or status == "complete":
            act_name = self.process_details['node_details'].get(node_id, {}).get('name', node_id)
            current_time = self.start_datetime_obj + timedelta(seconds=self.env.now)
            self.rows.append([
                self.num, act_name, current_time, status, nodeType, 
                self.name, self.instance_type, resourceid, taskCost, resource_cost_hour
            ])