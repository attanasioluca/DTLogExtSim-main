import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from simulator import generalized_simulate
from bpmn_processing_module.bpmn_handler import BPMNHandler
import numpy as np
import json
import parsingAgain

# --- THE AGENT (From Source 1) ---
class QLearningAgent:
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.Q: dict[tuple, np.ndarray] = {}

    def _get_q(self, state: tuple, n_actions: int) -> np.ndarray:
        if state not in self.Q:
            self.Q[state] = np.zeros(n_actions)
        return self.Q[state]

    def choose_action(self, state: tuple, n_actions: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(n_actions)
        return int(np.argmax(self._get_q(state, n_actions)))

    def update(self, state, action, reward, next_state, n_actions):
        q = self._get_q(state, n_actions)
        q_next = self._get_q(next_state, n_actions)
        q[action] += self.alpha * (reward + self.gamma * np.max(q_next) - q[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# --- THE DYNAMIC OPTIMIZER ---
class DynamicRLOptimizer:
    def __init__(self, bpmn_path, extra_path, scenario_idx):
        # 1. Pre-process the BPMN for sequence flow logic
        parsingAgain.parse_again(bpmn_path)
        
        # 2. Load the scenario configuration
        self.loader = BPMNHandler("", "", extra_path, bpmn_path, scenario_idx)
        self.loader.load_configuration_files()
        
        # 3. Build action space dynamically from the JSON
        self.ACTIONS = []
        self.BOUNDS = {}
        self._build_action_space()
        self.N_ACTIONS = len(self.ACTIONS)

    def _build_action_space(self):
        # Identify all resources that can be scaled
        for i, res in enumerate(self.loader.extra_data['resources']):
            self.ACTIONS.append(("resource", i, 1))  # Increment count
            self.ACTIONS.append(("resource", i, -1)) # Decrement count
            self.BOUNDS[f"res_{i}"] = (1, 20)

        # Identify all tasks with durations that can be tuned
        for i, elem in enumerate(self.loader.extra_data['elements']):
            if elem.get('durationDistribution'):
                self.ACTIONS.append(("duration", i, 0.90)) # Speed up 10%
                self.ACTIONS.append(("duration", i, 1.10)) # Slow down 10%
                self.BOUNDS[f"dur_{i}"] = (10, 14400)

    def apply_action(self, action_idx):
        # Directly modify the loader's extra_data in memory
        type, idx, delta = self.ACTIONS[action_idx]
        
        if type == "resource":
            current = int(self.loader.extra_data['resources'][idx]['totalAmount'])
            new_val = np.clip(current + delta, *self.BOUNDS[f"res_{idx}"])
            self.loader.extra_data['resources'][idx]['totalAmount'] = int(new_val)
            
        elif type == "duration":
            current = float(self.loader.extra_data['elements'][idx]['durationDistribution']['mean'])
            new_val = np.clip(current * delta, *self.BOUNDS[f"dur_{idx}"])
            self.loader.extra_data['elements'][idx]['durationDistribution']['mean'] = str(new_val)

    def get_state(self):
        # Convert current JSON configuration into a state tuple for the Q-table
        res_counts = [int(r['totalAmount']) for r in self.loader.extra_data['resources']]
        durations = [float(e['durationDistribution']['mean']) for e in self.loader.extra_data['elements'] if e.get('durationDistribution')]
        return tuple(res_counts + durations)

# --- TRAINING EXECUTION ---
def run_optimization(bpmn_file, extra_file, scenario_idx):
    optimizer = DynamicRLOptimizer(bpmn_file, extra_file, scenario_idx)
    agent = QLearningAgent()
    
    # Run episodes
    for ep in range(100):
        optimizer.loader.load_configuration_files() # Start from baseline
        state = optimizer.get_state()
        
        for _ in range(10):
            action_idx = agent.choose_action(state, optimizer.N_ACTIONS)
            optimizer.apply_action(action_idx)
            
            # Use the new generalized simulator
            results = generalized_simulate(optimizer.loader)
            reward = -results['mean_cost'] # Goal is to minimize cost
            
            next_state = optimizer.get_state()
            agent.update(state, action_idx, reward, next_state, optimizer.N_ACTIONS)
            state = next_state
        
        agent.decay_epsilon()
        print(f"Episode {ep} complete. Epsilon: {agent.epsilon:.3f}")


base_dir = os.path.dirname(os.path.abspath(__file__))
bpmn_file = os.path.join(base_dir, "data", "model.json")
extra_file = os.path.join(base_dir, "data", "extra.json")
run_optimization(bpmn_file, extra_file, 0)