import torch
import torch.nn as nn
from torchview import draw_graph

# ==========================================
# MODEL 1: Standard Surrogate
# ==========================================
class SurrogateModel1(nn.Module):
    def __init__(self, input_size):
        super(SurrogateModel1, self).__init__()
        self.shared_entry = nn.Sequential(
            nn.Linear(input_size, 64), nn.BatchNorm1d(64), nn.Mish(), nn.Dropout(0.2)
        )
        self.shared_out = nn.Sequential(nn.Linear(64, 32), nn.Mish())
        
        self.cost_branch = nn.Sequential(nn.Linear(32, 16), nn.Mish(), nn.Linear(16, 1))
        self.cycle_time_branch = nn.Sequential(nn.Linear(32, 16), nn.Mish(), nn.Linear(16, 1))
        self.waiting_time_branch = nn.Sequential(nn.Linear(32, 16), nn.Mish(), nn.Linear(16, 1))

    def forward(self, x):
        x = self.shared_entry(x)
        shared_features = self.shared_out(x)
        
        t_cost = self.cost_branch(shared_features)
        cycle = self.cycle_time_branch(shared_features)
        wait = self.waiting_time_branch(shared_features)

        return torch.cat((t_cost, cycle, wait), dim=1)


# ==========================================
# MODEL 2: Surrogate with Skip Connection
# ==========================================
class SurrogateModel2(nn.Module):
    def __init__(self, input_size, DROPOUT_RATE):
        super(SurrogateModel2, self).__init__()
        self.shared_entry = nn.Sequential(nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.Mish(), nn.Dropout(DROPOUT_RATE))
        self.shared_h1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.Mish()
        self.drop1 = nn.Dropout(DROPOUT_RATE)
        self.shared_h2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.Mish()
        self.drop2 = nn.Dropout(DROPOUT_RATE)
        self.shared_out = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.Mish())

        self.total_cost_branch   = nn.Sequential(nn.Linear(64, 32), nn.Mish(), nn.Linear(32, 1))
        self.cycle_time_branch   = nn.Sequential(nn.Linear(64, 32), nn.Mish(), nn.Linear(32, 1))
        self.waiting_time_branch = nn.Sequential(nn.Linear(64, 32), nn.Mish(), nn.Linear(32, 1))

    def forward(self, x):
        x = self.shared_entry(x)
        h1 = self.drop1(self.act1(self.bn1(self.shared_h1(x))))
        h2 = self.drop2(self.act2(self.bn2(self.shared_h2(h1))))
        
        # Skip connection
        h2 = h2 + h1 
        
        shared_features = self.shared_out(h2)
        
        t_cost = self.total_cost_branch(shared_features)
        cycle  = self.cycle_time_branch(shared_features)
        wait   = self.waiting_time_branch(shared_features)
        return torch.cat((t_cost, cycle, wait), dim=1)


# ==========================================
# MODEL 3: Deep ResNet Surrogate
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate)
        )
    def forward(self, x): 
        return x + self.net(x) 

class DeepSurrogateModel(nn.Module):
    def __init__(self, input_size, hidden_dim=256, num_blocks=4, dropout_rate=0.1):
        super(DeepSurrogateModel, self).__init__()
        
        self.entry = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish()
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)])
        self.shared_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish()
        )
        def make_branch():
            return nn.Sequential(
                nn.Linear(hidden_dim, 64), 
                nn.LayerNorm(64), 
                nn.Mish(),
                nn.Linear(64, 32), 
                nn.Mish(),
                nn.Linear(32, 1)
            )
            
        self.total_cost_branch   = make_branch()
        self.cycle_time_branch   = make_branch()
        self.waiting_time_branch = make_branch()

    def forward(self, x):
        x = self.entry(x)
        x = self.blocks(x)
        shared = self.shared_out(x)
        return torch.cat((
            self.total_cost_branch(shared), 
            self.cycle_time_branch(shared), 
            self.waiting_time_branch(shared)
        ), dim=1)


import torch
import torch.nn as nn
from torchview import draw_graph

# ==========================================
# MODEL 1: Standard Surrogate
# ==========================================
class SurrogateModel1(nn.Module):
    def __init__(self, input_size):
        super(SurrogateModel1, self).__init__()
        self.shared_entry = nn.Sequential(
            nn.Linear(input_size, 64), nn.BatchNorm1d(64), nn.Mish(), nn.Dropout(0.2)
        )
        self.shared_out = nn.Sequential(nn.Linear(64, 32), nn.Mish())
        
        self.cost_branch = nn.Sequential(nn.Linear(32, 16), nn.Mish(), nn.Linear(16, 1))
        self.cycle_time_branch = nn.Sequential(nn.Linear(32, 16), nn.Mish(), nn.Linear(16, 1))
        self.waiting_time_branch = nn.Sequential(nn.Linear(32, 16), nn.Mish(), nn.Linear(16, 1))

    def forward(self, x):
        x = self.shared_entry(x)
        shared_features = self.shared_out(x)
        
        t_cost = self.cost_branch(shared_features)
        cycle = self.cycle_time_branch(shared_features)
        wait = self.waiting_time_branch(shared_features)

        return torch.cat((t_cost, cycle, wait), dim=1)


# ==========================================
# MODEL 2: Surrogate with Skip Connection
# ==========================================
class SurrogateModel2(nn.Module):
    def __init__(self, input_size, DROPOUT_RATE):
        super(SurrogateModel2, self).__init__()
        self.shared_entry = nn.Sequential(nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.Mish(), nn.Dropout(DROPOUT_RATE))
        self.shared_h1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.Mish()
        self.drop1 = nn.Dropout(DROPOUT_RATE)
        self.shared_h2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.Mish()
        self.drop2 = nn.Dropout(DROPOUT_RATE)
        self.shared_out = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.Mish())

        self.total_cost_branch   = nn.Sequential(nn.Linear(64, 32), nn.Mish(), nn.Linear(32, 1))
        self.cycle_time_branch   = nn.Sequential(nn.Linear(64, 32), nn.Mish(), nn.Linear(32, 1))
        self.waiting_time_branch = nn.Sequential(nn.Linear(64, 32), nn.Mish(), nn.Linear(32, 1))

    def forward(self, x):
        x = self.shared_entry(x)
        h1 = self.drop1(self.act1(self.bn1(self.shared_h1(x))))
        h2 = self.drop2(self.act2(self.bn2(self.shared_h2(h1))))
        
        # Skip connection
        h2 = h2 + h1 
        
        shared_features = self.shared_out(h2)
        
        t_cost = self.total_cost_branch(shared_features)
        cycle  = self.cycle_time_branch(shared_features)
        wait   = self.waiting_time_branch(shared_features)
        return torch.cat((t_cost, cycle, wait), dim=1)


# ==========================================
# MODEL 3: Deep ResNet Surrogate
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate)
        )
    def forward(self, x): 
        return x + self.net(x) 

class DeepSurrogateModel(nn.Module):
    def __init__(self, input_size, hidden_dim=256, num_blocks=4, dropout_rate=0.1):
        super(DeepSurrogateModel, self).__init__()
        
        self.entry = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish()
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)])
        self.shared_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish()
        )
        def make_branch():
            return nn.Sequential(
                nn.Linear(hidden_dim, 64), 
                nn.LayerNorm(64), 
                nn.Mish(),
                nn.Linear(64, 32), 
                nn.Mish(),
                nn.Linear(32, 1)
            )
            
        self.total_cost_branch   = make_branch()
        self.cycle_time_branch   = make_branch()
        self.waiting_time_branch = make_branch()

    def forward(self, x):
        x = self.entry(x)
        x = self.blocks(x)
        shared = self.shared_out(x)
        return torch.cat((
            self.total_cost_branch(shared), 
            self.cycle_time_branch(shared), 
            self.waiting_time_branch(shared)
        ), dim=1)


 # ==========================================
# DIAGRAM GENERATION SCRIPT (TORCHVIEW)
# ==========================================
input_dim = 10
# We use a batch size of 2 to avoid BatchNorm errors
input_shape = (1, input_dim) 

# --- Generate Model 1 ---
model_1 = SurrogateModel1(input_size=input_dim)
model_1.eval()
graph_1 = draw_graph(model_1, input_size=input_shape, expand_nested=True)
# Changed format="pdf" to format="png"
graph_1.visual_graph.render("model_1_architecture", format="png")
print("Saved Model_1_torchview.png")

# --- Generate Model 2 ---
model_2 = SurrogateModel2(input_size=input_dim, DROPOUT_RATE=0.2)
model_2.eval()
graph_2 = draw_graph(model_2, input_size=input_shape, expand_nested=True)
graph_2.visual_graph.render("model_2_architecture", format="png")
print("Saved Model_2_with_Skip_torchview.png")

# --- Generate Model 3 ---
model_3 = DeepSurrogateModel(input_size=input_dim, hidden_dim=256, num_blocks=4, dropout_rate=0.1)
model_3.eval()
# Depth limits how far into the sub-modules the graph goes
graph_3 = draw_graph(model_3, input_size=input_shape, expand_nested=True, depth=2)
graph_3.visual_graph.render("model_3_architecture", format="png")
print("Saved Model_3_ResNet_torchview.png")