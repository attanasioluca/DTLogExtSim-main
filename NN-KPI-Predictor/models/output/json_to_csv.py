import json
import os
from collections import defaultdict

files = [
    "models/output/metrics_lr_model.json",
    "models/output/metrics_simple_nn.json",
    "models/output/metrics_complex_nn.json",
    "models/output/metrics_deep_nn.json"
]

# Removed "Data Source" from columns since it will be used as a section header
columns = [
    "Model Name",
    "Best Test Loss",
    "MedAE - Cost",
    "MedAE - Cycle Time",
    "MedAE - Waiting Time",
    "MAE - Cost",
    "MAE - Cycle Time",
    "MAE - Waiting Time",
    "MedAE % - Cost",
    "MedAE % - Cycle Time",
    "MedAE % - Waiting Time"
]

# Dictionary to hold the grouped data
grouped_data = defaultdict(list)

# 1. Read and group the data
for file_name in files:
    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        continue
        
    with open(file_name, 'r') as f:
        data_list = json.load(f)
        
    if isinstance(data_list, dict):
        data_list = [data_list]
        
    for data in data_list:
        source = str(data.get("Data Source", "Unknown"))
        grouped_data[source].append(data)

# 2. Print the formatted output
for source, records in grouped_data.items():
    print(source)
    print(",".join(columns))
    
    for data in records:
        test_loss = data.get("Best Test Loss", "")
        if test_loss != "":
            test_loss = f"{float(test_loss):.4f}"
            
        row = [
            str(data.get("Model Name", "Unknown Model")),
            test_loss,
            f"{data['MedAE']['cost']:.2f}",
            f"{data['MedAE']['cycle_time']:.2f}",
            f"{data['MedAE']['waiting_time']:.2f}",
            f"{data['MAE']['cost']:.2f}",
            f"{data['MAE']['cycle_time']:.2f}",
            f"{data['MAE']['waiting_time']:.2f}",
            f"{data['MedAE_Percentage']['cost']:.2f}",
            f"{data['MedAE_Percentage']['cycle_time']:.2f}",
            f"{data['MedAE_Percentage']['waiting_time']:.2f}"
        ]
        print(",".join(row))
    
    # Add a blank line for readability between sources
    print()