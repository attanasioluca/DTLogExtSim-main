#!/bin/bash

# Exit script immediately if any command fails
set -e

echo "=================================================="
echo " Starting Full NN-KPI-Predictor Benchmark Test"
echo "=================================================="

# Detect Python executable
if [ -f "./.venv/bin/python" ]; then
    PYTHON_CMD="./.venv/bin/python"
elif [ -f "./env/bin/python" ]; then
    PYTHON_CMD="./env/bin/python"
else
    PYTHON_CMD="python3"
fi
echo "Using Python executable: $PYTHON_CMD"

SOURCES=("synthetic" "real" "BIMP")
TRAIN_NUMS=(1000 5000 10000 25000 40000)

echo "=================================================="
echo " 1/2 Hypertuning Complex NN & Deep Network"
echo "=================================================="

for source in "${SOURCES[@]}"; do
#    echo -e "\n[Hypertuner] Complex Model ($source)..."
#    $PYTHON_CMD models/complex_model/hypertuner.py "$source"
#    
#    echo -e "\n[Hypertuner] Deep Network ($source)..."
#    $PYTHON_CMD models/deep_network/hypertuner.py "$source"
#    
    echo -e "\n[Hypertuner] LightGBM ($source)..."
    $PYTHON_CMD models/lightgbm_model/hypertuner.py "$source"
done


echo -e "\n=================================================="
echo " 2/2 Running Benchmark Trainings Across Train Sizes"
echo "=================================================="
for source in "${SOURCES[@]}"; do
    for num in "${TRAIN_NUMS[@]}"; do
        echo -e "\n--------------------------------------------------"
        echo " Data Source: $source | Train Samples: $num"
        echo "--------------------------------------------------"
        
        #echo -e "\n  -> Training LR Model ($source, train_num=$num)..."
        #$PYTHON_CMD models/LR_model/lr_model.py "$source" --train_num "$num"
        #
        echo -e "\n  -> Training Hypertuned LightGBM Model ($source, train_num=$num)..."
        $PYTHON_CMD models/lightgbm_model/hypertuned_model.py "$source" --train_num "$num"
        #
        #echo -e "\n  -> Training Simple Model ($source, train_num=$num)..."
        #$PYTHON_CMD models/simple_model/model.py "$source" --train_num "$num"
        #
        #echo -e "\n  -> Training Hypertuned Complex Model ($source, train_num=$num)..."
        #$PYTHON_CMD models/complex_model/hypertuned_model.py "$source" --train_num "$num"
        #
        #echo -e "\n  -> Training Hypertuned Deep Network ($source, train_num=$num)..."
        #$PYTHON_CMD models/deep_network/hypertuned_model.py "$source" --train_num "$num"
    done
done

echo -e "\n=================================================="
echo "  All Benchmark Runs Completed Successfully!"
echo "=================================================="

$PYTHON_CMD models/output/json_to_csv.py

