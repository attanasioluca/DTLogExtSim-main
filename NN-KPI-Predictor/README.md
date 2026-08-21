# NN-KPI-Predictor

Neural-network surrogate model for predicting business process KPIs from BPMN simulation scenarios.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

Master’s thesis project developed at Sapienza University of Rome.

This project trains a multi-output neural network to predict **cycle time, waiting time, and operational cost** from process simulation parameters. The objective is to support faster what-if analysis and process optimization by reducing the need for repeated simulation runs.

Technologies: PyTorch, SimPy, Pandas, Scikit-learn, BPMN, Docker

---

## Overview

Traditional BPMN what-if analysis may require a large number of simulation executions. This project learns the relationship between **process configuration parameters and KPI outcomes**, allowing near-instant estimation of KPIs for new scenarios.

<pre>
BPMN scenario → Simulation data → Neural network → KPI prediction
</pre>

---

## Training Results
<img width="1551" height="325" alt="image" src="https://github.com/user-attachments/assets/9c87d0de-cbf2-435c-88bd-3390e0bb14ff" />
<img width="1553" height="351" alt="image" src="https://github.com/user-attachments/assets/acea50b9-bcac-4e4b-83fb-0782f9fe832d" />

--- 

## Motivation

Business process analysts often need to evaluate thousands of alternative scenarios:

- number of workers
- resource allocation
- salaries
- machine availability
- task durations
- branching probabilities

Running a full simulation for every candidate configuration is slow. The goal of this project is to **replace repeated simulations with a fast neural-network predictor**, enabling near-instant KPI estimation and more efficient process optimization.

---

## Pipeline

<pre>
      Event Log
          │
          ▼
DTLog Process Extractor
          │
          ▼
BPMN process + scenario
          │
          ▼
Simulation engine (SimPy / process simulation)
          │
          ▼
Scenario generation (KPI dataset generation)
          │
          ▼
Feature preprocessing
          │
          ▼
Neural-network training
          │
          ▼
KPI prediction
</pre>

---

## Repository Structure

<pre>
NN-KPI-Predictor/
├── data/                  # Generated simulation datasets
├── models/                # Neural-network architectures
├── training/              # Training scripts
├── evaluation/            # Metrics and analysis
├── utils/                 # Helper functions
├── requirements.txt
└── README.md
</pre>

---

## Features
-
- Multi-output KPI prediction
- Configurable neural-network architectures
- Residual blocks and normalization layers
- Automated Hypertuning using Optuna
- Dataset merging and preprocessing utilities
- Evaluation with MAE, MSE, and MedAE
- Support for large process-simulation datasets

---

## Models

The main architecture is a **deep feed-forward surrogate network** (DeepNN). For additional context, 3 more models have been defined and tested:
- A Linear Regression Baseline model
- A simple Neural Network
- A more complex Neural Network, hypertuned using Optuna

## Data sources
Three different data sources were used for this thesis.
Each of the three source scenarios was used to create a simulation dataset, each with 25k alternative scenarios.
The data sources in question are:
- Synthetic: Based on a synthetic BPMN model and a realistic scenario.<img width="2806" height="730" alt="diagram" src="https://github.com/user-attachments/assets/4cb49aee-63d3-4111-9f9e-5ee935458b8b" />
- Real: Extracted from a real life event log: [here](https://data.4tu.nl/articles/_/12696884/1) using [simod](https://github.com/AutomatedProcessImprovement/Simod).
- BIMP: Using [this](https://bimp.cs.ut.ee/simulator/trial?sample=credit_card_application) realistic model + scenario combination from the BIMP simulator

## Optimizer: Example of usage
The optimizer.py code uses the trained model to optimize a given scenario (baseline) in order to achieve certain KPIs.
<img width="724" height="266" alt="image" src="https://github.com/user-attachments/assets/f2c4ca10-c164-4f9f-b1be-09512109de42" />

It then shows the user the changes made in the scenario to achieve them
<img width="849" height="423" alt="image" src="https://github.com/user-attachments/assets/2ae9a336-5de3-4344-b95c-f2e2ef8836aa" />




