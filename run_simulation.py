"""
Script function
1. Load and prepare partitioned dataset using data_utils.py
2. Define how individual clients are created and behave using client.py and model.py
3. Set up the Flower server with an aggregation strategy (we'll start with Federated Averaging - FedAvg)
4. Run the federated learning simulation
5. Evaluate global model on centralized test set
"""

# Import necessary Python Libraries
import flwr as fl
import torch
from collections import OrderedDict

# Import custom functions
from Data_Load_and_Prep.data_utils import load_and_preprocess_data, create_partitioned_dataloaders
from Model.model import FraudNet
from client import FraudDetectionClient, test as client_test_utility