import flwr as fl
import torch
from collections import OrderedDict
import numpy as np

from Data_Load_and_Prep.data_utils import load_and_preprocess_data, create_partitioned_dataloaders
from Model.model import FraudNet 
from client import FraudDetectionClient, test as client_test_utility 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Simulation orchestrator using {DEVICE}")

# --- Configuration Parameters ---
NUM_CLIENTS = 5
BATCH_SIZE = 64
NUM_ROUNDS = 10  
LOCAL_EPOCHS = 3 # Might need more epochs for DP training, or adjust epsilon
LEARNING_RATE = 0.001 
DATA_PATH = "Data/creditcard.csv"
SEED = 42

# Non-IID data configuration
NON_IID_ALPHA = 0.5 
# NON_IID_ALPHA = None # Uncomment for IID-like split

# --- Differential Privacy Configuration ---
DP_ENABLED = True # Set to False to run without DP
DP_TARGET_EPSILON = 10.0  # Target privacy budget (epsilon) per client per round
# Delta should generally be smaller than 1/dataset_size.
# Here, it's a global suggestion; client.py refines it slightly.
DP_TARGET_DELTA = 1e-5   
DP_MAX_GRAD_NORM = 1.2   # Max norm for gradient clipping

def set_seeds(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def main():
    set_seeds(SEED)
    print(f"Seeds have been set to: {SEED}")

    print("Loading and preprocessing data...")
    X_global, y_global = load_and_preprocess_data(csv_path=DATA_PATH)
    print(f"Data loaded. X shape: {X_global.shape}, y shape: {y_global.shape}")

    print(f"Creating partitioned DataLoaders for {NUM_CLIENTS} clients...")
    client_train_dataloaders, central_test_loader, num_features = create_partitioned_dataloaders(
        X_global, y_global, 
        num_clients=NUM_CLIENTS, 
        batch_size=BATCH_SIZE,
        seed=SEED,
        non_iid_alpha=NON_IID_ALPHA 
    )
    print(f"Number of features for the model: {num_features}")

    def model_fn():
        return FraudNet(num_features=num_features)

    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)
        if client_id >= len(client_train_dataloaders):
            raise ValueError(f"Client ID {cid} is out of bounds for the number of dataloaders provided.")
        trainloader = client_train_dataloaders[client_id]
        valloader = trainloader 
        
        if len(trainloader.dataset) == 0 and DP_ENABLED:
             print(f"Client {cid} has no data. DP training would be skipped by client.")
        
        return FraudDetectionClient(model_fn=model_fn, trainloader=trainloader, valloader=valloader).to_client()

    def get_evaluate_fn(test_loader, model_instantiation_fn):
        def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
            if server_round == 0 and not parameters: return None, {}
            model = model_instantiation_fn() 
            if parameters:
                params_dict = zip(model.state_dict().keys(), parameters)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=True)
            model.to(DEVICE) 
            loss, accuracy, num_examples = client_test_utility(model, test_loader)
            # Add DP info to server log if needed, though spent_epsilon is client-specific
            print(f"Round {server_round}: Server-side evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f} on {num_examples} test samples")
            return loss, {"accuracy": accuracy}
        return evaluate

    evaluate_fn_for_server = get_evaluate_fn(central_test_loader, model_fn)

    def fit_config(server_round: int):
        config = {
            "local_epochs": LOCAL_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "server_round": server_round, 
        }
        if DP_ENABLED:
            config["dp_target_epsilon"] = DP_TARGET_EPSILON
            config["dp_target_delta"] = DP_TARGET_DELTA
            config["dp_max_grad_norm"] = DP_MAX_GRAD_NORM
        return config
    
    initial_model = model_fn()
    initial_parameters = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    )

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  
        fraction_evaluate=0.0, 
        min_fit_clients=NUM_CLIENTS,  
        min_evaluate_clients=0, 
        min_available_clients=NUM_CLIENTS, 
        evaluate_fn=evaluate_fn_for_server, 
        on_fit_config_fn=fit_config,  
        initial_parameters=initial_parameters, 
    )

    print(f"Starting Federated Learning simulation for {NUM_ROUNDS} rounds with {NUM_CLIENTS} clients...")
    print(f"Using Non-IID Alpha: {NON_IID_ALPHA}")
    if DP_ENABLED:
        print(f"Differential Privacy ENABLED. Target Epsilon: {DP_TARGET_EPSILON}, Target Delta: {DP_TARGET_DELTA}, Max Grad Norm: {DP_MAX_GRAD_NORM}")
    else:
        print("Differential Privacy DISABLED.")
        
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0} if DEVICE == torch.device("cpu") else {"num_cpus": 1, "num_gpus": 0.2}
    )

    print("Federated Learning simulation finished.")
    print("History (server-side evaluation metrics):", history.metrics_centralized)
    if history.losses_centralized:
         print("History (server-side evaluation loss):", history.losses_centralized)
    # You could also try to access custom metrics from fit (like spent_epsilon) if strategy aggregates them
    # For example, history.metrics_distributed_fit might contain them if FedAvg is configured to aggregate.

if __name__ == "__main__":
    main()