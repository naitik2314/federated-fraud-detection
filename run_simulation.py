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

# Setup device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Simulation orchestrator using {DEVICE}")

# --- Configuration Parameters ---
NUM_CLIENTS = 5
BATCH_SIZE = 64
NUM_ROUNDS = 10  # Number of federated learning rounds
LOCAL_EPOCHS = 2 # Number of local training epochs on each client
LEARNING_RATE = 0.001
DATA_PATH = "Data/creditcard.csv" # Path to your CSV file

def main():
    # --- 1. Load and Prepare Data ---
    print("Loading and preprocessing data...")
    X_global, y_global = load_and_preprocess_data(csv_path=DATA_PATH)
    print(f"Data loaded. X shape: {X_global.shape}, y shape: {y_global.shape}")

    print(f"Creating partitioned DataLoaders for {NUM_CLIENTS} clients...")
    # For this simulation, each client will use its entire data partition for both
    # local training and local evaluation via the client's evaluate() method.
    # create_partitioned_dataloaders returns a list of train_loaders, one for each client.
    client_train_dataloaders, central_test_loader, num_features = create_partitioned_dataloaders(
        X_global, y_global, num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE
    )
    print(f"Number of features for the model: {num_features}")

    # --- 2. Define Model Instantiation Function ---
    # This function will be used by clients and the server (for initial params or eval)
    def model_fn():
        return FraudNet(num_features=num_features)

    # --- 3. Define Client Function (client_fn) ---
    # This function is called by Flower to create a client instance for simulation
    def client_fn(cid: str) -> FraudDetectionClient:
        """Create a Flower client instance for simulation."""
        client_id = int(cid)
        trainloader = client_train_dataloaders[client_id]
        # For FraudDetectionClient, valloader is used by its `evaluate` method.
        # Here, we let clients evaluate on their own training data partition.
        # In a more advanced setup, clients would have their own local validation split.
        valloader = trainloader # Or a dedicated validation split for the client
        
        return FraudDetectionClient(model_fn=model_fn, trainloader=trainloader, valloader=valloader)

    # --- 4. Define Server-Side Evaluation Function (get_evaluate_fn) ---
    # This function is passed to the strategy to evaluate the global model on the server
    def get_evaluate_fn(test_loader, model_instantiation_fn):
        """Return an evaluation function for server-side evaluation."""
        def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
            if server_round == 0 and not parameters: # Skip evaluation if parameters are empty (e.g. before first round for FedAdam)
                 return None, {}

            model = model_instantiation_fn() # Create a new model instance
            # Set model parameters from the server
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            model.to(DEVICE) # Move model to device

            # Use the test utility from client.py (or a dedicated server-side one)
            loss, accuracy, num_examples = client_test_utility(model, test_loader) # testloader here is central_test_loader
            print(f"Round {server_round}: Server-side evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f} on {num_examples} test samples")
            return loss, {"accuracy": accuracy}
        return evaluate

    evaluate_fn_for_server = get_evaluate_fn(central_test_loader, model_fn)

    # --- 5. Define Federated Learning Strategy (FedAvg) ---
    # Function to configure client training
    def fit_config(server_round: int):
        """Return training configuration dict for each round.
        Keep LR static for now, but could be adapted per round.
        """
        config = {
            "local_epochs": LOCAL_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "server_round": server_round, # Pass server_round for potential use by client
        }
        return config
    
    # For FedAvg, initial_parameters can be set to initialize the global model.
    # If None, the server will wait for the first client to send parameters.
    # Let's initialize it on the server.
    initial_model = model_fn()
    initial_parameters = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]


    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Fraction of clients to use for training (1.0 = all clients)
        fraction_evaluate=1.0,  # Fraction of clients to use for evaluation (1.0 = all clients)
        min_fit_clients=NUM_CLIENTS,  # Minimum number of clients to train
        min_evaluate_clients=NUM_CLIENTS,  # Minimum number of clients to evaluate
        min_available_clients=NUM_CLIENTS, # Minimum number of clients available for the round
        evaluate_fn=evaluate_fn_for_server,  # Server-side evaluation
        on_fit_config_fn=fit_config,  # Function to configure client training rounds
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters), # Initial global model
    )

    # --- 6. Start Simulation ---
    print(f"Starting Federated Learning simulation for {NUM_ROUNDS} rounds with {NUM_CLIENTS} clients...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0} if DEVICE == torch.device("cpu") else {"num_cpus": 1, "num_gpus": 0.2}
        # Adjust client_resources based on your setup. num_gpus is per client.
        # If you have 1 GPU and 5 clients, 0.2 means each client gets 1/5th of GPU time (Flower handles this).
        # Be cautious with GPU resources in simulation; CPU might be easier to start.
    )

    print("Federated Learning simulation finished.")
    print("History (server-side evaluation metrics):", history.metrics_centralized)
    # history.losses_centralized will also contain the loss from evaluate_fn

if __name__ == "__main__":
    main()