import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict

from Model.model import FraudNet 

# Opacus imports
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(
    net: FraudNet, 
    trainloader: DataLoader, 
    epochs: int, 
    learning_rate: float,
    # DP parameters
    dp_target_epsilon: float,
    dp_target_delta: float,
    dp_max_grad_norm: float
):
    if len(trainloader.dataset) == 0:
        print(f"Client on {DEVICE} has no data to train on. Skipping DP training.")
        return 0, 0.0 # No examples trained, no epsilon spent

    # --- Model Validation for Opacus (Optional but good practice) ---
    # errors = ModuleValidator.validate(net, strict=False)
    # if errors:
    #     print(f"Opacus ModuleValidator found issues: {errors}")
    #     # net = ModuleValidator.fix(net) # Attempt to fix, or handle error
    #     # print("Attempted to fix model with ModuleValidator.")
    # For FraudNet, this should generally be fine.

    criterion = nn.BCEWithLogitsLoss()
    # Use a standard optimizer; PrivacyEngine will wrap it or create a DPOptimizer.
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # --- Initialize PrivacyEngine ---
    privacy_engine = PrivacyEngine(
        # secure_mode=False # Set to True for more secure random number generation if needed, can be slower
    ) 
    
    # Opacus modifies the model, optimizer, and dataloader
    # target_delta is crucial and should usually be << 1/dataset_size
    # If dataset size is very small, using a fixed small delta might be necessary,
    # but be aware of its privacy implications.
    # For make_private_with_epsilon, Opacus expects the dataloader to determine sample size.
    
    # Ensure trainloader is not empty, otherwise make_private_with_epsilon can fail
    if len(trainloader) == 0: # Checks number of batches
        print(f"Client on {DEVICE} has an empty trainloader (0 batches). Skipping DP training.")
        return 0, 0.0

    try:
        # net, optimizer, and trainloader are modified in-place by attach
        # or new ones are returned if you use make_private variants.
        # Let's re-assign to be clear, make_private_with_epsilon is convenient.
        
        # Note: Opacus's `make_private_with_epsilon` will effectively run the training
        # for the specified number of epochs by wrapping the dataloader.
        # The actual training loop happens within this call if you use it directly for training,
        # or you set it up and then run your own loop.
        # For finer control, we will attach and run our own loop.

        # Attach PrivacyEngine to the optimizer
        net.train() # Ensure model is in train mode before attaching
        model_dp, optimizer_dp, dataloader_dp = privacy_engine.make_private_with_epsilon(
            module=net,
            optimizer=optimizer,
            data_loader=trainloader,
            epochs=epochs, # Total number of epochs this PE will be used for this model/optimizer/loader config
            target_epsilon=dp_target_epsilon,
            target_delta=dp_target_delta,
            max_grad_norm=dp_max_grad_norm,
            # poisson_sampling=True # Recommended for DP guarantees with shuffling
        )
        
        # print(f"Client on {DEVICE}: Attached PrivacyEngine. Target Epsilon: {dp_target_epsilon}, Target Delta: {dp_target_delta}")

        # --- DP Training Loop ---
        for epoch in range(epochs): # This outer loop is for conceptual clarity; PE's dataloader_dp is epoch-aware
            epoch_loss = 0.0
            num_batches = 0
            for features, labels in dataloader_dp: # Use the Opacus dataloader
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                
                optimizer_dp.zero_grad() 
                outputs = model_dp(features) # Use the DP model
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_dp.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            # if num_batches > 0:
            #     print(f"Client on {DEVICE} - Epoch {epoch+1}/{epochs}, DP Loss: {epoch_loss/num_batches:.4f}")
            # else:
            #     print(f"Client on {DEVICE} - Epoch {epoch+1}/{epochs}, No batches processed.")

    except Exception as e:
        print(f"Client on {DEVICE}: Error during DP training: {e}")
        # Detach if error occurs to prevent issues with non-DP use later if model is reused
        if hasattr(privacy_engine, 'detach'):
             privacy_engine.detach()
        return 0, 0.0 # Indicate failure or no training

    spent_epsilon = 0.0
    try:
        spent_epsilon = privacy_engine.get_epsilon(delta=dp_target_delta)
        # print(f"Client on {DEVICE}: DP Training finished. Spent Epsilon: {spent_epsilon:.4f} (for delta={dp_target_delta})")
    except Exception as e:
        print(f"Client on {DEVICE}: Could not get epsilon. {e}")
    
    # Detach the privacy engine to return the model to a standard model
    # This is important if the model instance is reused across rounds without re-initialization
    # or if other non-DP operations are performed on it.
    if hasattr(privacy_engine, 'detach'): # Older Opacus versions might not have it on PE directly
        privacy_engine.detach()
    elif hasattr(model_dp, 'detach'): # Check if model was wrapped and has detach
        model_dp.detach()


    return len(trainloader.dataset), spent_epsilon


def test(net: FraudNet, testloader: DataLoader): # test function remains non-DP
    if len(testloader.dataset) == 0:
        return float('inf'), 0.0, 0 
        
    criterion = nn.BCEWithLogitsLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            
            predicted = torch.sigmoid(outputs) > 0.5 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = loss / len(testloader) if len(testloader) > 0 else float('inf')
    return avg_loss, accuracy, len(testloader.dataset)


class FraudDetectionClient(fl.client.NumPyClient):
    def __init__(self, model_fn, trainloader: DataLoader, valloader: DataLoader):
        self.model_fn = model_fn 
        self.model = self.model_fn().to(DEVICE) # Original non-DP model is stored
        self.trainloader = trainloader
        self.valloader = valloader
        # print(f"FraudDetectionClient initialized on {DEVICE}. Train batches: {len(trainloader)}, Val batches: {len(valloader)}.")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters) # Load global model parameters

        # Extract DP parameters from config
        epochs = config.get("local_epochs", 1)
        learning_rate = config.get("learning_rate", 0.001)
        
        dp_target_epsilon = config.get("dp_target_epsilon", 2.0) 
        # It's crucial that target_delta is small, e.g., < 1/dataset_size
        # For simplicity, we use a fixed delta from config.
        # A more robust approach might involve clients calculating their own delta
        # or the server providing a global delta based on overall participant numbers.
        # Opacus's make_private_with_epsilon uses len(dataloader) for sample rate,
        # so it needs a non-empty dataloader.
        num_train_samples = len(self.trainloader.dataset)
        dp_target_delta = config.get("dp_target_delta", 1e-5 if num_train_samples == 0 else 1.0 / (num_train_samples * 10) ) # Ensure delta is smaller than 1/N
        dp_target_delta = min(dp_target_delta, 1e-4) # Cap delta to a reasonable small value

        dp_max_grad_norm = config.get("dp_max_grad_norm", 1.0)

        num_examples_trained, spent_epsilon = train(
            self.model, self.trainloader, epochs, learning_rate,
            dp_target_epsilon, dp_target_delta, dp_max_grad_norm
        )
        
        fit_metrics = {"spent_epsilon": float(spent_epsilon) if spent_epsilon is not None else 0.0}
        
        updated_parameters = self.get_parameters(config={})
        return updated_parameters, num_examples_trained, fit_metrics

    def evaluate(self, parameters, config): # Evaluation is non-DP
        self.set_parameters(parameters)
        loss, accuracy, num_examples_eval = test(self.model, self.valloader)
        return float(loss), num_examples_eval, {"accuracy": float(accuracy)}

if __name__ == '__main__':
    print("client.py executed directly (for informational purposes only).")
    print(f"PyTorch will use: {DEVICE} for client-side operations if instantiated.")