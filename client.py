import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict

from Model.model import FraudNet 

# Opacus imports (if DP is still intended to be used later, keep them)
from opacus import PrivacyEngine
# from opacus.validators import ModuleValidator # Keep if you plan to re-enable detailed validation

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(
    net: FraudNet, 
    trainloader: DataLoader, 
    epochs: int, 
    learning_rate: float,
    # DP parameters (keep if you plan to re-enable DP easily)
    dp_target_epsilon: float = None, # Default to None if DP is optional
    dp_target_delta: float = None,
    dp_max_grad_norm: float = None
):
    if len(trainloader.dataset) == 0:
        # print(f"Client on {DEVICE} has no data to train on. Skipping training.")
        return 0, 0.0 # No examples trained, no epsilon spent (if DP were active)
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    privacy_engine = None # Initialize to None
    model_dp = net # By default, use the passed net
    optimizer_dp = optimizer
    dataloader_dp = trainloader

    if dp_target_epsilon is not None and dp_target_delta is not None and dp_max_grad_norm is not None:
        # print(f"Client on {DEVICE}: Initializing PrivacyEngine for DP training...")
        privacy_engine = PrivacyEngine()
        
        # Ensure trainloader is not empty for Opacus
        if len(trainloader) == 0: # Checks number of batches
            print(f"Client on {DEVICE} has an empty trainloader (0 batches). Skipping DP setup.")
            # Fall back to non-DP training or return error? For now, let it try non-DP
        else:
            try:
                model_dp, optimizer_dp, dataloader_dp = privacy_engine.make_private_with_epsilon(
                    module=net,
                    optimizer=optimizer,
                    data_loader=trainloader,
                    epochs=epochs,
                    target_epsilon=dp_target_epsilon,
                    target_delta=dp_target_delta,
                    max_grad_norm=dp_max_grad_norm,
                )
                # print(f"Client on {DEVICE}: Attached PrivacyEngine.")
            except Exception as e:
                print(f"Client on {DEVICE}: Error initializing PrivacyEngine: {e}. Proceeding with non-DP training for this client.")
                privacy_engine = None # Ensure it's None if setup failed
                model_dp = net 
                optimizer_dp = optimizer
                dataloader_dp = trainloader
    
    model_dp.train()
    for epoch in range(epochs):
        # epoch_loss = 0.0
        # num_batches = 0
        for features, labels in dataloader_dp: 
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            optimizer_dp.zero_grad() 
            outputs = model_dp(features) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_dp.step()
            
            # epoch_loss += loss.item()
            # num_batches += 1
        # if num_batches > 0 and dp_target_epsilon is not None and privacy_engine: # Only print DP loss if DP active
        #     print(f"Client on {DEVICE} - Epoch {epoch+1}/{epochs}, DP Loss: {epoch_loss/num_batches:.4f}")
        # elif num_batches > 0 :
        #     print(f"Client on {DEVICE} - Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_batches:.4f}")


    spent_epsilon = 0.0
    if privacy_engine and dp_target_epsilon is not None: # If DP was active
        try:
            spent_epsilon = privacy_engine.get_epsilon(delta=dp_target_delta)
            # print(f"Client on {DEVICE}: DP Training finished. Spent Epsilon: {spent_epsilon:.4f}")
        except Exception as e:
            print(f"Client on {DEVICE}: Could not get epsilon. {e}")
        
        # Detach the privacy engine
        if hasattr(privacy_engine, 'detach'):
            privacy_engine.detach()
        elif hasattr(model_dp, 'detach'): # Check if model was wrapped and has detach
             model_dp.detach()
             
    return len(trainloader.dataset), spent_epsilon


def test(net: nn.Module, testloader: DataLoader, device_to_use: torch.device = DEVICE):
    """Evaluates the model and returns loss, detailed metrics, and number of samples."""
    if len(testloader.dataset) == 0:
        # Return structure consistent with expected metrics
        metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
                   "true_positives": 0, "false_positives": 0, 
                   "true_negatives": 0, "false_negatives": 0}
        return float('inf'), metrics, 0 
            
    criterion = nn.BCEWithLogitsLoss()
    loss_sum = 0.0 # Use loss_sum to accumulate loss correctly
    
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    total_samples = 0

    net.eval()
    net.to(device_to_use) 
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device_to_use), labels.to(device_to_use)
            outputs = net(features)
            # Sum loss for averaging later, weighted by batch size
            loss_sum += criterion(outputs, labels).item() * features.size(0) 
            
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float() 
            
            total_samples += labels.size(0)

            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
    
    avg_loss = loss_sum / total_samples if total_samples > 0 else float('inf')
    accuracy = (true_positives + true_negatives) / total_samples if total_samples > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives
    }
    
    return avg_loss, metrics, total_samples


class FraudDetectionClient(fl.client.NumPyClient):
    def __init__(self, model_fn, trainloader: DataLoader, valloader: DataLoader):
        self.model_fn = model_fn 
        self.model = self.model_fn().to(DEVICE)
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters) 

        epochs = config.get("local_epochs", 1)
        learning_rate = config.get("learning_rate", 0.001)
        
        # DP parameters from config (will be None if DP_ENABLED is False in run_simulation.py)
        dp_target_epsilon = config.get("dp_target_epsilon", None) 
        dp_target_delta = config.get("dp_target_delta", None)
        dp_max_grad_norm = config.get("dp_max_grad_norm", None)

        # Refine delta if DP is active and num_train_samples > 0
        if dp_target_epsilon is not None: # Check if DP is meant to be active
            num_train_samples = len(self.trainloader.dataset)
            if num_train_samples > 0 :
                 # Ensure delta is smaller than 1/N for this client's dataset
                calculated_delta = 1.0 / (num_train_samples * 10) 
                # Use the smaller of the configured delta or calculated, capping at a reasonable max
                dp_target_delta = min(config.get("dp_target_delta", 1e-5), calculated_delta, 1e-4)
            else: # If no samples, DP train won't run, but set a default delta from config
                dp_target_delta = config.get("dp_target_delta", 1e-5)


        num_examples_trained, spent_epsilon = train(
            self.model, self.trainloader, epochs, learning_rate,
            dp_target_epsilon, dp_target_delta, dp_max_grad_norm
        )
        
        fit_metrics = {"spent_epsilon": float(spent_epsilon) if spent_epsilon is not None else 0.0}
        
        updated_parameters = self.get_parameters(config={})
        return updated_parameters, num_examples_trained, fit_metrics

    def evaluate(self, parameters, config): 
        self.set_parameters(parameters)
        avg_loss, metrics_dict, num_examples_eval = test(self.model, self.valloader, DEVICE)
        return float(avg_loss), num_examples_eval, metrics_dict

if __name__ == '__main__':
    print("client.py executed directly (for informational purposes only).")
    print(f"PyTorch will use: {DEVICE} for client-side operations if instantiated.")