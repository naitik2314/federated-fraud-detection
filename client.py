import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict

from Model.model import FraudNet 

# Opacus imports
from opacus import PrivacyEngine
# from opacus.validators import ModuleValidator # Keep if you plan to re-enable detailed validation

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(
    net: FraudNet, 
    trainloader: DataLoader, 
    epochs: int, 
    learning_rate: float,
    # DP parameters
    dp_target_epsilon: float = None,
    dp_target_delta: float = None,
    dp_max_grad_norm: float = None
):
    if len(trainloader.dataset) == 0:
        # print(f"Client on {DEVICE} has no data to train on. Skipping training.")
        return 0, 0.0 
        
    # --- Calculate pos_weight for BCEWithLogitsLoss ---
    # This should be done before the PrivacyEngine potentially wraps/replaces the dataloader
    pos_weight_tensor = None
    if len(trainloader.dataset) > 0: 
        try:
            all_labels_list = []
            # Iterate through the original trainloader to get all labels
            # This is done once before training to determine class distribution
            for _, batch_labels in trainloader: 
                all_labels_list.append(batch_labels)
            
            if all_labels_list: 
                all_labels = torch.cat(all_labels_list).squeeze()
                if all_labels.dim() == 0: # Handle case where there's only one sample
                    all_labels = all_labels.unsqueeze(0)

                num_positives = torch.sum(all_labels == 1).item()
                num_negatives = torch.sum(all_labels == 0).item()
                
                # print(f"Client on {DEVICE}: Samples for pos_weight calc - Pos: {num_positives}, Neg: {num_negatives}")

                if num_positives > 0 and num_negatives > 0:
                    pos_weight_val = num_negatives / num_positives
                    # Cap pos_weight to prevent extreme values if num_positives is very small
                    # Example cap, can be tuned.
                    pos_weight_val = min(pos_weight_val, 100.0) 
                    pos_weight_tensor = torch.tensor([pos_weight_val], device=DEVICE)
                    print(f"Client on {DEVICE}: Using custom pos_weight: {pos_weight_val:.2f} ({num_negatives} neg / {num_positives} pos)")
                elif num_positives == 0 and num_negatives > 0:
                    print(f"Client on {DEVICE}: Only negative samples ({num_negatives}) found. Not using pos_weight.")
                elif num_negatives == 0 and num_positives > 0:
                    print(f"Client on {DEVICE}: Only positive samples ({num_positives}) found. Not using pos_weight (or consider a small weight for negatives).")
                # else: (both 0, handled by len(trainloader.dataset) == 0 already)
                #    print(f"Client on {DEVICE}: No samples or only one class with 0 samples. Not using pos_weight.")

        except Exception as e:
            print(f"Client on {DEVICE}: Error calculating pos_weight: {e}. Proceeding without custom pos_weight.")
            pos_weight_tensor = None

    if pos_weight_tensor is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()
    # --- End of pos_weight calculation ---

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    privacy_engine = None 
    model_dp = net 
    optimizer_dp = optimizer
    dataloader_dp = trainloader # This will be replaced by Opacus if DP is active

    if dp_target_epsilon is not None and dp_target_delta is not None and dp_max_grad_norm is not None:
        # print(f"Client on {DEVICE}: Initializing PrivacyEngine for DP training...")
        privacy_engine = PrivacyEngine()
        
        if len(trainloader) == 0: 
            print(f"Client on {DEVICE} has an empty trainloader (0 batches). Skipping DP setup.")
        else:
            try:
                # Note: Opacus's make_private_with_epsilon modifies the model, optimizer, and dataloader
                # It also handles iterating for `epochs` internally if you use its dataloader directly in a single loop.
                # For explicit epoch control as we have, we attach it, then loop.
                model_dp, optimizer_dp, dataloader_dp = privacy_engine.make_private_with_epsilon(
                    module=net, # The model passed to train()
                    optimizer=optimizer,
                    data_loader=trainloader, # The original trainloader
                    epochs=epochs, 
                    target_epsilon=dp_target_epsilon,
                    target_delta=dp_target_delta,
                    max_grad_norm=dp_max_grad_norm,
                )
                # print(f"Client on {DEVICE}: Attached PrivacyEngine.")
            except Exception as e:
                print(f"Client on {DEVICE}: Error initializing PrivacyEngine: {e}. DP training will be skipped for this client this round.")
                # If DP setup fails, we should not proceed with DP steps.
                # Depending on desired behavior, could fall back to non-DP or skip.
                # For now, if it fails, subsequent DP-specific calls will error or not work.
                # It's safer to just return if PE setup fails.
                return 0, 0.0 # Indicate no DP training happened
    
    model_dp.train() # Ensure model is in train mode
    for epoch in range(epochs):
        # Optional: per-epoch loss tracking
        # current_epoch_loss = 0.0
        # current_epoch_batches = 0
        for features, labels in dataloader_dp: # Use the Opacus-wrapped dataloader if DP is active
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            optimizer_dp.zero_grad() 
            outputs = model_dp(features) 
            loss = criterion(outputs, labels) # Criterion now potentially has pos_weight
            loss.backward()
            optimizer_dp.step()
            # current_epoch_loss += loss.item()
            # current_epoch_batches +=1
        # if current_epoch_batches > 0:
        #     print(f"Client on {DEVICE} Epoch {epoch+1}/{epochs} - Avg Loss: {current_epoch_loss/current_epoch_batches:.4f}")


    spent_epsilon = 0.0
    if privacy_engine and dp_target_epsilon is not None: 
        try:
            spent_epsilon = privacy_engine.get_epsilon(delta=dp_target_delta)
            # print(f"Client on {DEVICE}: DP Training finished. Spent Epsilon: {spent_epsilon:.4f}")
        except Exception as e:
            print(f"Client on {DEVICE}: Could not get epsilon. {e}")
        
        if hasattr(privacy_engine, 'detach'):
            privacy_engine.detach()
        # Opacus >= 1.0, model_dp is the original model instance, so detaching PE is enough
        # elif hasattr(model_dp, 'detach'): 
        #    model_dp.detach() 
             
    return len(trainloader.dataset), spent_epsilon


def test(net: nn.Module, testloader: DataLoader, device_to_use: torch.device = DEVICE):
    """Evaluates the model and returns loss, detailed metrics, and number of samples."""
    if len(testloader.dataset) == 0:
        metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
                   "true_positives": 0, "false_positives": 0, 
                   "true_negatives": 0, "false_negatives": 0}
        return float('inf'), metrics, 0 
            
    criterion = nn.BCEWithLogitsLoss() # Standard loss for evaluation
    loss_sum = 0.0 
    
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
        
        dp_target_epsilon = config.get("dp_target_epsilon", None) 
        dp_target_delta = config.get("dp_target_delta", None)
        dp_max_grad_norm = config.get("dp_max_grad_norm", None)

        # Refine delta if DP is active and num_train_samples > 0
        # This calculation of dp_target_delta in client's fit method is important
        if dp_target_epsilon is not None:
            num_train_samples = len(self.trainloader.dataset)
            if num_train_samples > 0:
                # Ensure delta is smaller than 1/N for this client's dataset
                # Using config.get("dp_target_delta", 1e-5) as a base if provided by server
                # This logic makes client delta adaptive if not explicitly fixed very low by server
                base_delta_from_config = config.get("dp_target_delta", 1e-5) # Default from config
                # calculated_delta = 1.0 / (num_train_samples) # Strictest interpretation
                # More common to use a slightly relaxed version or fixed small value:
                # dp_target_delta = min(base_delta_from_config, 1.0 / num_train_samples if num_train_samples > 0 else base_delta_from_config)
                # For simplicity and safety, often a fixed small delta from server config is used, and client ensures it's applied
                # Here, we'll just use the server provided or default if server doesn't provide.
                # The client.py's train() function will use this dp_target_delta
                pass # dp_target_delta is already set from config or default None
            elif dp_target_delta is None: # If no samples and server didn't set delta for DP
                 dp_target_delta = 1e-5 # Fallback default if DP active but no samples to derive from


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