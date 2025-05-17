import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict

from Model.model import FraudNet # Make sure your model.py file is in Model folder in the root directory of this project, or modify this accordingly

# Determine if a GPU is available and set the device accordingly
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Client-side training on {DEVICE}") # Clarified print statement

# Helper function for training the model on a client
def train(net: FraudNet, trainloader: DataLoader, epochs: int, learning_rate: float):
    """Train the network on the training set."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for features, labels in trainloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
    return len(trainloader.dataset)

# Helper function for evaluating the model on a client
def test(net: FraudNet, testloader: DataLoader):
    """Evaluate the network on the test set."""
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
            
    accuracy = correct / total if total > 0 else 0.0 # Avoid division by zero
    avg_loss = loss / len(testloader) if len(testloader) > 0 else float('inf') # Avoid division by zero
    return avg_loss, accuracy, len(testloader.dataset)


# Define the Flower client
class FraudDetectionClient(fl.client.NumPyClient):
    def __init__(self, model_fn, trainloader: DataLoader, valloader: DataLoader):
        # model_fn is a function that returns an instance of FraudNet
        self.model_fn = model_fn 
        self.model = self.model_fn().to(DEVICE) # Instantiate and move model to device
        self.trainloader = trainloader
        self.valloader = valloader 

    def get_parameters(self, config):
        """Return model parameters as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Update local model parameters with parameters received from the server."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = config.get("local_epochs", 1)
        learning_rate = config.get("learning_rate", 0.001)

        num_examples_trained = train(self.model, self.trainloader, epochs=epochs, learning_rate=learning_rate)
        
        updated_parameters = self.get_parameters(config={})
        return updated_parameters, num_examples_trained, {} 

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, num_examples_eval = test(self.model, self.valloader)
        return float(loss), num_examples_eval, {"accuracy": float(accuracy)}

if __name__ == '__main__':
    print("Client.py loaded. Contains FraudDetectionClient and train/test helper functions.")
    print(f"PyTorch will use: {DEVICE}")