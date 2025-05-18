import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict

from Model.model import FraudNet 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# This print statement will appear for each client actor if not guarded by a main check
# print(f"Client-side components loaded. Training will be on {DEVICE}") 

def train(net: FraudNet, trainloader: DataLoader, epochs: int, learning_rate: float):
    if len(trainloader.dataset) == 0:
        # print(f"Client has no data to train on. Skipping training.")
        return 0 # No examples trained
        
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

def test(net: FraudNet, testloader: DataLoader):
    if len(testloader.dataset) == 0:
        # print("Client has no data to evaluate on.")
        return float('inf'), 0.0, 0 # Infinite loss, 0 accuracy, 0 examples
        
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
        self.model = self.model_fn().to(DEVICE)
        self.trainloader = trainloader
        self.valloader = valloader 
        # Guard the print statement to avoid repetition from Ray actors
        # print(f"FraudDetectionClient initialized. Train batches: {len(trainloader)}, Val batches: {len(valloader)}. Device: {DEVICE}")


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

        num_examples_trained = train(self.model, self.trainloader, epochs=epochs, learning_rate=learning_rate)
        
        updated_parameters = self.get_parameters(config={})
        # If num_examples_trained is 0, Flower might have issues with weighted averaging.
        # It's better if clients with 0 data don't participate or are handled by strategy.
        return updated_parameters, num_examples_trained, {} 

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, num_examples_eval = test(self.model, self.valloader)
        return float(loss), num_examples_eval, {"accuracy": float(accuracy)}

if __name__ == '__main__':
    print("client.py executed directly (for logging purposes only).")
    print(f"PyTorch will use: {DEVICE} for client-side operations if instantiated.")
    print("Contains: FraudDetectionClient, train(), test()")