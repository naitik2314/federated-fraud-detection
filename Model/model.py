import torch
import torch.nn as nn
import torch.nn.functional as F

class FraudNet(nn.Module):
    def __init__(self, num_features):
        """
        Initializes the neural network model.
        
        Args:
            num_features (int): The number of input features for the model.
                                This corresponds to the number of columns in X_train.
        """
        super(FraudNet, self).__init__()
        # Define the layers
        # Layer 1: Input layer (num_features) -> Hidden layer 1 (e.g., 32 neurons)
        self.fc1 = nn.Linear(num_features, 32)
        # Layer 2: Hidden layer 1 (32 neurons) -> Hidden layer 2 (e.g., 16 neurons)
        self.fc2 = nn.Linear(32, 16)
        # Layer 3: Hidden layer 2 (16 neurons) -> Output layer (1 neuron for binary classification)
        self.fc3 = nn.Linear(16, 1)
        
        # Define dropout layer for regularization (optional, but good practice)
        self.dropout = nn.Dropout(0.2) # 20% dropout rate

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The output of the model (logits).
        """
        # Apply ReLU activation function after the first hidden layer
        x = F.relu(self.fc1(x))
        # Apply dropout
        x = self.dropout(x)
        # Apply ReLU activation function after the second hidden layer
        x = F.relu(self.fc2(x))
        # Apply dropout
        x = self.dropout(x)
        # Output layer (logits - no activation here as BCEWithLogitsLoss will handle it)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    # Example usage:
    # From our data_utils.py, we know the number of features is 29
    NUM_FEATURES = 29 
    
    # Instantiate the model
    model = FraudNet(num_features=NUM_FEATURES)
    print("Model architecture:")
    print(model)
    
    # Create a dummy input tensor to test the forward pass
    # Batch size of 10, 29 features
    dummy_input = torch.randn(10, NUM_FEATURES) 
    print(f"\nShape of dummy input: {dummy_input.shape}")
    
    # Get the model output
    output = model(dummy_input)
    print(f"Shape of model output: {output.shape}")
    print(f"Sample output (logits):\n{output[:5]}") # Print first 5 output logits