import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt

player_data = "../../csv/player_data.csv"         # Path to your training dataset
player_fixtures = "../../csv/player_fixture.csv"  # Path to your inference dataset
prediction_csv = "../../csv/predictions.csv"

model_weights = "../../model/model_weights.pth"

class HybridRNN(nn.Module):
    def __init__(self, sequence_input_size, hidden_size, num_layers, metadata_size, output_size=1):
        super(HybridRNN, self).__init__()

        # RNN (LSTM in this case) for sequential data
        self.rnn = nn.LSTM(sequence_input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layers for constant metadata (id, cost, team)
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Final fully connected layer for prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 16, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, sequence_data, metadata):
        # Process sequence data through RNN
        rnn_out, _ = self.rnn(sequence_data)
        rnn_out = rnn_out[:, -1, :]  # Take the last hidden state for sequence data

        # Process metadata through fully connected layers
        metadata_out = self.metadata_fc(metadata)

        # Combine RNN output and metadata output
        combined = torch.cat((rnn_out, metadata_out), dim=1)

        # Final prediction
        output = self.fc(combined)
        return output

# Function to load and preprocess the dataset
def load_data(file_path):
    # Load data from CSV
    data = pd.read_csv(file_path)
    print(f"Raw data shape: {data.shape}")  # Check the number of rows and columns

    # Extract constant metadata fields (id, cost, team) and opponent, home/away
    metadata = data.iloc[:, [0, 1, 2, 15, 16]].values  # Columns 0-2 (id, cost, team) and 16-17 (opponent, home/away)

    # Extract features: Columns 3-15 (12 features per player)
    features = data.iloc[:, 3:15].values  # Columns 3-15 are the feature columns

    # Extract target: Column 18 (final score)
    targets = data.iloc[:, -1].values    # Column 18 is the final score

    print(f"Features shape: {features.shape}")
    print(f"Metadata shape: {metadata.shape}")
    print(f"Targets shape: {targets.shape}")

    # Normalize the features and metadata
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    metadata_scaler = StandardScaler()
    metadata = metadata_scaler.fit_transform(metadata)

    # Reshape the features into a format suitable for RNN: (batch_size, sequence_length, input_size)
    sequence_length = 4  # Number of tuples (5 tuples per player)
    input_size = 3  # Number of features per tuple (e.g., match rating, opponent, home/away)

    # Reshape features into (batch_size, sequence_length, input_size)
    reshaped_features = features.reshape(-1, sequence_length, input_size)

    print(f"Reshaped features shape: {reshaped_features.shape}")

    # Check consistency: ensure the number of samples is the same across all arrays
    assert reshaped_features.shape[0] == metadata.shape[0] == len(targets), "Shape mismatch"

    # Convert data to tensors
    X_features = torch.tensor(reshaped_features, dtype=torch.float32)
    X_metadata = torch.tensor(metadata, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)

    return X_features, X_metadata, y, scaler, metadata_scaler

# Function to create DataLoader
def create_dataloader(X_features, X_metadata, y, batch_size=32):

    print(f"X_features shape: {X_features.shape}")
    print(f"X_metadata shape: {X_metadata.shape}")
    print(f"y shape: {y.shape}")

    dataset = TensorDataset(X_features, X_metadata, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Training the model
def train_model(model, train_loader, optimizer, criterion, num_epochs=10):
    plt.ion()  # Turn on interactive mode for real-time plotting
    figure, ax = plt.subplots()
    epochs = []
    losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for sequence_data, metadata, targets in train_loader:
            sequence_data = sequence_data.to(device)
            metadata = metadata.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(sequence_data, metadata)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epochs.append(epoch + 1)
        losses.append(avg_loss)

        # Update the plot
        ax.clear()
        ax.plot(epochs, losses, label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss vs Epoch')
        ax.legend()
        plt.pause(0.1)  # Pause to update the plot

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    plt.ioff()  # Turn off interactive mode
    plt.close(figure)  # Close the plot window

# Function to load and preprocess the inference dataset
def load_inference_data(file_path, features_scaler, metadata_scaler):
    # Load data from CSV
    data = pd.read_csv(file_path)
    print(f"Inference data shape: {data.shape}")  # Check the number of rows and columns

    # Extract constant metadata fields (id, cost, team) and opponent, home/away
    metadata = data.iloc[:, [0, 1, 2, 15, 16]].values  # Columns 0-2 (id, cost, team) and 16-17 (opponent, home/away)

    # Extract features: Columns 3-15 (12 features per player)
    features = data.iloc[:, 3:15].values  # Columns 3-15 are the feature columns

    print(f"Inference features shape: {features.shape}")
    print(f"Inference metadata shape: {metadata.shape}")

    # Normalize the features and metadata using the same scalers as training data
    features = features_scaler.transform(features)
    metadata = metadata_scaler.transform(metadata)

    # Reshape the features into a format suitable for RNN: (batch_size, sequence_length, input_size)
    sequence_length = 4  # Number of tuples (5 tuples per player)
    input_size = 3  # Number of features per tuple (e.g., match rating, opponent, home/away)

    # Reshape features into (batch_size, sequence_length, input_size)
    reshaped_features = features.reshape(-1, sequence_length, input_size)

    print(f"Reshaped inference features shape: {reshaped_features.shape}")

    # Convert data to tensors
    X_features = torch.tensor(reshaped_features, dtype=torch.float32)
    X_metadata = torch.tensor(metadata, dtype=torch.float32)

    return X_features, X_metadata, data.iloc[:, 0].values  # Return player IDs

# Function to perform inference
def infer(model, X_features, X_metadata, player_ids):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(X_features)):
            sequence_data = X_features[i].unsqueeze(0).to(device)
            metadata = X_metadata[i].unsqueeze(0).to(device)
            output = model(sequence_data, metadata)
            predictions.append(output.cpu().numpy())

    # Create a DataFrame with player IDs and predictions
    predictions = np.array(predictions).flatten()
    result_df = pd.DataFrame({'Player ID': player_ids, 'Predicted Score': predictions})
    return result_df

# Function to sort players by predicted score
def sort_players_by_score(result_df):
    sorted_df = result_df.sort_values(by='Predicted Score', ascending=False)
    return sorted_df

X_features, X_metadata, y, features_scaler, metadata_scaler = load_data(player_data)

# Create a DataLoader for batch processing
train_loader = create_dataloader(X_features, X_metadata, y)

# Define device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
sequence_input_size = 3  # Number of features per tuple (e.g., match rating, opponent, home/away)
hidden_size = 64  # Hidden size for RNN
num_layers = 2  # Number of LSTM layers
metadata_size = 5  # Number of constant metadata features (id, cost, team)
output_size = 1  # Target score (single value)

# Initialize the model
model = HybridRNN(sequence_input_size, hidden_size, num_layers, metadata_size, output_size).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 500
train_model(model, train_loader, optimizer, criterion, num_epochs)

# Load model weights
#model.load_state_dict(torch.load(model_weights))
#print("Model weights loaded from {model_weights}")

# Load inference data
X_features_infer, X_metadata_infer, player_ids = load_inference_data(player_fixtures, features_scaler, metadata_scaler)

# Perform inference
result_df = infer(model, X_features_infer, X_metadata_infer, player_ids)

# Sort players by predicted score
sorted_result_df = sort_players_by_score(result_df)

# Save the sorted results to a CSV file
sorted_result_df.to_csv(prediction_csv, index=False)
print("Sorted predictions saved to {prediction_csv}")
