import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data
def load_data(file_path):
    print("Loading data from CSV...")
    data = pd.read_csv(file_path)
    print(f"Data loaded successfully. Number of records: {len(data)}")

    # Extract metadata and features
    print("Separating metadata and features...")
    metadata = data.iloc[:, :3].values  # First 3 columns: id, cost, team
    features = data.iloc[:, 3:-1].values  # Triplet data (all but the last column)
    targets = data.iloc[:, -1].values    # Final column: target score
    print("Metadata and features separated.")

    # Extract and append opponent and home/away values from the last triplet
    print("Appending opponent and home/away values to metadata...")
    num_samples = len(data)
    sequence_length = 5
    input_size = 3

    # Calculate the indices of the last triplet's opponent and home/away columns
    opponent_index = sequence_length * input_size - 2
    home_away_index = sequence_length * input_size - 1

    # Extract the last tuple's opponent and home/away values
    last_opponent = features[:, opponent_index]
    last_home_away = features[:, home_away_index]

    # Append to metadata
    metadata = np.hstack((metadata, last_opponent.reshape(-1, 1), last_home_away.reshape(-1, 1)))
    print(f"Updated metadata shape: {metadata.shape}")

    # Remove the last tuple's opponent and home/away from features
    print("Removing last opponent and home/away values from features...")
    features = np.delete(features, [opponent_index, home_away_index], axis=1)

    # Normalize features (triplet data only)
    print("Normalizing triplet features...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    print("Normalization complete.")

    # Reshape triplet features for RNN
    print("Reshaping features for RNN...")
    if features.shape[1] % (sequence_length * input_size - 2) != 0:
        print("Error: The number of triplet values is not divisible by the adjusted sequence length.")
        return None

    reshaped_features = features.reshape(-1, sequence_length, input_size - 2)
    reshaped_targets = targets[sequence_length - 1::sequence_length]  # Use valid y values
    print(f"Features reshaped to {reshaped_features.shape}. Targets reshaped to {len(reshaped_targets)}.")

    # Convert to tensors
    print("Converting to tensors...")
    X_train, X_test, y_train, y_test, M_train, M_test = train_test_split(
        reshaped_features, reshaped_targets, metadata, test_size=0.2, random_state=42
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        torch.tensor(M_train, dtype=torch.float32),
        torch.tensor(M_test, dtype=torch.float32)
    )

# Define the RNN Model
class PlayerPerformancePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, metadata_size, output_size=1):
        super(PlayerPerformancePredictor, self).__init__()
        # RNN for sequential data
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Dense network for metadata
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Final fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 16, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, sequence_data, metadata):
        # Process sequence data through RNN
        rnn_out, _ = self.rnn(sequence_data)
        rnn_out = rnn_out[:, -1, :]  # Take the last hidden state
        
        # Process metadata through dense network
        metadata_out = self.metadata_fc(metadata)
        
        # Concatenate RNN and metadata outputs
        combined = torch.cat((rnn_out, metadata_out), dim=1)
        
        # Final prediction
        output = self.fc(combined)
        return output

# Training function
def train_model(model, train_loader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for sequence_data, metadata, targets in train_loader:

            optimizer.zero_grad()
            outputs = model(sequence_data, metadata)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation function
def evaluate_model(model, criterion, test_loader):
    print("Evaluating model on test data...")
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(test_loader):.4f}")

# Main script
if __name__ == "__main__":
    print("Starting script...")
    # Load the data
    file_path = "player_data.csv"
    X_train, X_test, y_train, y_test, M_train, M_test = load_data(file_path)
    
    # Reshape data for RNN (batch_size, sequence_length, input_size)
    print("Reshaping data for RNN...")
    sequence_length = 5
    input_size = 3  # Match rating, opponent team, home/away
    X_train = X_train.view(-1, sequence_length, input_size)
    X_test = X_test.view(-1, sequence_length, input_size)
    print("Reshaping complete.")

    # Create data loaders
    print("Creating data loaders...")
    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X_train, y_train, M_train),
        batch_size=32,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X_test, y_test, M_train),
        batch_size=32,
        shuffle=False
    )
    print("Data loaders ready.")

    # Define the model, criterion, and optimizer
    print("Initializing model...")
    hidden_size = 64
    output_size = 1
    num_layers = 1
    model = PlayerPerformanceRNN(input_size, hidden_size, output_size, num_layers)
    print("Model initialized.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Training the model...")
    num_epochs = 100
    train_model(model, criterion, optimizer, train_loader, num_epochs)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, criterion, test_loader)
    print("Script complete.")
