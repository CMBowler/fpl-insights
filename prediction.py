import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Example dataset (replace with actual data)
# Each row: [cost, form1, ..., form8, diff1, ..., diff8, next_diff], target: performance
data = np.random.rand(1000, 18)  # 1000 samples, 17 inputs + 1 target
X = data[:, :-1]  # Features (cost, forms, difficulties)
y = data[:, -1]   # Target (performance)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define the RNN-based neural network
class PlayerPerformanceRNN(nn.Module):
    def __init__(self, input_dim=2, rnn_hidden_size=32, fc_hidden_size=64, seq_length=8):
        super(PlayerPerformanceRNN, self).__init__()
        
        # RNN to process sequential data (form and difficulty ratings)
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=rnn_hidden_size, batch_first=True)
        
        # Fully connected layers for combining RNN output and non-sequential features
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden_size + 2, fc_hidden_size),  # +2 for cost and next_diff
            nn.ReLU(),
            nn.Linear(fc_hidden_size, 1)  # Output: performance
        )

    def forward(self, x):
        # Split inputs
        cost = x[:, 0:1]  # Cost: shape (batch_size, 1)
        next_diff = x[:, -1:]  # Next team difficulty: shape (batch_size, 1)
        seq_features = x[:, 1:-1].view(-1, 8, 2)  # Sequential data: shape (batch_size, seq_length, 2)
        
        # Pass sequential data through RNN
        _, (hidden_state, _) = self.rnn(seq_features)  # hidden_state: shape (1, batch_size, rnn_hidden_size)
        hidden_state = hidden_state.squeeze(0)  # Remove extra dimension: shape (batch_size, rnn_hidden_size)
        
        # Concatenate RNN output with cost and next_diff
        combined_features = torch.cat([hidden_state, cost, next_diff], dim=1)
        
        # Pass through fully connected layers
        output = self.fc(combined_features)
        return output

# Initialize the model, loss function, and optimizer
model = PlayerPerformanceRNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")
