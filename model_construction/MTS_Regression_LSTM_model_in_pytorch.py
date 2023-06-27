import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Generate sample data (same as the Keras example)
X1 = torch.randn(100, 5)
X2 = torch.randn(80, 3)
X3 = torch.randn(100, 1)
y = torch.randn(100, 1)

# Align X1 and X2 with the common time index using interpolation (same as the Keras example)
from scipy.interpolate import interp1d
f1 = interp1d(np.arange(X1.shape[0]), X1.numpy(), axis=0)
f2 = interp1d(np.arange(X2.shape[0]), X2.numpy(), axis=0)
X1 = torch.from_numpy(f1(time_index))
X2 = torch.from_numpy(f2(time_index))

# Concatenate all features (same as the Keras example)
X = torch.cat((X1, X2, X3), dim=1)

# Split data into train and test sets (same as the Keras example)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train.unsqueeze(2), y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out

model = LSTMModel(X_train.shape[1], 64, 1)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

# Evaluate the model
with torch.no_grad():
    test_outputs = model(X_test.unsqueeze(2).float())
    test_loss = criterion(test_outputs, y_test.float())
    print('Test loss:', test_loss.item())

# Make predictions
with torch.no_grad():
    predictions = model(X_test.unsqueeze(2).float())
