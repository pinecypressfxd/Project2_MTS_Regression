import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Generate sample data (same as previous examples)
X1 = torch.randn(100, 5)
X2 = torch.randn(80, 3)
X3 = torch.randn(100, 1)
y = torch.randn(100, 1)

# Align X1 and X2 with the common time index using interpolation (same as previous examples)
from scipy.interpolate import interp1d
f1 = interp1d(np.arange(X1.shape[0]), X1.numpy(), axis=0)
f2 = interp1d(np.arange(X2.shape[0]), X2.numpy(), axis=0)
X1 = torch.from_numpy(f1(time_index))
X2 = torch.from_numpy(f2(time_index))

# Concatenate all features (same as previous examples)
X = torch.cat((X1, X2, X3), dim=1)

# Split data into train and test sets (same as previous examples)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create DataLoader for batch processing (same as previous examples)
train_dataset = TensorDataset(X_train.unsqueeze(2), y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, num_inputs, num_outputs, d_model, num_heads, num_layers, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(num_inputs, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, dropout=dropout_rate),
            num_layers
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(d_model, num_outputs)

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.transformer_encoder(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Create an instance of the model
model = TransformerModel(X_train.shape[1], 1, d_model=32, num_heads=4, num_layers=2)

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
