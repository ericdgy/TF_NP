import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set the seed
my_seed = 2023
torch.manual_seed(my_seed)
np.random.seed(my_seed)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
all_data = np.fromfile("uk_data")
stand_scaler = MinMaxScaler()
all_data = stand_scaler.fit_transform(all_data.reshape(-1, 1))

sequence_len = 10
X = []
Y = []
for i in range(len(all_data) - sequence_len - 1):
    X.append(all_data[i:i + sequence_len])
    Y.append(all_data[i + sequence_len])
X = np.array(X)
Y = np.array(Y)

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Convert the data to PyTorch tensors and move to GPU
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

class LSTMTransformerModel(nn.Module):
    def __init__(self):
        super(LSTMTransformerModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=100, num_layers=3, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=5)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])
        return x

# Create the model and move to GPU
lstm_transformer = LSTMTransformerModel().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_transformer.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Training loop
lstm_transformer.train()  # Switch to training mode
num_epochs = 100
batch_size = 32
patience = 10  # early stopping patience; how long to wait after last time validation loss improved.
best_loss = np.inf
stop_counter = 0

train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.view(x_batch.size(0), -1, 1)
        optimizer.zero_grad()
        output = lstm_transformer(x_batch)
        loss = criterion(output.view(-1), y_batch.view(-1))
        loss.backward()
        optimizer.step()

    scheduler.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Check early stopping condition
    if loss < best_loss:
        best_loss = loss
        stop_counter = 0
    else:
        stop_counter += 1
    if stop_counter >= patience:
        print("Early stopping!")
        break

# Switch to evaluation mode
lstm_transformer.eval()
with torch.no_grad():
    y_train_pred = lstm_transformer(X_train.view(X_train.size(0), -1, 1))
    y_test_pred = lstm_transformer(X_test.view(X_test.size(0), -1, 1))

# Plot
plt.figure(figsize=(20, 2))
plt.plot(Y_test.cpu().numpy(), label='True')
plt.plot(y_test_pred.view(-1).cpu().numpy(), label='Predicted')
plt.legend()
plt.show()

# Calculation of RMSE and MAPE
def RMSE(y_true, y_pred):
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

train_rmse = RMSE(Y_train.cpu().numpy(), y_train_pred.view(-1).cpu().numpy())
test_rmse = RMSE(Y_test.cpu().numpy(), y_test_pred.view(-1).cpu().numpy())

train_mape = MAPE(Y_train.cpu().numpy(), y_train_pred.view(-1).cpu().numpy())
test_mape = MAPE(Y_test.cpu().numpy(), y_test_pred.view(-1).cpu().numpy())

print(f'Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}')
print(f'Train MAPE: {train_mape:.4f}, Test MAPE: {test_mape:.4f}')

