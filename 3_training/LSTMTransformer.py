import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set the seed
my_seed = 2023
torch.manual_seed(my_seed)
np.random.seed(my_seed)

# Load data
data = pd.read_csv("isp.csv")
stand_scaler = MinMaxScaler()
all_data = stand_scaler.fit_transform(data["Internet traffic data (in bits)"].values.reshape(-1, 1))

sequence_len = 10
X = []
Y = []
for i in range(len(all_data) - sequence_len):
    X.append(all_data[i:i + sequence_len])
    Y.append(all_data[i + sequence_len])
X = np.array(X)
Y = np.array(Y)

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

# Convert the data to PyTorch tensors and move to GPU
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

# Create DataLoader for training
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

# Define the LSTM with Attention model using PyTorch
class LSTMAttentionModel(nn.Module):
    def __init__(self):
        super(LSTMAttentionModel, self).__init__()
        self.lstm1 = nn.LSTM(1, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 128, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8)
        self.fc = nn.Linear(128, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x[:, -1, :].unsqueeze(1))
        x = x.permute(1, 0, 2)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 0, 2)
        x = self.fc(x[:, -1, :])
        x = self.activation(x)
        return x

# Create the model
lstm_attention = LSTMAttentionModel()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_attention.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    pbar = tqdm(train_loader, total=len(train_loader))
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = lstm_attention(batch_x)
        loss = criterion(output.squeeze(), batch_y.squeeze())
        loss.backward()
        optimizer.step()
        pbar.set_description(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}')

lstm_attention.eval()

# Perform inference using test data
with torch.no_grad():
    X_test = X_test.view(-1, sequence_len, 1)
    Y_predict = lstm_attention(X_test).squeeze().cpu()
    Y_predict_real = stand_scaler.inverse_transform(Y_predict.detach().numpy().reshape(-1, 1))
    Y_test_real = stand_scaler.inverse_transform(Y_test.numpy().reshape(-1, 1))
    Y_test_real = Y_test_real[sequence_len:]

# Plot the results
plt.figure(figsize=(20, 5))
plt.plot(data["Time"].values[sequence_len:][:len(Y_test_real)], Y_test_real, label='Actual')
plt.plot(data["Time"].values[sequence_len:][:len(Y_predict_real)], Y_predict_real, label='Predictions')
plt.xlabel('Time')
plt.ylabel('Internet Traffic')
plt.legend()
plt.show()

def calculate_mape(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    mape = np.mean(diff / np.abs(y_true)) * 100
    return mape

def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

min_length = min(len(Y_test_real), len(Y_predict_real))
Y_test_real = Y_test_real[:min_length]
Y_predict_real = Y_predict_real[:min_length]

print(f"根均方误差(RMSE): {RMSE(Y_predict_real / (1024 * 1024), Y_test_real / (1024 * 1024))}")
print(f"平均绝对百分比误差(MAPE): {calculate_mape(Y_test_real, Y_predict_real)}")