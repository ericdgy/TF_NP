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

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
torch.cuda.manual_seed(my_seed)

# Load data
data = pd.read_csv("isp.csv", parse_dates=["Time"])
stand_scaler = MinMaxScaler()
all_data = stand_scaler.fit_transform(data["Internet traffic data (in bits)"].values.reshape(-1, 1))

# Smooth data using moving average
window_size = 5
smoothed_data = pd.Series(all_data.squeeze()).rolling(window=window_size).mean().fillna(0).values.reshape(-1, 1)

# Normalize data using Min-Max scaler
normalized_data = stand_scaler.fit_transform(smoothed_data)

sequence_len = 10
X = []
Y = []
for i in range(len(normalized_data) - sequence_len):
    X.append(normalized_data[i:i + sequence_len])
    Y.append(normalized_data[i + sequence_len])
X = np.array(X)
Y = np.array(Y)

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

# Convert the data to PyTorch tensors and move to GPU
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

# Create DataLoader for training
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
# Define the Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x

# Define the LSTM model using PyTorch with Transformer Encoder
# Define the LSTM model using PyTorch with Transformer Encoder
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(1, 512, batch_first=True)
        self.transformer_encoder = TransformerEncoder(d_model=512, nhead=8, num_layers=2)
        self.lstm2 = nn.LSTM(512, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.transformer_encoder(x)
        x, _ = self.lstm2(x)
        x = self.fc(x[:, -1, :])
        x = self.activation(x)
        return x
# Create the model and move to GPU
lstm = LSTMModel().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    pbar = tqdm(train_loader, total=len(train_loader))
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = lstm(batch_x)
        loss = criterion(output.squeeze(), batch_y.squeeze())
        loss.backward()
        optimizer.step()
        pbar.set_description(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}')

lstm.eval()

# Perform inference using test data
with torch.no_grad():
    X_test = X_test.view(-1, sequence_len, 1)
    Y_predict = lstm(X_test).squeeze().cpu()
    Y_predict_real = stand_scaler.inverse_transform(Y_predict.detach().numpy().reshape(-1, 1))
    Y_test_real = stand_scaler.inverse_transform(Y_test.cpu().numpy().reshape(-1, 1))
# Plot the results
plt.figure(figsize=(20, 5))
x_ticks = range(sequence_len, sequence_len + len(Y_test_real))
plt.plot(x_ticks, Y_test_real, label='Actual')
plt.plot(x_ticks, Y_predict_real, label='Predictions')
plt.xlabel('Data Point')
plt.ylabel('Internet Traffic')
plt.legend()
plt.show()

# Evaluation metrics
def MAPE(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)

def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

print(f"根均方误差(RMSE): {RMSE(Y_predict_real / (1024 * 1024), Y_test_real / (1024 * 1024))}")
print(f"平均绝对百分比误差(MAPE): {MAPE(Y_predict_real, Y_test_real)}")