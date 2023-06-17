import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set the seed
my_seed = 2023
torch.manual_seed(my_seed)
np.random.seed(my_seed)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
data = pd.read_csv("isp.csv")
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
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device) #去掉unsqueeze
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device) #去掉unsqueeze

# Define the Transformer model using PyTorch
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.input_dim = 1
        self.model_dim = 10  # ensure model_dim is divisible by nhead
        self.nhead = 10
        self.num_layers = 6
        self.fc_in = nn.Linear(self.input_dim, self.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.fc_out = nn.Linear(self.model_dim * sequence_len, 1)

    def forward(self, x):
        #x = x.view(x.size(0), -1, 1)
        x = self.fc_in(x)  # transform input dimension to model dimension
        x = self.transformer_encoder(x)
        x = self.fc_out(x.view(x.size(0), -1))
        return x

# Create the model and move to GPU
transformer = TransformerModel().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(transformer.parameters(), lr=0.001)

# Training loop
num_epochs = 50
batch_size = 32  # Choose a batch size that fits in your memory
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for i, (x_batch, y_batch) in enumerate(train_loader):
        # Reshape the input data
        x_batch = x_batch.view(x_batch.size(0), -1, 1)
        optimizer.zero_grad()
        output = transformer(x_batch)
        loss = criterion(output.view(-1, 1), y_batch.view(-1, 1)) # 调整输出和标签到正确的维度
        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Make predictions on the test set
transformer.eval()

with torch.no_grad():
    X_test = X_test.view(-1, sequence_len, 1).to(device)
    Y_predict = transformer(X_test).squeeze().cpu()
    Y_predict_real = stand_scaler.inverse_transform(Y_predict.squeeze().detach().numpy().reshape(-1, 1))
    Y_test_real = stand_scaler.inverse_transform(Y_test.cpu().numpy().reshape(-1, 1))

# Save the model parameters
torch.save(transformer.state_dict(), '../4_prediction/uk_transformer_py.pth')

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

print(f"Root Mean Squared Error (RMSE): {RMSE(Y_predict_real/(1024 * 1024), Y_test_real/(1024 * 1024))}")
print(f"Mean Absolute Percentage Error (MAPE): {MAPE(Y_predict_real, Y_test_real)}")