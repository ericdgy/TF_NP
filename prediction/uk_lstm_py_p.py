import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.rcParams['font.family'] = 'SimHei'

# Set the seed
my_seed = 2023
torch.manual_seed(my_seed)
np.random.seed(my_seed)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
torch.cuda.manual_seed(my_seed)

# Load data
# 这里你需要加载你的预测数据，替换'all_data'
all_data = np.fromfile("ec_data")
stand_scaler = MinMaxScaler()
all_data = stand_scaler.fit_transform(all_data.reshape(-1, 1))

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
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)


# Define the LSTM model using PyTorch
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(1, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x[:, -1, :].unsqueeze(1))
        x = self.fc(x[:, -1, :])
        x = self.activation(x)
        return x


# Create the model and move to GPU
lstm = LSTMModel().to(device)

# Load the model parameters
lstm.load_state_dict(torch.load("uk_lstm_py.pth"))

# Set the model to evaluation mode
lstm.eval()

# Make predictions
with torch.no_grad():
    X_test = X_test.view(-1, sequence_len, 1).to(device)
    Y_predict = lstm(X_test).squeeze().cpu()
    Y_predict_real = stand_scaler.inverse_transform(Y_predict.squeeze().numpy().reshape(-1, 1))
    Y_test_real = stand_scaler.inverse_transform(Y_test.cpu().numpy().reshape(-1, 1))


# Plot the results
plt.figure(figsize=(20, 2))
plt.plot(Y_predict_real / (1024 * 1024))
plt.plot(Y_test_real / (1024 * 1024))
plt.show()
