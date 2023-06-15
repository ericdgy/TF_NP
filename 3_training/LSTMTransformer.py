import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, TensorDataset


class LSTMTransformer(nn.Module):
    def __init__(self, input_dim, lstm_units, num_layers, num_heads, dff, dropout=0.1):
        super(LSTMTransformer, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_units, batch_first=True)
        encoder_layers = TransformerEncoderLayer(lstm_units, num_heads, dff, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.linear = nn.Linear(lstm_units, 1)  # 预测的流量值只有一个

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.transpose(0, 1)  # 将LSTM输出reshape为适合Transformer输入的形式
        x = self.transformer_encoder(x)
        x = self.linear(x[-1])
        return x


# 实例化模型
model = LSTMTransformer(input_dim=1, lstm_units=512, num_layers=2, num_heads=8, dff=2048)

# 设置优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # L2正则化（权重衰减）
criterion = nn.MSELoss()  # 使用MSE作为损失函数

# 定义L1正则化
l1_lambda = 0.01

# 设定训练轮数
num_epochs = 50

# 假设你的输入数据在X中，目标数据在y中
dataset = TensorDataset(X, y)

# 创建数据加载器，设定批量大小为32，并在每个epoch之后打乱数据
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练模型
for epoch in range(num_epochs):
    for batch in dataloader:  # 假设dataloader是你的数据加载器
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 添加L1正则化
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm

        loss.backward()
        optimizer.step()
