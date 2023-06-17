# TF_NP
Network performance prediction, Transfer Learning
# Data processing

# LSTM
50 epoch  
根均方误差(RMSE): 0.00013777947449586337  
平均绝对百分比误差(MAPE): 0.023309320211410522
![uk_lstm.png](3_training%2Fuk_lstm.png)
# Transformer
只使用 Transformer 的 Encoder 部分  
50 epoch  
根均方误差(RMSE): 0.0001601030546610923  
平均绝对百分比误差(MAPE): 0.030823806300759315
![uk_transformer.png](3_training%2Fuk_transformer.png)
# LSTM+Transformer
目前失败
# LSTM+Self-Attention
根均方误差(RMSE): 6.786046166866807e-09  
平均绝对百分比误差(MAPE): 0.025098925456404686
![LSTM_Self-attention.png](3_training%2FLSTM_Self-attention.png)