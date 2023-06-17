# TF_NP
Network performance prediction, Transfer Learning
# Data processing

# LSTM
50 epoch  
根均方误差(RMSE): 1.4027239163850947e-08<br>
平均绝对百分比误差(MAPE): 0.041976578533649445
![uk_lstm.png](3_training%2Fuk_lstm.png)
# Transformer
只使用 Transformer 的 Encoder 部分  
50 epoch  
根均方误差(RMSE): 1.638216185056248e-08  
平均绝对百分比误差(MAPE): 0.4802953600883484
![uk_transformer.png](3_training%2Fuk_transformer.png)
# LSTM+Transformer
根均方误差(RMSE): 8.098309055346415e-09<br>
平均绝对百分比误差(MAPE): 0.038710303604602814
![lstm_transformer.png](3_training%2Flstm_transformer.png)
# LSTM+Self-Attention
根均方误差(RMSE): 9.609785795465063e-09  
平均绝对百分比误差(MAPE): 0.03314043954014778
![LSTM_Self-attention.png](3_training%2FLSTM_Self-attention.png)