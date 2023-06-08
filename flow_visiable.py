import dpkt
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import pandas as pd

# 创建字典来存储每个时间点的数据流量
packet_dict = defaultdict(int)

# 打开 pcap 文件
with open('D:\\流量抓包\\2023-05-30-11-31-17\\all.pcap', 'rb') as f:
    pcap = dpkt.pcap.Reader(f)

    for timestamp, buf in pcap:
        # 将时间戳转换为 datetime 对象
        dt = datetime.fromtimestamp(timestamp)
        # 累加包的大小
        packet_dict[dt] += len(buf)

# 将字典转换为 pandas DataFrame
df = pd.DataFrame(list(packet_dict.items()), columns=['Timestamp', 'Traffic'])

# 将时间戳设为索引
df = df.set_index('Timestamp')

# 可以选择进行重采样，比如按分钟，小时，等等
df = df.resample('1min').sum()
df['Traffic']=df['Traffic'] / (1024 * 1024)
fig, ax = plt.subplots(figsize=(15,5))
# 画图
plt.plot(df.index, df['Traffic'])
plt.title('Traffic over Time')
plt.xlabel('Timestamp')
plt.ylabel('Traffic (MB)')
plt.show()