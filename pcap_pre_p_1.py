import hashlib
import dpkt
import numpy as np

'''
在这个类中，构造函数接收 Netflow 数据和 PCAP 文件路径作为参数，
并在内部调用 _generate_hash_dict 方法来生成字典 N_hash。
然后，process_pcap 方法处理 PCAP 文件，其他部分的代码逻辑保持不变
'''
import hashlib
import dpkt
import numpy as np

class PcapProcessor:
    def __init__(self, N, p):
        """
        初始化PcapProcessor对象。

        :param N: 包含Netflow数据的列表。
        :param p: PCAP文件的路径。
        """
        self.N = N  # Netflow 数据
        self.p = p  # PCAP 文件路径
        self.N_hash = self._generate_hash_dict()  # 五元组哈希和流信息（开始时间，结束时间，标签）的映射

    def create_hash(self, pkt):
        """
        为数据包创建五元组哈希。

        :param pkt: 数据包。
        :return: 数据包的五元组哈希。
        """
        header_5_tuple = (pkt.src, pkt.dst, pkt.sport, pkt.dport, pkt.protocol)
        return hashlib.sha256(str(header_5_tuple).encode()).hexdigest()

    def _generate_hash_dict(self):
        """
        为Netflow数据生成五元组哈希字典。

        :return: 五元组哈希和流信息（开始时间，结束时间，标签）的映射。
        """
        N_hash = {}
        for f in self.N:
            fhash = self.create_hash(f)
            N_hash[fhash] = (f['start_time'], f['end_time'], f['label'])
        return N_hash

    def process_pcap(self):
        """
        处理PCAP文件，返回标签和有效负载数据的列表。

        :return: 标签和有效负载数据的列表。
        """
        X = []
        with open(self.p, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            for timestamp, buf in pcap:
                eth = dpkt.ethernet.Ethernet(buf)
                if isinstance(eth.data, dpkt.ip.IP):
                    ip = eth.data
                    pkt_hash = self.create_hash(ip)
                    # 检查数据包的五元组哈希是否在Netflow数据中，并且数据包的协议是否为TCP或UDP
                    if pkt_hash in self.N_hash and (isinstance(ip.data, dpkt.tcp.TCP) or isinstance(ip.data, dpkt.udp.UDP)):
                        fst, fet, label = self.N_hash[pkt_hash]
                        # 检查数据包的时间戳是否在流的开始和结束时间之间
                        if fst <= timestamp <= fet:
                            payload = ip.data.data
                            # 调整有效负载的长度为1500字节
                            if len(payload) <= 1500:
                                payload += b'\x00' * (1500 - len(payload))
                            else:
                                payload = payload[:1500]
                            # 添加标签和有效负载数据到结果列表
                            X.append((label, np.array([byte/255 for byte in payload])))
        return X

# 使用示例
N = [...]  # Netflow 数据
p = 'test.pcap'  # PCAP 文件路径
processor = PcapProcessor(N, p)
result = processor.process_pcap()