import hashlib
import dpkt
import numpy as np

def create_hash(pkt):
    header_5_tuple = (pkt.src, pkt.dst, pkt.sport, pkt.dport, pkt.protocol)
    return hashlib.sha256(str(header_5_tuple).encode()).hexdigest()

def process_pcap(file_name, N_hash):
    X = []
    with open(file_name, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        for timestamp, buf in pcap:
            eth = dpkt.ethernet.Ethernet(buf)
            if isinstance(eth.data, dpkt.ip.IP):
                ip = eth.data
                pkt_hash = create_hash(ip)
                if pkt_hash in N_hash and (isinstance(ip.data, dpkt.tcp.TCP) or isinstance(ip.data, dpkt.udp.UDP)):
                    fst, fet, label = N_hash[pkt_hash]
                    if fst <= timestamp <= fet:
                        payload = ip.data.data
                        if len(payload) <= 1500:
                            payload += b'\x00' * (1500 - len(payload))
                        else:
                            payload = payload[:1500]
                        X.append((label, np.array([byte/255 for byte in payload])))
    return X

def main(N, p):
    N_hash = {}
    for f in N:
        fhash = create_hash(f)
        N_hash[fhash] = (f['start_time'], f['end_time'], f['label'])
    X = process_pcap(p, N_hash)
    return X

# 使用示例
N = [...]  # Netflow 数据
p = 'test.pcap'  # PCAP 文件路径
result = main(N, p)
