
import configparser
from multiprocessing import Pool

import iperf3
from ping3 import ping

CONFIG_FILE = "./config.ini"

config = configparser.ConfigParser()
config.read(CONFIG_FILE)


MAX_IT_LAT = int(config["NETWORK"]["max_it_latency"])
LATENCY_IT = int(config["NETWORK"]["latency_iterations"])

MAX_IT_BW = int(config["NETWORK"]["max_it_bandwidth"])
BW_DURATION = int(config["NETWORK"]["bandwidth_duration"])


def get_latency(host):
    for _ in range(MAX_IT_LAT):
        lat = []
        try:
            for _ in range(LATENCY_IT):
                lat.append(ping(host))
            return sum(lat) / len(lat)
        except Exception as e:
            print("Network error - LATENCY: {}".format(e))

    return -1


def get_bandwidth(host):
    for _ in range(MAX_IT_BW):
        iperf_client = iperf3.Client()
        iperf_client.duration = BW_DURATION
        try:
            iperf_client.server_hostname = host
            data = iperf_client.run().json
            return {"upload": data["end"]["sum_sent"]["bits_per_second"], "download": data["end"]["sum_received"]["bits_per_second"]}
        except Exception as e:
            print("Network error - BW: {}".format(e))
    return {}

def probe_newtork(host):
    pool = Pool()

    lat = pool.map_async(get_latency, [host])
    bw = pool.map_async(get_bandwidth, [host])

    return {"latency": lat.get()[0], "bandwidth": bw.get()[0]}


if __name__ == "__main__":
    import sys
    print(probe_newtork(sys.argv[1]))