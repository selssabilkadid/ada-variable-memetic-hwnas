# experiments/test_loaders.py

import random
import csv
from src.benchmark.hwnas_loader import HWNASLoader
from src.benchmark.nasbench201csv_loader import NASBench201CSVLoader  # CSV loader

def get_arch_ops_from_hwnas(loader_hwnas, arch_index: int):
    """
    Given a HWNASLoader instance and an arch_index, return the list of ops
    for that architecture by parsing the 'arch_str' from the network config.
    """
    config = loader_hwnas.get_net_config(arch_index)
    arch_str = config["arch_str"]  # Example: '|conv3x3~0|conv1x1~1+|conv3x3~0|none~1|...'

    ops = []
    nodes = arch_str.split('+')
    for node in nodes:
        parts = node.strip('|').split('|')
        for p in parts:
            op = p.split('~')[0]  # remove the edge index
            ops.append(op)
    return ops

def test_full_arch_info():
    print("=== Testing Full Architecture Info (CSV + HW-NAS-Bench) ===")

    # --- Load CSV accuracies ---
    csv_file = "data/nasbench201_accuracies.csv"
    csv_data = {}
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["arch_index"])
            csv_data[idx] = {
                "cifar10": float(row["acc_cifar10"]),
                "cifar100": float(row["acc_cifar100"]),
                "ImageNet16-120": float(row["acc_ImageNet16-120"])
            }

    # --- Instantiate HW-NAS-Bench loader ---
    loader_hwnas = HWNASLoader()

    # --- Pick a random architecture ---
    arch_index = random.choice(list(csv_data.keys()))
    print(f"\nRandom arch index: {arch_index}")

    # --- Accuracy from CSV ---
    print("=== Accuracy ===")
    for dataset in ["cifar10", "cifar100", "ImageNet16-120"]:
        acc = csv_data[arch_index][dataset]
        print(f"{dataset}: {acc:.2f}%")

    # --- Architecture ops from HW-NAS-Bench ---
    print("\n=== Architecture Ops ===")
    ops = get_arch_ops_from_hwnas(loader_hwnas, arch_index)
    print(ops)

    # --- Network config ---
    print("\n=== Network Config ===")
    net_config = loader_hwnas.get_net_config(arch_index)
    for k, v in net_config.items():
        print(f"{k}: {v}")

    # --- Hardware metrics ---
    print("\n=== Hardware Metrics ===")
    hw_metrics = loader_hwnas.get_hw_metrics(arch_index)
    for k, v in hw_metrics.items():
        print(f"{k}: {v}")

    # --- Example single latency / energy ---
    print("\n=== Selected Hardware Metrics ===")
    print(f"raspi4 latency: {loader_hwnas.get_latency(arch_index, 'raspi4')} ms")
    print(f"fpga energy: {loader_hwnas.get_energy(arch_index, 'fpga')} mJ")


if __name__ == "__main__":
    test_full_arch_info()