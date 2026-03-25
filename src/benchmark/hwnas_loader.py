import sys
import os
import numpy as np

HW_NAS_BENCH_PATH = os.path.join(os.path.dirname(__file__), "../../../HW-NAS-Bench")
sys.path.insert(0, HW_NAS_BENCH_PATH)

from hw_nas_bench_api import HWNASBenchAPI as HWAPI


class HWNASLoader:
    """
    Wrapper around HW-NAS-Bench API (NASBench-201 space).
    query_by_index returns ALL hardware metrics at once as a dict.
    Accuracy requires NAS-Bench-201 API separately.
    """

    DATASETS = ["cifar10", "cifar100", "ImageNet16-120"]

    # Exact keys as returned by the API
    HW_METRICS = {
        "edgegpu_latency",   # ms
        "edgegpu_energy",    # mJ
        "raspi4_latency",    # ms
        "edgetpu_latency",   # ms
        "pixel3_latency",    # ms
        "eyeriss_latency",   # ms
        "eyeriss_energy",    # mJ
        "fpga_latency",      # ms
        "fpga_energy",       # mJ
    }

    SEARCH_SPACE_SIZE = 15625  # NASBench-201

    def __init__(self, data_path: str = "HW-NAS-Bench-v1_0.pickle"):
        """
        data_path: path to the pickle file.
        The pickle file is included in the HW-NAS-Bench repo — no separate download.
        """
        self.api = HWAPI(data_path, search_space="nasbench201")
        print(f"[HWNASLoader] Ready. Search space size: {self.SEARCH_SPACE_SIZE}")

    def get_hw_metrics(self, arch_index: int, dataset: str = "cifar10") -> dict:
        """
        Returns all hardware metrics for one architecture as a dict.
        
        Example output:
        {
            'edgegpu_latency': 5.807,
            'edgegpu_energy': 24.226,
            'raspi4_latency': 10.481,
            'edgetpu_latency': 0.957,
            'pixel3_latency': 3.605,
            'eyeriss_latency': 3.645,
            'eyeriss_energy': 0.687,
            'fpga_latency': 2.572,
            'fpga_energy': 18.010
        }
        """
        return self.api.query_by_index(arch_index, dataset)

    def get_latency(self, arch_index: int,
                    hw_platform: str = "raspi4",
                    dataset: str = "cifar10") -> float:
        """
        Convenience method to get a single latency value.
        hw_platform: one of edgegpu, raspi4, edgetpu, pixel3, eyeriss, fpga
        """
        key = f"{hw_platform}_latency"
        if key not in self.HW_METRICS:
            raise ValueError(f"Unknown platform key '{key}'. "
                             f"Available: {[k for k in self.HW_METRICS if 'latency' in k]}")
        metrics = self.get_hw_metrics(arch_index, dataset)
        return metrics[key]

    def get_net_config(self, arch_index: int, dataset: str = "cifar10") -> dict:
        """Returns the network configuration dict for architecture building."""
        return self.api.get_net_config(arch_index, dataset)

    def build_network(self, arch_index: int, dataset: str = "cifar10"):
        """Builds and returns the actual PyTorch network for a given architecture."""
        from hw_nas_bench_api.nas_201_models import get_cell_based_tiny_net
        config = self.get_net_config(arch_index, dataset)
        return get_cell_based_tiny_net(config)

    def random_architecture(self) -> int:
        return np.random.randint(0, self.SEARCH_SPACE_SIZE)