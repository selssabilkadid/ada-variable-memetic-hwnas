# # pareto_loader.py

# import os
# import pandas as pd
# import numpy as np
# from src.benchmark.hwnas_loader import HWNASLoader
# from attribution import train_model, compute_shap
# from pareto import get_pareto_mask

# class ParetoFrontLoader:
#     def __init__(self, base_dir="../../ground_truth"):
#         """
#         base_dir: top-level folder containing dataset/device subfolders
#                   e.g., results/cifar10/edgegpu
#         """
#         self.base_dir = base_dir
#         self.loader_hwnas = HWNASLoader()
#         self.datasets = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
#         self.devices = ["edgegpu", "raspi4", "edgetpu", "pixel3", "eyeriss", "fpga"]
#         self.data = {}  # dataset -> device -> dataframe

#         self._load_all_csvs()

#     def _load_all_csvs(self):
#         """Load pareto_front.csv and search_results.csv for all datasets/devices."""
#         for dataset in self.datasets:
#             self.data[dataset] = {}
#             for device in self.devices:
#                 folder = os.path.join(self.base_dir, dataset, device)
#                 pareto_csv = os.path.join(folder, "pareto_front.csv")
#                 search_csv = os.path.join(folder, "search_results.csv")
#                 if os.path.exists(pareto_csv) and os.path.exists(search_csv):
#                     pareto_df = pd.read_csv(pareto_csv)
#                     search_df = pd.read_csv(search_csv)
#                     self.data[dataset][device] = {
#                         "pareto": pareto_df,
#                         "search": search_df
#                     }
#                 else:
#                     print(f"[Warning] Missing CSVs for {dataset}/{device}")

#     def get_arch_ops(self, arch_index):
#         """Return flat list of ops for an architecture using HW-NAS-Bench."""
#         config = self.loader_hwnas.get_net_config(arch_index)
#         arch_str = config["arch_str"]
#         ops = []
#         for node in arch_str.split('+'):
#             parts = node.strip('|').split('|')
#             for p in parts:
#                 op = p.split('~')[0]
#                 ops.append(op)
#         return ops

#     def compute_pareto_attribution(self, dataset, device, target_metric="latency"):
#         """
#         Train LightGBM on search results for a specific dataset/device.
#         target_metric: the column name in the CSV to predict (e.g., 'latency', 'energy')
#         Returns:
#             pareto_indices, shap_values, importance
#         """
#         df = self.data[dataset][device]["search"]
#         arch_indices = df["arch_index"].values
#         X = df.drop(columns=["arch_index", target_metric]).values
#         Y = df[[target_metric]].values

#         # Identify Pareto front
#         mask = get_pareto_mask(Y)
#         pareto_indices = arch_indices[mask]
#         print(f"{dataset}/{device} Pareto front indices: {pareto_indices}")

#         # Train LightGBM and compute SHAP
#         model = train_model(X, Y[:,0])
#         shap_values, importance = compute_shap(model, X)
#         top_features = np.argsort(-importance)
#         print(f"Top feature indices for {dataset}/{device}: {top_features}")

#         # Map top features to architecture ops (example: first Pareto arch)
#         example_arch = pareto_indices[0]
#         ops = self.get_arch_ops(example_arch)
#         mapping = {idx: ops[idx] if idx < len(ops) else "unknown" for idx in top_features}
#         print(f"Top features mapped to ops for arch {example_arch}:")
#         for idx in top_features[:10]:
#             print(f"Feature {idx} -> {mapping[idx]}")

#         return pareto_indices, shap_values, importance

# pareto_loader.py

import os
import pandas as pd
import numpy as np
from src.benchmark.hwnas_loader import HWNASLoader
from attribution import train_model, compute_shap
from pareto import get_pareto_mask

class ParetoFrontLoader:
    def __init__(self, base_dir="../../ground_truth"):
        """
        base_dir: top-level folder containing dataset/device subfolders
                  e.g., ground_truth/cifar10/edgegpu
        """
        self.base_dir = base_dir
        self.loader_hwnas = HWNASLoader()
        self.datasets = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        self.devices = ["edgegpu", "raspi4", "edgetpu", "pixel3", "eyeriss", "fpga"]
        self.data = {}  # dataset -> device -> {'pareto': df, 'search': df}

        self._load_all_csvs()

    def _load_all_csvs(self):
        """Load pareto_front.csv and search_results.csv for all datasets/devices."""
        for dataset in self.datasets:
            self.data[dataset] = {}
            for device in self.devices:
                folder = os.path.join(self.base_dir, dataset, device)
                pareto_csv = os.path.join(folder, "pareto_front.csv")
                search_csv = os.path.join(folder, "search_results.csv")
                if os.path.exists(pareto_csv) and os.path.exists(search_csv):
                    pareto_df = pd.read_csv(pareto_csv)
                    search_df = pd.read_csv(search_csv)
                    self.data[dataset][device] = {
                        "pareto": pareto_df,
                        "search": search_df
                    }
                else:
                    print(f"[Warning] Missing CSVs for {dataset}/{device}")

    def get_arch_ops(self, arch_index):
        """Return flat list of ops for an architecture using HW-NAS-Bench."""
        config = self.loader_hwnas.get_net_config(arch_index)
        arch_str = config["arch_str"]
        ops = []
        for node in arch_str.split('+'):
            parts = node.strip('|').split('|')
            for p in parts:
                op = p.split('~')[0]
                ops.append(op)
        return ops

    def compute_pareto_attribution(self, dataset, device, target_metric="latency"):
        """
        Train LightGBM on search results for a specific dataset/device.
        target_metric: the column name in the CSV to predict (e.g., 'latency', 'energy')
        Returns:
            pareto_indices, shap_values, importance
        """
        if dataset not in self.data or device not in self.data[dataset]:
            print(f"[Error] No data for {dataset}/{device}")
            return None, None, None

        df = self.data[dataset][device]["search"]
        arch_indices = df["arch_index"].values

        # Features: all columns except arch_index and target
        X = df.drop(columns=["arch_index", target_metric]).values
        Y = df[[target_metric]].values

        # Compute Pareto front mask
        mask = get_pareto_mask(Y)
        pareto_indices = arch_indices[mask]
        print(f"{dataset}/{device} Pareto front indices: {pareto_indices}")

        # Train LightGBM and compute SHAP
        model = train_model(X, Y[:, 0])
        shap_values, importance = compute_shap(model, X)
        top_features = np.argsort(-importance)
        print(f"Top feature indices for {dataset}/{device}: {top_features[:10]}")

        # Map top features to architecture ops (example: first Pareto arch)
        if len(pareto_indices) > 0:
            example_arch = pareto_indices[0]
            ops = self.get_arch_ops(example_arch)
            mapping = {idx: ops[idx] if idx < len(ops) else "unknown" for idx in top_features}
            print(f"Top features mapped to ops for arch {example_arch}:")
            for idx in top_features[:10]:
                print(f"Feature {idx} -> {mapping[idx]}")

        return pareto_indices, shap_values, importance

    def get_search_results(self, dataset, device):
        """Return the full search results DataFrame for a given dataset/device."""
        if dataset in self.data and device in self.data[dataset]:
            return self.data[dataset][device]["search"]
        else:
            print(f"[Error] No search results for {dataset}/{device}")
            return None