# import sys
# import os

# NAS_BENCH_PATH = os.path.join(os.path.dirname(__file__), "../../HW-NAS-Bench")
# sys.path.insert(0, NAS_BENCH_PATH)


# class NASBench201Loader:
#     """
#     Wrapper to get accuracy from NAS-Bench-201.
#     Based on the official API:
#       - query_by_index(idx, dataset) returns dict {seed: ResultsCount}
#       - ResultsCount.get_eval('x-test') returns {'accuracy': ..., 'loss': ...}
#       - get_more_info() is the simplest high-level method
#     """

#     # NAS-Bench-201 uses these dataset keys internally
#     DATASET_KEYS = {
#         "cifar10":          "cifar10-valid",
#         "cifar100":         "cifar100",
#         "ImageNet16-120":   "ImageNet16-120",
#     }

#     def __init__(self, pth_path: str = "data/NAS-Bench-201-v1_1-096897.pth"):
#         from nas_201_api import NASBench201API as API
#         self.api = API(pth_path, verbose=False)
#         print("[NASBench201Loader] Loaded accuracy data.")

#     def get_accuracy(self, arch_index: int, dataset: str = "cifar10") -> float:
#         """
#         Returns test accuracy (%) for a given architecture.

#         Uses get_more_info() which is the recommended high-level method.
#         hp='200' means trained for 200 epochs (full training).
#         is_random=True picks one trial randomly (all seeds give similar results).
#         """
#         dataset_key = self.DATASET_KEYS.get(dataset, dataset)

#         info = self.api.get_more_info(
#             arch_index,
#             dataset_key,
#             iepoch=None,    # last epoch
#             hp="200",       # 200-epoch hyperparameters
#             is_random=True  # randomly pick one seed
#         )

#         # get_more_info returns keys like 'test-accuracy', 'train-accuracy'
#         # the exact key depends on the dataset
#         if "test-accuracy" in info:
#             return info["test-accuracy"]
#         elif "valtest-accuracy" in info:
#             return info["valtest-accuracy"]
#         else:
#             # fallback: print what's available and pick the best key
#             print(f"[DEBUG] Available keys in get_more_info: {list(info.keys())}")
#             acc_keys = [k for k in info.keys() if "accuracy" in k]
#             if acc_keys:
#                 return info[acc_keys[0]]
#             raise KeyError(f"No accuracy key found in: {list(info.keys())}")

#     def get_accuracy_via_query(self, arch_index: int, dataset: str = "cifar10") -> float:
#         """
#         Alternative method using query_by_index directly.
#         Returns average test accuracy across all seeds.
        
#         query_by_index returns: {seed: ResultsCount}
#         ResultsCount.get_eval('x-test') returns: {'accuracy': float, 'loss': float}
#         """
#         dataset_key = self.DATASET_KEYS.get(dataset, dataset)

#         # dict of {seed: ResultsCount}
#         results = self.api.query_by_index(arch_index, dataset_key)

#         accuracies = []
#         for seed, result in results.items():
#             try:
#                 eval_info = result.get_eval("x-test")
#                 accuracies.append(eval_info["accuracy"])
#             except Exception:
#                 try:
#                     eval_info = result.get_eval("x-valid")
#                     accuracies.append(eval_info["accuracy"])
#                 except Exception:
#                     pass

#         if not accuracies:
#             raise ValueError(
#                 f"Could not extract accuracy for arch {arch_index} on {dataset_key}. "
#                 f"Available seeds: {list(results.keys())}"
#             )

#         return sum(accuracies) / len(accuracies)

#     def inspect(self, arch_index: int, dataset: str = "cifar10"):
#         """
#         Debug method — prints everything available for one architecture.
#         Run this once to understand the data structure.
#         """
#         dataset_key = self.DATASET_KEYS.get(dataset, dataset)
#         print(f"\n[Inspect] arch_index={arch_index}, dataset={dataset_key}")

#         # Method 1: get_more_info
#         info = self.api.get_more_info(arch_index, dataset_key,
#                                        iepoch=None, hp="200", is_random=True)
#         print(f"  get_more_info keys: {list(info.keys())}")
#         print(f"  get_more_info values: {info}")

#         # Method 2: query_by_index
#         results = self.api.query_by_index(arch_index, dataset_key)
#         print(f"  query_by_index seeds: {list(results.keys())}")
#         if results:
#             first_seed = list(results.keys())[0]
#             r = results[first_seed]
#             print(f"  ResultsCount type: {type(r)}")
#             try:
#                 print(f"  get_eval('x-test'): {r.get_eval('x-test')}")
#             except Exception as e:
#                 print(f"  get_eval('x-test') failed: {e}")
#             try:
#                 print(f"  get_eval('x-valid'): {r.get_eval('x-valid')}")
#             except Exception as e:
#                 print(f"  get_eval('x-valid') failed: {e}")
#             try:
#                 print(f"  get_train(): {r.get_train()}")
#             except Exception as e:
#                 print(f"  get_train() failed: {e}")
                
                
#     def get_arch_ops(self, arch_index: int):
#         arch_str = self.api.arch(index=arch_index)
    
#     # parse string → list of ops (6 edges)
#         ops = []
#         nodes = arch_str.split('+')
#         for node in nodes:
#                parts = node.strip('|').split('|')
#                for p in parts:
#                    op = p.split('~')[0]
#                    ops.append(op)
    
#         return ops  # length = 6

import sys
import os
import torch

# Path to HW-NAS-Bench root (where .pth is located)
HW_NAS_BENCH_PATH = os.path.join(os.path.dirname(__file__), "../../HW-NAS-Bench")
sys.path.insert(0, HW_NAS_BENCH_PATH)


class NASBench201Loader:
    """
    Wrapper to get accuracy + architecture structure from NAS-Bench-201.
    """

    DATASET_KEYS = {
        "cifar10": "cifar10-valid",
        "cifar100": "cifar100",
        "ImageNet16-120": "ImageNet16-120",
    }

    def __init__(self, pth_path: str = None):
        from nas_201_api import NASBench201API as API

        if pth_path is None:
            pth_path = os.path.join(
                HW_NAS_BENCH_PATH,
                "NAS-Bench-201-v1_1-096897.pth"
            )

        if not os.path.exists(pth_path):
            raise FileNotFoundError(f"NAS-Bench-201 .pth not found at: {pth_path}")

        self.api = API(pth_path, verbose=False)
        print("[NASBench201Loader] Loaded accuracy data.")

    # ----------------------------
    # Accuracy (fast + stable)
    # ----------------------------
    def get_accuracy(self, arch_index: int, dataset: str = "cifar10") -> float:
        dataset_key = self.DATASET_KEYS.get(dataset, dataset)

        info = self.api.get_more_info(
            arch_index,
            dataset_key,
            iepoch=None,
            hp="200",
            is_random=True
        )

        if "test-accuracy" in info:
            return info["test-accuracy"]
        elif "valtest-accuracy" in info:
            return info["valtest-accuracy"]
        else:
            acc_keys = [k for k in info if "accuracy" in k]
            if acc_keys:
                return info[acc_keys[0]]
            raise KeyError(f"No accuracy key found in: {list(info.keys())}")

    # ----------------------------
    # Faster bulk-friendly version
    # ----------------------------
    def get_accuracy_via_query(self, arch_index: int, dataset: str = "cifar10") -> float:
        dataset_key = self.DATASET_KEYS.get(dataset, dataset)
        results = self.api.query_by_index(arch_index, dataset_key)

        accs = []
        for _, r in results.items():
            try:
                accs.append(r.get_eval("x-test")["accuracy"])
            except Exception:
                try:
                    accs.append(r.get_eval("x-valid")["accuracy"])
                except Exception:
                    continue

        if not accs:
            raise ValueError(f"Failed to extract accuracy for arch {arch_index}")

        return sum(accs) / len(accs)

    # ----------------------------
    # CRITICAL: architecture encoding
    # ----------------------------
    def get_arch_ops(self, arch_index: int):
        """
        Returns list of 6 operations (one per edge).
        """
        arch_str = self.api.arch(index=arch_index)

        ops = []
        nodes = arch_str.split('+')

        for node in nodes:
            parts = node.strip('|').split('|')
            for p in parts:
                if not p:
                    continue
                op = p.split('~')[0]
                ops.append(op)

        # sanity check (VERY important)
        if len(ops) != 6:
            raise ValueError(
                f"Expected 6 ops, got {len(ops)} for arch {arch_index}: {arch_str}"
            )

        return ops

    # ----------------------------
    # Debug helper
    # ----------------------------
    def inspect(self, arch_index: int, dataset: str = "cifar10"):
        dataset_key = self.DATASET_KEYS.get(dataset, dataset)

        print(f"\n[Inspect] arch_index={arch_index}, dataset={dataset_key}")

        info = self.api.get_more_info(
            arch_index, dataset_key, iepoch=None, hp="200", is_random=True
        )
        print(f"get_more_info keys: {list(info.keys())}")
        print(f"get_more_info values: {info}")

        results = self.api.query_by_index(arch_index, dataset_key)
        print(f"query_by_index seeds: {list(results.keys())}")

        if results:
            r = list(results.values())[0]
            try:
                print("x-test:", r.get_eval("x-test"))
            except Exception:
                pass
            try:
                print("x-valid:", r.get_eval("x-valid"))
            except Exception:
                pass