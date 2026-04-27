"""
variance_weights.py
-------------------
Offline precomputation of per-edge latency leverage weights using
between-group variance (ANOVA decomposition) over the HW-NAS-Bench
NAS-Bench-201 search space (15,625 architectures).

Place this file in:  src/utils/variance_weights.py

Usage (run once from project root)
-----------------------------------
    python src/utils/variance_weights.py \
        --dataset   cifar10 \
        --output_dir ./data/variance_weights

    # To run a specific device only:
    python src/utils/variance_weights.py \
        --dataset cifar10 \
        --devices edgegpu raspi4

    # To use a random sample instead of all 15625:
    python src/utils/variance_weights.py \
        --dataset cifar10 \
        --sample_size 500

Outputs
-------
    data/variance_weights/
        weights_<device>_<dataset>.npy      # shape (6,) normalised weights
        weights_all_<dataset>.json          # all devices in one readable file
"""

import argparse
import json
import os
import sys

import numpy as np

# ── Make sure project root is on the path ─────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.benchmark.hwnas_loader import HWNASLoader

# ── NAS-Bench-201 constants ────────────────────────────────────────────────
N_ARCHS = 15_625
N_EDGES = 6
N_OPS   = 5

OP_NAMES = {
    0: "none",
    1: "skip_connect",
    2: "nor_conv_1x1",
    3: "nor_conv_3x3",
    4: "avg_pool_3x3",
}

DEVICES = ["fpga", "edgegpu", "edgetpu", "raspi4", "pixel3", "eyeriss"]


# ── Architecture index decoder ─────────────────────────────────────────────
def index_to_ops(arch_index: int) -> np.ndarray:
    """
    Decode a flat architecture index (0-15624) into a length-6 array
    of operation indices (0-4).

    NAS-Bench-201 encodes architectures as mixed-radix numbers in base 5:
        arch_index = op[0] + 5*op[1] + 25*op[2] + ... + 5^5*op[5]
    """
    ops = np.zeros(N_EDGES, dtype=np.int32)
    remaining = arch_index
    for edge in range(N_EDGES):
        ops[edge] = remaining % N_OPS
        remaining  = remaining // N_OPS
    return ops


# ── Core computation ───────────────────────────────────────────────────────
def compute_variance_weights(
    latencies: np.ndarray,
    ops_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute weighted between-group variance (ANOVA decomposition) for each
    of the 6 edges.

    Parameters
    ----------
    latencies  : shape (N,)   latency value for each architecture
    ops_matrix : shape (N, 6) operation index per edge per architecture

    Returns
    -------
    weights : shape (6,) normalised to sum to 1.
              High weight = swapping this edge's operation strongly
              shifts latency -> high-leverage edge for local search.
    """
    N = len(latencies)
    global_mean = latencies.mean()
    weights = np.zeros(N_EDGES)

    for edge in range(N_EDGES):
        between_var = 0.0
        for op in range(N_OPS):
            mask = ops_matrix[:, edge] == op
            n_k  = mask.sum()
            if n_k == 0:
                continue
            mu_k = latencies[mask].mean()
            between_var += n_k * (mu_k - global_mean) ** 2
        weights[edge] = between_var / N     # normalised between-group variance

    total = weights.sum()
    if total < 1e-12:
        return np.ones(N_EDGES) / N_EDGES  # degenerate: uniform fallback

    return weights / total


# ── Data collection ────────────────────────────────────────────────────────
def collect_latencies(
    hw_loader: HWNASLoader,
    device: str,
    dataset: str,
    indices: np.ndarray,
) -> tuple:
    """
    Pull latencies from HWNASLoader for the given indices.
    Returns (latencies, ops_matrix).
    """
    N          = len(indices)
    latencies  = np.zeros(N)
    ops_matrix = np.zeros((N, N_EDGES), dtype=np.int32)

    for i, arch_idx in enumerate(indices):
        latencies[i]  = hw_loader.get_latency(int(arch_idx), device, dataset)
        ops_matrix[i] = index_to_ops(int(arch_idx))

        if (i + 1) % 3000 == 0:
            print(f"    [{i+1}/{N}] collected ...")

    return latencies, ops_matrix


# ── Loader for precomputed weights (used by memetic NSGA-II later) ─────────
def load_weights(device: str, dataset: str, weights_dir: str) -> np.ndarray:
    """
    Load precomputed variance weights for a given device and dataset.
    Returns shape (6,) array, or uniform weights if file not found.
    """
    path = os.path.join(weights_dir, f"weights_{device}_{dataset}.npy")
    if os.path.isfile(path):
        return np.load(path)
    print(f"[WARNING] Weight file not found: {path}. Using uniform weights.")
    return np.ones(N_EDGES) / N_EDGES


# ── Main ───────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] Initialising HWNASLoader ...")
    hw_loader = HWNASLoader()

    all_indices = np.arange(N_ARCHS)

    if args.sample_size > 0:
        np.random.seed(args.seed)
        indices = np.random.choice(all_indices, size=args.sample_size, replace=False)
        print(f"[INFO] Using random sample of {args.sample_size} architectures.")
    else:
        indices = all_indices
        print(f"[INFO] Using all {N_ARCHS} architectures (exact computation).")

    devices     = args.devices if args.devices else DEVICES
    all_weights = {}

    for device in devices:
        print(f"\n-- Device: {device} --")
        print(f"  Collecting latencies ...")

        latencies, ops_matrix = collect_latencies(
            hw_loader, device, args.dataset, indices
        )

        print(f"  Computing variance weights ...")
        weights = compute_variance_weights(latencies, ops_matrix)

        # Save .npy
        npy_path = os.path.join(
            args.output_dir, f"weights_{device}_{args.dataset}.npy"
        )
        np.save(npy_path, weights)
        print(f"  Saved -> {npy_path}")

        # Print summary
        print(f"  Edge weights:")
        for edge, w in enumerate(weights):
            bar = "#" * int(w * 40)
            print(f"    edge {edge}: {w:.4f}  {bar}")

        all_weights[device] = weights.tolist()

    # Save all devices in one JSON
    json_path = os.path.join(
        args.output_dir, f"weights_all_{args.dataset}.json"
    )
    with open(json_path, "w") as f:
        json.dump(
            {
                "dataset":      args.dataset,
                "n_archs_used": int(len(indices)),
                "op_names":     OP_NAMES,
                "weights":      all_weights,
            },
            f,
            indent=2,
        )
    print(f"\n[INFO] All weights saved -> {json_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-edge latency leverage weights (offline, run once)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "ImageNet16-120"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/variance_weights",
        help="Directory where .npy and .json weight files are saved.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Number of architectures to sample (0 = use all 15625).",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=None,
        help="Devices to process (default: all 6).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()
    main(args)