
# run_pareto_attribution.py

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler

from src.benchmark.hwnas_loader import HWNASLoader
from src.pareto_attribution.attribution import train_model, compute_shap
from src.pareto_attribution.pareto import compute_distance_to_front

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #
DATASET = "cifar10"
DEVICE  = "edgegpu"
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "../ground_truth")
OUTPUT_DIR       = os.path.join(BASE_DIR, "../outputs", DATASET, DEVICE)

OP_CHOICES = [
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
]

# Number of background samples for SHAP stability
SHAP_BACKGROUND_SIZE = 500


# ------------------------------------------------------------------ #
# Architecture helpers                                                 #
# ------------------------------------------------------------------ #

def get_arch_ops(loader: HWNASLoader, arch_index: int) -> list[str]:
    """Extract ordered list of op names from a NAS-Bench-201 arch string."""
    config   = loader.get_net_config(arch_index)
    arch_str = config["arch_str"]
    ops = []
    for node in arch_str.split("+"):
        for part in node.strip("|").split("|"):
            ops.append(part.split("~")[0])
    return ops


def encode_ops_onehot(ops: list[str], op_choices: list[str]) -> np.ndarray:
    """
    One-hot encode a list of ops into a flat feature vector.
    Shape: (len(ops) * len(op_choices),)
    Avoids treating op labels as ordinal integers.
    """
    vec = []
    for op in ops:
        vec.extend(1 if op == choice else 0 for choice in op_choices)
    return np.array(vec, dtype=np.float32)


def build_feature_names(n_ops: int, op_choices: list[str]) -> list[str]:
    """Return feature names aligned with the one-hot encoding."""
    return [f"op{i}_{op}" for i in range(n_ops) for op in op_choices]


# def add_derived_features(
#     X_onehot: np.ndarray,
#     op_choices: list[str],
# ) -> tuple[np.ndarray, list[str]]:
#     """
#     Append two summary features per architecture:
#       - none_count : number of pruned edges  (proxy for sparsity / low latency)
#       - conv_count : number of conv edges    (proxy for representational capacity)

#     These help the model capture coarse architectural patterns that
#     individual one-hot columns express only in scattered form,
#     preventing the 'none' op from dominating purely through repetition.
#     """
#     n_ops      = X_onehot.shape[1] // len(op_choices)
#     none_idx   = op_choices.index("none")
#     conv_idxs  = [i for i, op in enumerate(op_choices) if "conv" in op]
#     conv1_idxs = [i for i, op in enumerate(op_choices) if "1x1" in op]
#     conv3_idxs = [i for i, op in enumerate(op_choices) if "3x3" in op]
#     skip_idx = [i for i, op in enumerate(op_choices) if "skip" in op]
#     pool_idx = [i for i, op in enumerate(op_choices) if "pool" in op]
    

#     # Slice every len(op_choices)-th column starting at the op's position
#     none_cols  = X_onehot[:, none_idx::len(op_choices)]          # (N, n_ops)
#     conv_cols  = np.stack(
#         [X_onehot[:, ci::len(op_choices)] for ci in conv_idxs], axis=-1
#     ).sum(axis=-1)                                                 # (N, n_ops)
    

#     none_count = none_cols.sum(axis=1, keepdims=True)
#     conv_count = conv_cols.sum(axis=1, keepdims=True)

#     X_aug           = np.hstack([X_onehot, none_count, conv_count])
#     extra_names     = ["derived_none_count", "derived_conv_count"]
#     return X_aug, extra_names

def add_derived_features(
    X_onehot: np.ndarray,
    op_choices: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Append one count feature per operation type in op_choices.
    Each derived feature represents how many edges in the architecture
    use that operation — a coarse summary that helps the model capture
    global architectural patterns beyond scattered one-hot columns.

    Example derived features for the default OP_CHOICES:
      - derived_none_count        (sparsity / low-latency proxy)
      - derived_skip_connect_count
      - derived_nor_conv_1x1_count
      - derived_nor_conv_3x3_count
      - derived_avg_pool_3x3_count (capacity proxy)
    """
    n_cols = len(op_choices)

    # For each op at position op_idx, its one-hot columns in the flat vector
    # are at indices: op_idx, op_idx + n_cols, op_idx + 2*n_cols, ...
    # Summing across those columns gives the count of that op in the arch.
    op_counts = np.stack(
        [X_onehot[:, op_idx::n_cols].sum(axis=1) for op_idx in range(n_cols)],
        axis=1,
    )  # shape: (N, len(op_choices))

    extra_names = [f"derived_{op}_count" for op in op_choices]
    X_aug = np.hstack([X_onehot, op_counts])
    return X_aug, extra_names


# ------------------------------------------------------------------ #
# Scaling                                                              #
# ------------------------------------------------------------------ #

def robust_normalize(arr: np.ndarray) -> np.ndarray:
    """
    Robust scaling using median and IQR.
    Clips to [-3, 3] so that outlier architectures don't distort
    the Pareto distance computation.
    """
    scaled = RobustScaler().fit_transform(arr.reshape(-1, 1)).ravel()
    return np.clip(scaled, -3, 3)


# ------------------------------------------------------------------ #
# Persistence                                                          #
# ------------------------------------------------------------------ #

def save_results(results: dict, output_dir: str) -> None:
    """Serialize attribution summary (R², top-10 features) to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    summary = {}
    for label, res in results.items():
        key = (
            label.replace(" ", "_")
                 .replace("(", "")
                 .replace(")", "")
        )
        top_idx = np.argsort(-res["importance"])[:10]
        summary[key] = {
            "r2_test": round(res["r2_test"], 4),
            "top10_features": [
                {
                    "rank":       rank + 1,
                    "feature":    res["feature_names"][i],
                    "importance": round(float(res["importance"][i]), 6),
                }
                for rank, i in enumerate(top_idx)
            ],
        }
    out_path = os.path.join(output_dir, "attribution_summary_position.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[✓] Results saved → {out_path}")


# ------------------------------------------------------------------ #
# Core pipeline                                                        #
# ------------------------------------------------------------------ #

def process_search_results(
    dataset: str,
    device:  str,
    loader:  HWNASLoader,
) -> dict | None:

    folder     = os.path.join(GROUND_TRUTH_DIR, dataset, device)
    search_csv = os.path.join(folder, "search_results.csv")
    pareto_csv = os.path.join(folder, "pareto_front.csv")

    if not os.path.exists(search_csv) or not os.path.exists(pareto_csv):
        print(f"[WARN] Missing CSVs for {dataset}/{device}")
        return None

    # ---------------------------------------------------------------- #
    # 1. Load data                                                       #
    # ---------------------------------------------------------------- #
    search_df     = pd.read_csv(search_csv)
    pareto_df     = pd.read_csv(pareto_csv)
    arch_indices  = search_df["arch_index"].values
    pareto_indices = pareto_df["arch_index"].values
    pareto_mask   = np.isin(arch_indices, pareto_indices)
    print(f"{dataset}/{device} — total: {len(arch_indices)}, Pareto: {pareto_mask.sum()}")

    # ---------------------------------------------------------------- #
    # 2. Build feature matrix                                            #
    #    One-hot encoded ops  +  derived summary features               #
    # ---------------------------------------------------------------- #
    all_ops  = [get_arch_ops(loader, idx) for idx in arch_indices]
    n_ops    = len(all_ops[0])
    X_onehot = np.stack([encode_ops_onehot(ops, OP_CHOICES) for ops in all_ops])
    X=X_onehot

    # X, extra_names = add_derived_features(X_onehot, OP_CHOICES)
    # feature_names  = build_feature_names(n_ops, OP_CHOICES) + extra_names  # all 35 names
    feature_names  = build_feature_names(n_ops, OP_CHOICES)   # all 35 names
    # count_cols     = [name for name in feature_names if name.startswith("derived_")]  # 5 names

    print(f"Feature matrix: {X.shape}  ({n_ops} ops × {len(OP_CHOICES)} choices +  derived)")
    print(X)

    # ---------------------------------------------------------------- #
    # 3. Objectives                                                      #
    # ---------------------------------------------------------------- #
    error_rate = 1.0 - search_df["accuracy"].values / 100.0
    latency    = search_df["latency"].values

    # Robust normalization for Pareto distance (outlier-resistant)
    error_norm   = robust_normalize(error_rate)
    latency_norm = robust_normalize(latency)
    Y_norm = np.column_stack([error_norm, latency_norm])

    # Log-transform for regression targets (both are right-skewed)
    latency_log = np.log1p(latency)

    # ---------------------------------------------------------------- #
    # 4. Distance to Pareto front (normalized objective space)           #
    # ---------------------------------------------------------------- #
    pareto_Y_norm = Y_norm[pareto_mask]
    distance      = compute_distance_to_front(Y_norm, pareto_Y_norm)
    distance_log  = np.log1p(distance)

    # ---------------------------------------------------------------- #
    # 5. Stratified train/test split                                     #
    #    Stratify by Pareto membership so both splits always contain     #
    #    Pareto-front architectures (only 23 out of 15 625).            #
    # ---------------------------------------------------------------- #
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    idx_train, idx_test = next(sss.split(X, pareto_mask.astype(int)))

    # X_train = pd.DataFrame(X[idx_train], columns=feature_names)[count_cols]
    # X_test  = pd.DataFrame(X[idx_test],  columns=feature_names)[count_cols]
    # X_df    = pd.DataFrame(X,            columns=feature_names)[count_cols]
    
    X_train = pd.DataFrame(X[idx_train], columns=feature_names)
    X_test  = pd.DataFrame(X[idx_test],  columns=feature_names)
    X_df    = pd.DataFrame(X,            columns=feature_names)

    print(
        f"Split → train: {len(idx_train)} "
        f"(Pareto: {pareto_mask[idx_train].sum()}) | "
        f"test: {len(idx_test)} "
        f"(Pareto: {pareto_mask[idx_test].sum()})"
    )

    # ---------------------------------------------------------------- #
    # 6. Fixed background sample for SHAP stability                     #
    # ---------------------------------------------------------------- #
    rng    = np.random.default_rng(42)
    bg_idx = rng.choice(len(X), size=min(SHAP_BACKGROUND_SIZE, len(X)), replace=False)
    X_bg   = pd.DataFrame(X[bg_idx], columns=feature_names)
    # X_bg = pd.DataFrame(X[bg_idx], columns=feature_names)[count_cols]

    # ---------------------------------------------------------------- #
    # 7. Train one model per objective, evaluate, compute SHAP          #
    # ---------------------------------------------------------------- #
    targets = {
        "tradeoff (distance)":   distance_log,
        "accuracy (error rate)": error_rate,
        "latency (log)":         latency_log,
    }

    results = {}

    for label, y in targets.items():
        print(f"\n{'=' * 55}")
        print(f"  Training: {label}")
        print(f"{'=' * 55}")

        # Pass DataFrame so LightGBM stores feature names → no sklearn warning
        model = train_model(X_train, y[idx_train])

        # Held-out R²
        y_pred = model.predict(X_test)
        ss_res = np.sum((y[idx_test] - y_pred) ** 2)
        ss_tot = np.sum((y[idx_test] - y[idx_test].mean()) ** 2)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"  Test R²: {r2:.4f}")

        # SHAP on fixed background sample for stability
        shap_vals, importance = compute_shap(model, X_bg)
        top_idx = np.argsort(-importance)[:10]

        print(f"  Top 10 SHAP features:")
        for rank, fi in enumerate(top_idx, 1):
            print(f"    {rank:2d}. {feature_names[fi]:<30s}  importance={importance[fi]:.6f}")
        
        # print(f"  Top 10 SHAP features:")
        # for rank, fi in enumerate(top_idx, 1):
        #     print(f"    {rank:2d}. {count_cols[fi]:<30s}  importance={importance[fi]:.6f}")

        results[label] = {
            "model":        model,
            "shap_values":  shap_vals,
            "importance":   importance,
            "feature_names": feature_names,
            # "feature_names": count_cols,
            "r2_test":      r2,
        }

    return results


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def main() -> None:
    # Single loader instance — avoids reloading the pickle per model
    loader  = HWNASLoader()
    results = process_search_results(DATASET, DEVICE, loader)
    if results:
        save_results(results, OUTPUT_DIR)
        print("\n[✓] All models trained successfully.")


if __name__ == "__main__":
    main()