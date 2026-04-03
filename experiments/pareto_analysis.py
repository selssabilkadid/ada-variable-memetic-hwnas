# experiments/pareto_analysis.py
"""
Pareto Front Statistical Analysis Toolkit
=========================================

Enhanced version:
- Saves experiment results to JSON
- Generates and saves plots for marginal distributions, pairwise enrichment,
  HV contribution concentration, and cross-device correlations.
- Dataset and other devices are now global, easy to modify.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
from scipy.stats import chi2_contingency, entropy
from src.benchmark.hwnas_loader import HWNASLoader
from pymoo.indicators.hv import HV

# ------------------------ CONFIG ------------------------
# Global parameters
DATASETS = ["cifar10"]            # list of datasets to analyze
DEVICES = ["fpga", "edgegpu", "edgetpu", "raspi4", "pixel3", "eyeriss"]             # list of primary devices
OTHER_DEVICES = ["fpga", "edgegpu", "edgetpu", "raspi4", "pixel3", "eyeriss"]        # devices to compare for cross-device analysis

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "../ground_truth")
OUTPUT_DIR = os.path.join(BASE_DIR, "../outputs")

# ------------------ HELPER FUNCTIONS -------------------

def get_arch_ops(loader, arch_index: int):
    """Flatten NAS-Bench-201 architecture string into per-edge ops list."""
    config = loader.get_net_config(arch_index)
    arch_str = config["arch_str"]
    ops = []
    nodes = arch_str.split('+')
    for node in nodes:
        parts = node.strip('|').split('|')
        for p in parts:
            op = p.split('~')[0]
            ops.append(op)
    return ops

def compute_pareto_front(df):
    """Compute Pareto front indices minimizing error_rate & latency."""
    objs = df[["error_rate", "latency"]].values
    num_points = objs.shape[0]
    pareto_mask = np.ones(num_points, dtype=bool)
    for i in range(num_points):
        for j in range(num_points):
            if i != j and all(objs[j] <= objs[i]) and any(objs[j] < objs[i]):
                pareto_mask[i] = False
                break
    return df.index[pareto_mask].tolist()

def compute_operation_frequencies(arch_indices, loader):
    """Compute marginal operation frequencies per edge."""
    edge_count = len(get_arch_ops(loader, 0))
    freq = [Counter() for _ in range(edge_count)]
    for idx in arch_indices:
        ops = get_arch_ops(loader, idx)
        for e, op in enumerate(ops):
            freq[e][op] += 1
    return freq

def pairwise_enrichment(pareto_indices, all_indices, loader):
    """Compute pairwise enrichment ratios for operation pairs on edges."""
    edge_count = len(get_arch_ops(loader, 0))
    pair_counts_pareto = Counter()
    pair_counts_all = Counter()
    
    for indices, counter in [(pareto_indices, pair_counts_pareto), (all_indices, pair_counts_all)]:
        for idx in indices:
            ops = get_arch_ops(loader, idx)
            for (e1, op1), (e2, op2) in combinations(enumerate(ops), 2):
                counter[((e1, op1), (e2, op2))] += 1
    
    enrichment = {}
    total_pareto = len(pareto_indices)
    total_all = len(all_indices)
    
    for pair in pair_counts_all:
        freq_p = pair_counts_pareto.get(pair, 0) / total_pareto
        freq_all = pair_counts_all[pair] / total_all
        enrichment[pair] = freq_p / freq_all if freq_all > 0 else np.nan
    return enrichment

def hv_contribution(df, pareto_indices):
    """Compute normalized Hypervolume contributions per architecture."""
    objs = df.loc[pareto_indices, ["error_rate", "latency"]].values
    ref_point = np.max(df[["error_rate", "latency"]].values, axis=0) * 1.1
    hv_metric = HV(ref_point=ref_point)
    full_hv = hv_metric(objs)
    
    contributions = []
    for i in range(len(objs)):
        subset = np.delete(objs, i, axis=0)
        hv_subset = hv_metric(subset)
        contributions.append(full_hv - hv_subset)
    contributions = np.array(contributions)
    contributions /= contributions.sum()
    return contributions

def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

def per_edge_corr(freq1, freq2):
    """Compute correlation of per-edge op frequencies."""
    corrs = []
    for e in range(len(freq1)):
        all_ops = set(freq1[e].keys()) | set(freq2[e].keys())
        vec1 = np.array([freq1[e].get(op,0) for op in all_ops])
        vec2 = np.array([freq2[e].get(op,0) for op in all_ops])
        if vec1.std() == 0 or vec2.std() == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(vec1, vec2)[0,1]
        corrs.append(corr)
    return corrs

def save_json(results, dataset, device):
    """Save results dictionary as JSON."""
    out_dir = os.path.join(OUTPUT_DIR, dataset, device)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "pareto_analysis.json"), "w") as f:
        json.dump(results, f, indent=4)

def save_plot(fig, dataset, device, name):
    """Save matplotlib figure."""
    out_dir = os.path.join(OUTPUT_DIR, dataset, device)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"))
    plt.close(fig)

# -------------------- MAIN SCRIPT ----------------------

def main():
    loader = HWNASLoader()
    
    for dataset in DATASETS:
        for device in DEVICES:
            print(f"Processing {dataset}/{device}...")
            folder = os.path.join(GROUND_TRUTH_DIR, dataset, device)
            df = pd.read_csv(os.path.join(folder, "search_results.csv"))
            df["error_rate"] = 1 - df["accuracy"]/100.0
            
            all_indices = df.index.tolist()
            pareto_indices = compute_pareto_front(df)
            
            results = {"dataset": dataset, "device": device, "pareto_size": len(pareto_indices)}
            
            # ---------------- 1. Marginal operation frequencies ----------------
            freq_pareto = compute_operation_frequencies(pareto_indices, loader)
            freq_nonpareto = compute_operation_frequencies(list(set(all_indices)-set(pareto_indices)), loader)
            
            marginal_stats = []
            for e, (c1, c2) in enumerate(zip(freq_pareto, freq_nonpareto)):
                all_ops = set(c1.keys()) | set(c2.keys())
                obs = np.array([[c1.get(op,0) for op in all_ops],
                                [c2.get(op,0) for op in all_ops]])
                chi2, p, _, _ = chi2_contingency(obs)
                kl_div = entropy(obs[0]+1e-9, obs[1]+1e-9)
                marginal_stats.append({"edge": e, "chi2_p": p, "kl_div": kl_div})
            results["marginal_stats"] = marginal_stats
            
            # Plot marginal KL
            fig, ax = plt.subplots()
            ax.bar([s["edge"] for s in marginal_stats], [s["kl_div"] for s in marginal_stats])
            ax.set_xlabel("Edge")
            ax.set_ylabel("KL divergence")
            ax.set_title("Marginal KL divergence per edge")
            save_plot(fig, dataset, device, "marginal_kl")
            
            # ---------------- 2. Pairwise enrichment ----------------
            enrichment = pairwise_enrichment(pareto_indices, all_indices, loader)
            top_pairs = sorted(enrichment.items(), key=lambda x: -x[1])[:10]
            results["top_pairs"] = [{"pair": str(pair), "enrichment": float(ratio)} for pair, ratio in top_pairs]
            
            # Plot top pairwise enrichment
            fig, ax = plt.subplots()
            labels = [str(pair) for pair,_ in top_pairs]
            ratios = [ratio for _, ratio in top_pairs]
            ax.barh(labels[::-1], ratios[::-1])
            ax.set_xlabel("Enrichment ratio")
            ax.set_title("Top 10 enriched operation pairs")
            save_plot(fig, dataset, device, "pairwise_enrichment")
            
            # ---------------- 3. HV contribution ----------------
            try:
                contributions = hv_contribution(df, pareto_indices)
                top10sum = contributions[np.argsort(contributions)[-max(1,len(contributions)//10):]].sum()
                results["hv_stats"] = {
                    "min": float(contributions.min()),
                    "max": float(contributions.max()),
                    "top10_percent_sum": float(top10sum)
                }
                
                # Plot HV contributions
                fig, ax = plt.subplots()
                ax.bar(range(len(contributions)), contributions)
                ax.set_xlabel("Pareto architecture index")
                ax.set_ylabel("Normalized HV contribution")
                ax.set_title("HV contributions per architecture")
                save_plot(fig, dataset, device, "hv_contributions")
                
            except Exception as e:
                print("Error computing HV contributions:", e)
            
            # ---------------- 4. Cross-device comparison ----------------
            cross_results = []
            for other_device in OTHER_DEVICES:
                df2 = pd.read_csv(os.path.join(GROUND_TRUTH_DIR, dataset, other_device, "search_results.csv"))
                df2["error_rate"] = 1 - df2["accuracy"]/100.0
                pareto_indices2 = compute_pareto_front(df2)
                freq_pareto2 = compute_operation_frequencies(pareto_indices2, loader)
                
                jacc = jaccard_similarity(set(pareto_indices), set(pareto_indices2))
                edge_corrs = per_edge_corr(freq_pareto, freq_pareto2)
                
                cross_results.append({
                    "other_device": other_device,
                    "jaccard": jacc,
                    "edge_corrs": edge_corrs
                })
                
                # Plot per-edge correlation
                fig, ax = plt.subplots()
                ax.bar(range(len(edge_corrs)), edge_corrs)
                ax.set_xlabel("Edge")
                ax.set_ylabel("Correlation")
                ax.set_title(f"Per-edge op frequency correlation ({device} vs {other_device})")
                save_plot(fig, dataset, device, f"edge_corr_{other_device}")
                
            results["cross_device"] = cross_results
            
            # ---------------- Save results ----------------
            save_json(results, dataset, device)
            print(f"Results saved for {dataset}/{device}")

if __name__ == "__main__":
    main()