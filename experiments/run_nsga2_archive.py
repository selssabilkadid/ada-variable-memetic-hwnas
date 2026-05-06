"""
experiments/run_nsga2_fh_ca.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-run experiment: NSGA-II vs NSGA-II-CA contains NSGA-II-FH comparison (commented out for now, but can be re-enabled by uncommenting the relevant lines).
across all (dataset × hw_platform) combinations.
Each algorithm gets its own independent eval cache to prevent result bleed.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.benchmark.nasbench201csv_loader import NASBench201CSVLoader
from src.benchmark.hwnas_loader          import HWNASLoader
from src.metaheuristic.archives          import CrowdingArchive # FHArchive
from src.metaheuristic.nsga2_with_archive import (
    run_nsga2_with_archive,
    compute_pareto_df,
)
from src.evaluation.metrics import (
    hypervolume_2d_arr,
    generational_distance,
    inverted_generational_distance,
)

DATASETS     = ["cifar10", "cifar100", "ImageNet16-120"]
HW_PLATFORMS = ["raspi4", "edgegpu", "edgetpu", "pixel3", "eyeriss", "fpga"]

SEED              = 42
SEARCH_SPACE_SIZE = 15_625
POP_SIZE          = 100
N_GENERATIONS     = 50
CROSSOVER_PROB    = 0.9
MUTATION_PROB     = 0.1
TOURNAMENT_K      = 2

# FH params
# FH_REF_ACC   = 0.0
# FH_REF_LAT   = 0.0
# FH_DELTA_ACC = 0.5
# FH_DELTA_LAT = 0.5
# FH_MAX_CELLS = 50 #2000 lowered it to have fh and ca diffrent
# FH_MAX_SOLS  = 1 #5

# CA params
CA_MAX_SIZE = 20 #500

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, "..", "results", "results_nsga2_ca")


def best_by_ideal(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    rng_acc = d["accuracy"].max() - d["accuracy"].min() + 1e-8
    rng_lat = d["latency"].max()  - d["latency"].min()  + 1e-8
    d["acc_norm"] = (d["accuracy"] - d["accuracy"].min()) / rng_acc
    d["lat_norm"] = (d["latency"]  - d["latency"].min())  / rng_lat
    d["dist"]     = np.sqrt((d["acc_norm"] - 1.0) ** 2 + d["lat_norm"] ** 2)
    return d.loc[d["dist"].idxmin()]


def make_eval_fn(nb201_loader, hw_loader, dataset: str, hw_platform: str):
    """Return a fresh cached eval function — call once per algorithm per run."""
    cache: dict[int, tuple[float, float]] = {}
    def eval_fn(arch_index: int) -> tuple[float, float]:
        if arch_index not in cache:
            acc = nb201_loader.get_accuracy(arch_index, dataset)
            lat = hw_loader.get_latency(arch_index, hw_platform=hw_platform,
                                        dataset=dataset)
            cache[arch_index] = (float(acc), float(lat))
        return cache[arch_index]
    return eval_fn


def build_exhaustive_df(eval_fn, search_space_size=SEARCH_SPACE_SIZE) -> pd.DataFrame:
    rows = [{"arch_index": i, "accuracy": eval_fn(i)[0], "latency": eval_fn(i)[1]}
            for i in range(search_space_size)]
    return pd.DataFrame(rows)


def plot_run(dataset, hw_platform, df_all, true_pareto,
             nsga2_pareto, ca_pareto,  # fh_pareto, hist_fh,
             hist_nsga2, hist_ca, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{dataset} / {hw_platform}", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.scatter(df_all["latency"], df_all["accuracy"],
               s=4, alpha=0.1, color="lightgray", label="All architectures")
    for df_pf, color, marker, lbl in [
        (true_pareto,  "steelblue",  "o", f"True Pareto (n={len(true_pareto)})"),
        (ca_pareto,    "seagreen",   "^", f"NSGA-II-CA  (n={len(ca_pareto)})"),
        # (fh_pareto,    "darkorange", "D", f"NSGA-II-FH  (n={len(fh_pareto)})"),
        (nsga2_pareto, "tomato",     "s", f"NSGA-II     (n={len(nsga2_pareto)})"),
    ]:
        s = df_pf.sort_values("latency")
        ax.plot(s["latency"], s["accuracy"], marker=marker, color=color,
                linestyle="--", markersize=5, linewidth=1.5, label=lbl)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Pareto Fronts")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    h_ns = pd.DataFrame(hist_nsga2)
    # h_fh = pd.DataFrame(hist_fh)
    h_ca = pd.DataFrame(hist_ca)
    ax2.plot(h_ns["generation"], h_ns["pareto_front_size"],
             color="tomato",     linewidth=2, label="NSGA-II (pop front)")
    # ax2.plot(h_fh["generation"], h_fh["archive_size"],
    #          color="darkorange", linewidth=2, label="NSGA-II-FH (archive)")
    ax2.plot(h_ca["generation"], h_ca["archive_size"],
             color="seagreen",   linewidth=2, linestyle="-.",
             label="NSGA-II-CA (archive)")
    ax2.axhline(len(true_pareto), color="steelblue", linestyle="--",
                label=f"True Pareto ({len(true_pareto)})")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("# Non-dominated solutions")
    ax2.set_title("Convergence")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()


def main() -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    np.random.seed(SEED)

    print("[INFO] Loading benchmark loaders …")
    nb201_loader = NASBench201CSVLoader()
    hw_loader    = HWNASLoader()

    run_combinations = [(ds, hw) for ds in DATASETS for hw in HW_PLATFORMS]
    total            = len(run_combinations)

    print(f"Multi-run grid : {len(DATASETS)} datasets × {len(HW_PLATFORMS)} devices = {total} combinations")
    print(f"NSGA-II   : pop={POP_SIZE}, generations={N_GENERATIONS}")
    # print(f"FH archive: delta_acc={FH_DELTA_ACC}%, delta_lat={FH_DELTA_LAT}ms")
    print(f"CA archive: max_size={CA_MAX_SIZE} (no range params needed)\n")

    all_results: list[dict] = []
    algos = ["NSGA-II", "NSGA-II-FH", "NSGA-II-CA"]

    for run_idx, (dataset, hw_platform) in enumerate(run_combinations):
        print(f"[{run_idx + 1:02d}/{total}] {dataset}/{hw_platform}")

        run_dir = os.path.join(OUTPUT_ROOT, dataset, hw_platform)
        os.makedirs(run_dir, exist_ok=True)

        # ── One independent eval cache per algorithm — prevents result bleed ──
        eval_fn_gt    = make_eval_fn(nb201_loader, hw_loader, dataset, hw_platform)
        eval_fn_nsga2 = make_eval_fn(nb201_loader, hw_loader, dataset, hw_platform)
        # eval_fn_fh    = make_eval_fn(nb201_loader, hw_loader, dataset, hw_platform)
        eval_fn_ca    = make_eval_fn(nb201_loader, hw_loader, dataset, hw_platform)

        # ── Ground-truth ───────────────────────────────────────────────────
        t_gt        = time.time()
        df_all      = build_exhaustive_df(eval_fn_gt)
        true_pareto = compute_pareto_df(df_all)
        ref_lat     = df_all["latency"].max()  * 1.1
        ref_acc     = df_all["accuracy"].min() * 0.9
        ref_point   = (ref_lat, ref_acc)
        hv_true     = hypervolume_2d_arr(
            true_pareto[["accuracy", "latency"]].values, ref_point)
        print(f"    True Pareto: {len(true_pareto)} pts  |  "
              f"HV={hv_true:.4f}  ({time.time()-t_gt:.1f}s)")

        # ── NSGA-II baseline ───────────────────────────────────────────────
        t0 = time.time()
        pop_nsga2, hist_nsga2 = run_nsga2_with_archive(
            eval_fn=eval_fn_nsga2, archive=None,
            pop_size=POP_SIZE, n_generations=N_GENERATIONS,
            crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB,
            tournament_k=TOURNAMENT_K, search_space_size=SEARCH_SPACE_SIZE,
            seed=SEED,
        )
        t_nsga2 = time.time() - t0
        df_nsga2_pop = pd.DataFrame([
            {"arch_index": i,
             "accuracy":   eval_fn_nsga2(i)[0],
             "latency":    eval_fn_nsga2(i)[1]}
            for i in pop_nsga2
        ])
        nsga2_pareto = compute_pareto_df(df_nsga2_pop)

        # ── NSGA-II-FH ────────────────────────────────────────────────────
        # fh_archive = FHArchive(
        #     ref_acc=FH_REF_ACC, ref_lat=FH_REF_LAT,
        #     delta_acc=FH_DELTA_ACC, delta_lat=FH_DELTA_LAT,
        #     max_cells=FH_MAX_CELLS, max_sols=FH_MAX_SOLS,
        # )
        # t0 = time.time()
        # _, hist_fh = run_nsga2_with_archive(
        #     eval_fn=eval_fn_fh, archive=fh_archive,
        #     pop_size=POP_SIZE, n_generations=N_GENERATIONS,
        #     crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB,
        #     tournament_k=TOURNAMENT_K, search_space_size=SEARCH_SPACE_SIZE,
        #     seed=SEED,
        # )
        # t_fh      = time.time() - t0
        # fh_pareto = compute_pareto_df(fh_archive.to_dataframe())

        # ── NSGA-II-CA ────────────────────────────────────────────────────
        ca_archive = CrowdingArchive(max_size=CA_MAX_SIZE)
        t0 = time.time()
        _, hist_ca = run_nsga2_with_archive(
            eval_fn=eval_fn_ca, archive=ca_archive,
            pop_size=POP_SIZE, n_generations=N_GENERATIONS,
            crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB,
            tournament_k=TOURNAMENT_K, search_space_size=SEARCH_SPACE_SIZE,
            seed=SEED,
        )
        t_ca      = time.time() - t0
        ca_pareto = compute_pareto_df(ca_archive.to_dataframe())

        # ── Metrics ────────────────────────────────────────────────────────
        true_idx = set(true_pareto["arch_index"])

        def _metrics(pf: pd.DataFrame):
            hv  = hypervolume_2d_arr(pf[["accuracy", "latency"]].values, ref_point)
            cov = len(set(pf["arch_index"]) & true_idx) / len(true_idx) * 100
            gd  = generational_distance(pf, true_pareto)
            igd = inverted_generational_distance(pf, true_pareto)
            return hv, cov, gd, igd

        hv_nsga2, cov_nsga2, gd_nsga2, igd_nsga2 = _metrics(nsga2_pareto)
        # hv_fh,    cov_fh,    gd_fh,    igd_fh    = _metrics(fh_pareto)
        hv_ca,    cov_ca,    gd_ca,    igd_ca    = _metrics(ca_pareto)

        print(f"    NSGA-II    : Pareto={len(nsga2_pareto):3d}  "
              f"HV={hv_nsga2:.4f} ({hv_nsga2/hv_true*100:.1f}%)  "
              f"GD={gd_nsga2:.4f}  IGD={igd_nsga2:.4f}  t={t_nsga2:.1f}s")
        # print(f"    NSGA-II-FH : Pareto={len(fh_pareto):3d}  "
        #       f"HV={hv_fh:.4f} ({hv_fh/hv_true*100:.1f}%)  "
        #       f"GD={gd_fh:.4f}  IGD={igd_fh:.4f}  t={t_fh:.1f}s")
        print(f"    NSGA-II-CA : Pareto={len(ca_pareto):3d}  "
              f"HV={hv_ca:.4f} ({hv_ca/hv_true*100:.1f}%)  "
              f"GD={gd_ca:.4f}  IGD={igd_ca:.4f}  t={t_ca:.1f}s")

        # ── Save CSVs ──────────────────────────────────────────────────────
        true_pareto.to_csv(os.path.join(run_dir, "true_pareto.csv"),   index=False)
        nsga2_pareto.to_csv(os.path.join(run_dir, "nsga2_pareto.csv"), index=False)
        # fh_pareto.to_csv(os.path.join(run_dir, "fh_pareto.csv"),       index=False)
        ca_pareto.to_csv(os.path.join(run_dir, "ca_pareto.csv"),       index=False)
        pd.DataFrame(hist_nsga2).to_csv(
            os.path.join(run_dir, "hist_nsga2.csv"), index=False)
        # pd.DataFrame(hist_fh).to_csv(
        #     os.path.join(run_dir, "hist_fh.csv"), index=False)
        pd.DataFrame(hist_ca).to_csv(
            os.path.join(run_dir, "hist_ca.csv"), index=False)

        plot_run(
            dataset=dataset, hw_platform=hw_platform, df_all=df_all,
            true_pareto=true_pareto, nsga2_pareto=nsga2_pareto,
             ca_pareto=ca_pareto, #fh_pareto=fh_pareto,
            hist_nsga2=hist_nsga2,  hist_ca=hist_ca, # hist_fh=hist_fh,
            output_path=os.path.join(run_dir, "pareto_comparison.png"),
        )

        for algo, pf, hv, cov, gd, igd, t_run in [
            ("NSGA-II",    nsga2_pareto, hv_nsga2, cov_nsga2, gd_nsga2, igd_nsga2, t_nsga2),
            # ("NSGA-II-FH", fh_pareto,   hv_fh,    cov_fh,    gd_fh,    igd_fh,    t_fh),
            ("NSGA-II-CA", ca_pareto,   hv_ca,    cov_ca,    gd_ca,    igd_ca,    t_ca),
            ("True",       true_pareto,  hv_true,  100.0,     0.0,      0.0,       None),
        ]:
            best = best_by_ideal(pf)
            all_results.append({
                "dataset":           dataset,
                "hw_platform":       hw_platform,
                "algorithm":         algo,
                "pareto_front_size": len(pf),
                "hv":                round(hv, 4),
                "hv_ratio_pct":      round(hv / hv_true * 100, 2) if hv_true > 0 else 0.0,
                "coverage_pct":      round(cov, 2),
                "gd":                round(gd,  6) if not np.isnan(gd)  else None,
                "igd":               round(igd, 6) if not np.isnan(igd) else None,
                "best_arch_index":   int(best["arch_index"]),
                "best_accuracy":     round(float(best["accuracy"]), 4),
                "best_latency":      round(float(best["latency"]),  4),
                "wall_time_s":       round(t_run, 2) if t_run is not None else None,
            })
        print()

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(os.path.join(OUTPUT_ROOT, "multirun_summary.csv"), index=False)
    print(f"\n✓ All {total} runs complete. Summary saved.")

    # ── HV heatmaps per algorithm ─────────────────────────────────────────────
    for algo, cmap in [("NSGA-II",    "YlOrRd"),
                    #    ("NSGA-II-FH", "YlOrBr"),
                       ("NSGA-II-CA", "YlGn")]:
        sub   = summary_df[summary_df["algorithm"] == algo]
        pivot = sub.pivot(index="hw_platform", columns="dataset", values="hv_ratio_pct")
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=11)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=11)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=10, color="black" if val > 30 else "gray")
        plt.colorbar(im, ax=ax, label="HV Ratio (%)")
        ax.set_title(f"{algo} — Hypervolume Ratio vs. True Pareto Front (%)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        out = os.path.join(OUTPUT_ROOT,
                           f"heatmap_hv_{algo.replace('-','_').lower()}.png")
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")

    # ── FH vs CA side-by-side heatmap ─────────────────────────────────────────
    # fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    # fig.suptitle("HV Ratio (%) — NSGA-II-FH vs NSGA-II-CA",
    #              fontsize=13, fontweight="bold")
    # for ax, algo, cmap in zip(axes,
    #                            ["NSGA-II-FH", "NSGA-II-CA"],
    #                            ["YlOrBr",     "YlGn"]):
    #     sub   = summary_df[summary_df["algorithm"] == algo]
    #     pivot = sub.pivot(index="hw_platform", columns="dataset", values="hv_ratio_pct")
    #     im = ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    #     ax.set_xticks(range(len(pivot.columns)))
    #     ax.set_xticklabels(pivot.columns, fontsize=10)
    #     ax.set_yticks(range(len(pivot.index)))
    #     ax.set_yticklabels(pivot.index, fontsize=10)
    #     for i in range(len(pivot.index)):
    #         for j in range(len(pivot.columns)):
    #             val = pivot.values[i, j]
    #             ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
    #                     fontsize=9, color="black" if val > 30 else "gray")
    #     plt.colorbar(im, ax=ax, label="HV Ratio (%)")
    #     ax.set_title(algo, fontsize=12, fontweight="bold")
    # plt.tight_layout()
    # out = os.path.join(OUTPUT_ROOT, "heatmap_fh_vs_ca_comparison.png")
    # plt.savefig(out, dpi=130, bbox_inches="tight")
    # plt.close()
    # print(f"Saved: {out}")

    # ── Pareto size bar chart ─────────────────────────────────────────────────
    fig, axes2 = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 6))
    colors = {"True": "steelblue", "NSGA-II": "tomato",
              "NSGA-II-FH": "darkorange", "NSGA-II-CA": "seagreen"}
    for ax, ds in zip(axes2, DATASETS):
        sub   = summary_df[summary_df["dataset"] == ds]
        x     = np.arange(len(HW_PLATFORMS))
        width = 0.22
        for k, algo in enumerate(["True", "NSGA-II",  "NSGA-II-CA"]): # "NSGA-II-FH",
            vals = [sub[(sub["hw_platform"] == d) & (sub["algorithm"] == algo)][
                        "pareto_front_size"].values[0] for d in HW_PLATFORMS]
            ax.bar(x + k * width, vals, width,
                   label=algo, color=colors[algo], alpha=0.85)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(HW_PLATFORMS, rotation=30, ha="right", fontsize=9)
        ax.set_title(ds, fontsize=12, fontweight="bold")
        ax.set_ylabel("Pareto Front Size")
        if ds == DATASETS[0]:
            ax.legend(fontsize=9)
    plt.suptitle("Pareto Front Size — All Datasets × Devices",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_ROOT, "barplot_pareto_size.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── GD / IGD heatmaps ─────────────────────────────────────────────────────
    for metric, title in [("gd",  "GD (lower = better convergence)"),
                           ("igd", "IGD (lower = convergence + spread)")]:
        fig, axes3 = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(title, fontsize=13, fontweight="bold")
        vmax = summary_df[summary_df["algorithm"].isin(algos)][metric].max()
        for ax, algo in zip(axes3, algos):
            sub   = summary_df[summary_df["algorithm"] == algo]
            pivot = sub.pivot(index="hw_platform", columns="dataset", values=metric)
            im = ax.imshow(pivot.values, cmap="YlOrRd_r",
                           vmin=0, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, fontsize=10, rotation=15)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=10)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                            fontsize=9,
                            color="black" if val < vmax * 0.6 else "white")
            plt.colorbar(im, ax=ax)
            ax.set_title(algo, fontsize=11, fontweight="bold")
        plt.tight_layout()
        out = os.path.join(OUTPUT_ROOT, f"heatmap_{metric}_comparison.png")
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")

    # ── Final mean metrics summary ─────────────────────────────────────────────
    print("\n── Mean metrics (all 18 dataset × device combinations) ──")
    print(f"{'Metric':<14} {'NSGA-II':>12}{'NSGA-II-CA':>14}") # {'NSGA-II-FH':>14} 
    print("-" * 58)
    for metric, label, fmt in [
        ("hv_ratio_pct", "HV ratio %", ".2f"),
        ("coverage_pct", "Coverage %", ".2f"),
        ("gd",           "GD",         ".6f"),
        ("igd",          "IGD",        ".6f"),
    ]:
        vals = {a: summary_df[summary_df["algorithm"] == a][metric].mean()
                for a in algos}
        print(f"{label:<14} {vals['NSGA-II']:>12{fmt}} {vals['NSGA-II-CA']:>14{fmt}}") # f"{vals['NSGA-II-FH']:>14{fmt}}"

    print(f"\nAll outputs saved under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()