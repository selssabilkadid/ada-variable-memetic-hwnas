"""
src/metaheuristic/regularised_evolution.py
==========================================
Regularized Evolution with Amnesia Tracker.

Fixes over previous version:
- Real NAS-Bench-201 mutation (change one edge op, look up resulting arch)
- Proper HV-drop logging at eviction time
- Tracks first_discovery_gen, eviction_gen, hv_drop, rediscovery_gap
- op_to_index map built once at init for O(1) mutation lookup
- Column names aligned with actual CSV (acc_cifar10, not accuracy)
"""

import random
import collections
import numpy as np
from pymoo.indicators.hv import HV


# NAS-Bench-201 operation set (fixed, 5 ops)
NB201_OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
NUM_EDGES = 6  # NAS-Bench-201 cell has 6 edges


class Individual:
    def __init__(self, arch_index, ops):
        self.arch_index = arch_index
        self.ops = ops          # list of 6 op strings
        self.accuracy = None
        self.latency = None


class RegularizedEvolutionTracker:
    def __init__(self, df, loader, device, population_size=50, sample_size=10):
        """
        df      : DataFrame with columns [arch_index, acc_cifar10, latency]
        loader  : HWNASLoader instance (needed for get_arch_ops / get_net_config)
        device  : string, e.g. 'edgegpu'
        """
        self.df = df
        self.loader = loader
        self.device = device
        self.population_size = population_size
        self.sample_size = sample_size

        self.population = collections.deque()
        self.history = []

        # ── Align column names ──────────────────────────────────────────────
        # CSV uses acc_cifar10; rename to accuracy for internal use
        if "accuracy" not in self.df.columns:
            if "acc_cifar10" in self.df.columns:
                self.df = self.df.rename(columns={"acc_cifar10": "accuracy"})
            else:
                raise ValueError("DataFrame must have 'accuracy' or 'acc_cifar10' column")

        self.df = self.df.copy()
        self.df["error_rate"] = 1.0 - self.df["accuracy"] / 100.0

        # ── HV setup ────────────────────────────────────────────────────────
        objs = self.df[["error_rate", "latency"]].values
        self.ref_point = np.max(objs, axis=0) * 1.1
        self.hv_metric = HV(ref_point=self.ref_point)

        # ── Fast lookup: arch_index → row ───────────────────────────────────
        self.lookup = {
            int(row["arch_index"]): row
            for _, row in self.df.iterrows()
        }

        # ── Build ops lookup: arch_index → [op0, op1, ..., op5] ────────────
        print("[RE] Building ops lookup table (this may take ~30s)...")
        self.arch_ops = {}
        for idx in self.lookup:
            self.arch_ops[idx] = self._get_ops(idx)

        # ── Build reverse map: ops tuple → arch_index ───────────────────────
        print("[RE] Building reverse ops→index map...")
        self.ops_to_index = {
            tuple(ops): idx
            for idx, ops in self.arch_ops.items()
        }
        print(f"[RE] Ready. {len(self.ops_to_index)} architectures indexed.")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_ops(self, arch_index):
        """Parse NAS-Bench-201 arch string into list of 6 op strings."""
        config = self.loader.get_net_config(arch_index)
        arch_str = config["arch_str"]
        ops = []
        nodes = arch_str.split('+')
        for node in nodes:
            parts = node.strip('|').split('|')
            for p in parts:
                op = p.split('~')[0]
                ops.append(op)
        return ops  # length 6

    def _evaluate(self, individual):
        row = self.lookup[individual.arch_index]
        individual.accuracy = float(row["accuracy"])
        individual.latency = float(row["latency"])

    def _make_individual(self, arch_index):
        ind = Individual(arch_index, self.arch_ops[arch_index])
        self._evaluate(ind)
        return ind

    # ── Real mutation ────────────────────────────────────────────────────────

    def _mutate(self, parent_index):
        """
        NAS-Bench-201 mutation: copy parent ops, change exactly one edge
        to a different op, look up resulting arch_index.
        Falls back to random arch if mutant not in search space (shouldn't happen).
        """
        parent_ops = list(self.arch_ops[parent_index])

        # Try random edge mutations until we find a valid neighbour
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        for edge in edges:
            current_op = parent_ops[edge]
            other_ops = [op for op in NB201_OPS if op != current_op]
            random.shuffle(other_ops)
            for new_op in other_ops:
                candidate = parent_ops.copy()
                candidate[edge] = new_op
                key = tuple(candidate)
                if key in self.ops_to_index:
                    return self.ops_to_index[key]

        # Fallback (should never trigger on NB201)
        return random.choice(list(self.lookup.keys()))

    # ── Tournament parent selection ──────────────────────────────────────────

    def _sample_parent(self):
        sample = random.sample(list(self.population), self.sample_size)
        # Minimise weighted sum (scalarization); adjust weights if needed
        return min(sample, key=lambda x: (1 - x.accuracy / 100.0) + x.latency)

    # ── HV of current population ─────────────────────────────────────────────

    def _compute_hv(self):
        objs = np.array([[1 - ind.accuracy / 100.0, ind.latency]
                         for ind in self.population])
        return self.hv_metric(objs)

    # ── Main run ─────────────────────────────────────────────────────────────

    def run(self, cycles, god_archs):
        """
        Run RE for `cycles` generations, tracking god architectures.

        Returns
        -------
        tracker : dict
            Per god-arch stats:
              first_discovery_gen  : int or None
              eviction_gen         : list[int]
              hv_drop              : list[float]  (HV lost at each eviction)
              rediscovery_gap      : list[int]    (gens between eviction & rediscovery)
              currently_alive      : bool

        hv_history : list[float]
            Population HV at each generation.

        eviction_events : list[dict]
            One entry per eviction of any god arch, with gen, arch_index, hv_before, hv_after.
        """

        tracker = {
            g: {
                "first_discovery_gen": None,
                "eviction_gen": [],
                "hv_drop": [],
                "rediscovery_gap": [],
                "last_death_gen": None,   # internal use
                "currently_alive": False,
            }
            for g in god_archs
        }

        hv_history = []
        eviction_events = []

        # ── Initialise population ────────────────────────────────────────────
        all_indices = list(self.lookup.keys())
        for _ in range(self.population_size):
            idx = random.choice(all_indices)
            ind = self._make_individual(idx)
            self.population.append(ind)

            # Check if god arch landed in initial population
            if idx in tracker:
                if tracker[idx]["first_discovery_gen"] is None:
                    tracker[idx]["first_discovery_gen"] = 0
                tracker[idx]["currently_alive"] = True

        hv_history.append(self._compute_hv())

        # ── Evolution loop ───────────────────────────────────────────────────
        for gen in range(1, cycles + 1):

            parent = self._sample_parent()
            child_idx = self._mutate(parent.arch_index)
            child = self._make_individual(child_idx)
            self.population.append(child)

            # ── Discovery tracking ───────────────────────────────────────────
            if child_idx in tracker:
                t = tracker[child_idx]
                if t["first_discovery_gen"] is None:
                    t["first_discovery_gen"] = gen

                if not t["currently_alive"]:
                    # Rediscovery
                    if t["last_death_gen"] is not None:
                        gap = gen - t["last_death_gen"]
                        t["rediscovery_gap"].append(gap)
                    t["currently_alive"] = True
                    t["last_death_gen"] = None

            # ── Aging eviction ───────────────────────────────────────────────
            hv_before = self._compute_hv()
            removed = self.population.popleft()
            hv_after = self._compute_hv()

            # ── Eviction tracking ────────────────────────────────────────────
            if removed.arch_index in tracker:
                t = tracker[removed.arch_index]
                hv_drop = hv_before - hv_after
                t["eviction_gen"].append(gen)
                t["hv_drop"].append(hv_drop)
                t["currently_alive"] = False
                t["last_death_gen"] = gen

                eviction_events.append({
                    "gen": gen,
                    "arch_index": removed.arch_index,
                    "hv_before": hv_before,
                    "hv_after": hv_after,
                    "hv_drop": hv_drop,
                })

            hv_history.append(hv_after)

            if gen % 100 == 0:
                alive = sum(1 for g in god_archs if tracker[g]["currently_alive"])
                print(f"  Gen {gen:4d} | HV={hv_after:.4f} | "
                      f"God archs alive: {alive}/{len(god_archs)}")

        return tracker, hv_history, eviction_events