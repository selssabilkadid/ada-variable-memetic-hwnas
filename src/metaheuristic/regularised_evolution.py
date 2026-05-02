# import random
# import collections
# import numpy as np
# from pymoo.indicators.hv import HV

# NB201_OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
# NUM_EDGES = 6


# class Individual:
#     def __init__(self, arch_index, ops):
#         self.arch_index = arch_index
#         self.ops = ops
#         self.accuracy = None
#         self.latency = None


# class RegularizedEvolutionTracker:
#     def __init__(self, df, loader, device, population_size=50, sample_size=10, ref_point=None):
#         self.df = df.copy()
#         self.loader = loader
#         self.device = device
#         self.population_size = population_size
#         self.sample_size = sample_size

#         self.population = collections.deque()
#         self.history = []

#         # Normalize column names
#         if "accuracy" not in self.df.columns:
#             if "acc_cifar10" in self.df.columns:
#                 self.df.rename(columns={"acc_cifar10": "accuracy"}, inplace=True)
#             else:
#                 raise ValueError("DataFrame must contain 'accuracy' or 'acc_cifar10'")
#         self.df["error_rate"] = 1.0 - self.df["accuracy"] / 100.0

#         # Compute min/max for normalization
#         self.min_error = self.df["error_rate"].min()
#         self.max_error = self.df["error_rate"].max()
#         self.min_lat = self.df["latency"].min()
#         self.max_lat = self.df["latency"].max()

#         # HV setup
#         objs = self.df[["error_rate", "latency"]].values
#         # self.ref_point = np.max(objs, axis=0) * 1.1
#         self.ref_point = ref_point if ref_point is not None else np.max(objs, axis=0) * 1.1
#         self.hv_metric = HV(ref_point=self.ref_point)

#         # Lookup tables
#         self.lookup = {int(row["arch_index"]): row for _, row in self.df.iterrows()}

#         print("[RE] Building ops lookup table...")
#         self.arch_ops = {idx: self._get_ops(idx) for idx in self.lookup}

#         print("[RE] Building reverse ops→index map...")
#         self.ops_to_index = {tuple(ops): idx for idx, ops in self.arch_ops.items()}
#         print(f"[RE] Ready. {len(self.ops_to_index)} architectures indexed.")

#     def reset(self):
#         """Clear population and history between seeds."""
#         self.population.clear()
#         self.history = []

#     def _get_ops(self, arch_index):
#         config = self.loader.get_net_config(arch_index)
#         arch_str = config["arch_str"]
#         ops = []
#         nodes = arch_str.split('+')
#         for node in nodes:
#             parts = node.strip('|').split('|')
#             for p in parts:
#                 op = p.split('~')[0]
#                 ops.append(op)
#         return ops

#     def _evaluate(self, individual):
#         row = self.lookup[individual.arch_index]
#         individual.accuracy = float(row["accuracy"])
#         individual.latency = float(row["latency"])

#     def _make_individual(self, arch_index):
#         ind = Individual(arch_index, self.arch_ops[arch_index])
#         self._evaluate(ind)
#         return ind

#     def _mutate(self, parent_index):
#         parent_ops = list(self.arch_ops[parent_index])
#         edges = list(range(NUM_EDGES))
#         random.shuffle(edges)
#         for edge in edges:
#             current_op = parent_ops[edge]
#             for new_op in [op for op in NB201_OPS if op != current_op]:
#                 candidate = parent_ops.copy()
#                 candidate[edge] = new_op
#                 key = tuple(candidate)
#                 if key in self.ops_to_index:
#                     return self.ops_to_index[key]
#         return random.choice(list(self.lookup.keys()))

#     # ── New Chebyshev scalarization ─────────────────────────────────────────────
#     # def _scalarize(self, ind):
#     #     norm_error = (1 - ind.accuracy / 100.0 - self.min_error) / (self.max_error - self.min_error)
#     #     norm_lat = (ind.latency - self.min_lat) / (self.max_lat - self.min_lat)
#     #     return max(norm_error, norm_lat)  # minimize worst objective

#     def _fitness(self, ind):
#         return ind.accuracy

#     # def _sample_parent(self):
#     #     sample = random.sample(list(self.population), self.sample_size)
#     #     return min(sample, key=self._scalarize)
    
#     def _sample_parent(self):
#         sample = random.sample(list(self.population), self.sample_size)
#         return max(sample, key=self._fitness)

#     def _population_best_acc(self):
#         return max(ind.accuracy for ind in self.population)

#     def _compute_hv(self):
#         objs = np.array([[1 - ind.accuracy / 100.0, ind.latency] for ind in self.population])
#         return self.hv_metric(objs)

#     def run(self, cycles, god_archs):
#         tracker = {g: {"first_discovery_gen": None,
#                        "eviction_gen": [], "hv_drop": [], "rediscovery_gap": [],
#                        "last_death_gen": None, "currently_alive": False} for g in god_archs}
#         hv_history = []
#         eviction_events = []

#         # Initialize population
#         all_indices = list(self.lookup.keys())
#         for _ in range(self.population_size):
#             idx = random.choice(all_indices)
#             ind = self._make_individual(idx)
#             self.population.append(ind)
#             if idx in tracker:
#                 tracker[idx]["first_discovery_gen"] = 0
#                 tracker[idx]["currently_alive"] = True

#         hv_history.append(self._compute_hv())

#         # Evolution loop
#         for gen in range(1, cycles + 1):
#             parent = self._sample_parent()
#             child_idx = self._mutate(parent.arch_index)
#             child = self._make_individual(child_idx)
#             self.population.append(child)

#             # Discovery tracking
#             if child_idx in tracker:
#                 t = tracker[child_idx]
#                 if t["first_discovery_gen"] is None:
#                     t["first_discovery_gen"] = gen
#                 if not t["currently_alive"]:
#                     if t["last_death_gen"] is not None:
#                         gap = gen - t["last_death_gen"]
#                         t["rediscovery_gap"].append(gap)
#                     t["currently_alive"] = True
#                     t["last_death_gen"] = None

#             # Aging eviction
#             hv_before = self._compute_hv()
#             removed = self.population.popleft()
#             hv_after = self._compute_hv()

#             # Eviction tracking
#             if removed.arch_index in tracker:
#                 t = tracker[removed.arch_index]
#                 hv_drop = hv_before - hv_after
#                 t["eviction_gen"].append(gen)
#                 t["hv_drop"].append(hv_drop)
#                 t["currently_alive"] = False
#                 t["last_death_gen"] = gen
#                 eviction_events.append({"gen": gen, "arch_index": removed.arch_index,
#                                         "hv_before": hv_before, "hv_after": hv_after,
#                                         "hv_drop": hv_drop})

#             hv_history.append(hv_after)

#         return tracker, hv_history, eviction_events

import random
import collections
import numpy as np

NB201_OPS = [
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3"
]

NUM_EDGES = 6


# ─────────────────────────────────────────────
# Individual
# ─────────────────────────────────────────────

class Individual:

    def __init__(self, arch_index, ops):

        self.arch_index = arch_index
        self.ops = ops

        self.accuracy = None


# ─────────────────────────────────────────────
# Regularized Evolution Tracker
# Accuracy-only + Biased Initialization
# ─────────────────────────────────────────────

class RegularizedEvolutionTracker:

    def __init__(
        self,
        df,
        loader,
        population_size=50,
        sample_size=10
    ):

        self.df = df.copy()

        self.loader = loader

        self.population_size = population_size
        self.sample_size = sample_size

        self.population = collections.deque()

        # Normalize accuracy column

        if "accuracy" not in self.df.columns:

            if "acc_cifar10" in self.df.columns:

                self.df.rename(
                    columns={"acc_cifar10": "accuracy"},
                    inplace=True
                )

            else:

                raise ValueError(
                    "DataFrame must contain accuracy column"
                )

        # Lookup table

        self.lookup = {
            int(row["arch_index"]): row
            for _, row in self.df.iterrows()
        }

        print("[RE] Building ops lookup...")

        self.arch_ops = {
            idx: self._get_ops(idx)
            for idx in self.lookup
        }

        print("[RE] Building reverse mapping...")

        self.ops_to_index = {
            tuple(ops): idx
            for idx, ops in self.arch_ops.items()
        }

        print(
            f"[RE] Ready: {len(self.ops_to_index)} architectures indexed."
        )


    # ─────────────────────────────────────────

    def reset(self):

        self.population.clear()


    # ─────────────────────────────────────────

    def _get_ops(self, arch_index):

        config = self.loader.get_net_config(
            arch_index
        )

        arch_str = config["arch_str"]

        ops = []

        nodes = arch_str.split("+")

        for node in nodes:

            parts = node.strip("|").split("|")

            for p in parts:

                op = p.split("~")[0]

                ops.append(op)

        return ops


    # ─────────────────────────────────────────

    def _evaluate(self, individual):

        row = self.lookup[
            individual.arch_index
        ]

        individual.accuracy = float(
            row["accuracy"]
        )


    # ─────────────────────────────────────────

    def _make_individual(self, arch_index):

        ind = Individual(
            arch_index,
            self.arch_ops[arch_index]
        )

        self._evaluate(ind)

        return ind


    # ─────────────────────────────────────────
    # Mutation (NAS-Bench-201 valid mutation)
    # ─────────────────────────────────────────

    def _mutate(self, parent_index):

        parent_ops = list(
            self.arch_ops[parent_index]
        )

        edges = list(
            range(NUM_EDGES)
        )

        random.shuffle(edges)

        for edge in edges:

            current_op = parent_ops[edge]

            for new_op in NB201_OPS:

                if new_op == current_op:

                    continue

                candidate = parent_ops.copy()

                candidate[edge] = new_op

                key = tuple(candidate)

                if key in self.ops_to_index:

                    return self.ops_to_index[key]

        return random.choice(
            list(self.lookup.keys())
        )


    # ─────────────────────────────────────────
    # Accuracy Fitness
    # ─────────────────────────────────────────

    def _fitness(self, ind):

        return ind.accuracy


    def _sample_parent(self):

        sample = random.sample(
            list(self.population),
            self.sample_size
        )

        return max(
            sample,
            key=self._fitness
        )


    # ─────────────────────────────────────────
    # Best Accuracy
    # ─────────────────────────────────────────

    def _population_best_acc(self):

        return max(
            ind.accuracy
            for ind in self.population
        )


    # ─────────────────────────────────────────
    # Main Evolution Loop
    # ─────────────────────────────────────────

    def run(
        self,
        cycles,
        optimal_archs
    ):

        tracker = {

            g: {

                "first_discovery_gen": None,
                "eviction_gen": [],
                "acc_drop": [],
                "rediscovery_gap": [],

                "last_death_gen": None,
                "currently_alive": False

            }

            for g in optimal_archs
        }

        acc_history = []

        eviction_events = []

        all_indices = list(
            self.lookup.keys()
        )

        # ─────────────────────────────
        # Biased Initialization
        # ─────────────────────────────

        print("[RE] Biased initialization...")

        for idx in optimal_archs:

            ind = self._make_individual(idx)

            self.population.append(ind)

            tracker[idx][
                "first_discovery_gen"
            ] = 0

            tracker[idx][
                "currently_alive"
            ] = True


        while len(self.population) < self.population_size:

            idx = random.choice(
                all_indices
            )

            ind = self._make_individual(idx)

            self.population.append(ind)


        acc_history.append(
            self._population_best_acc()
        )


        # ─────────────────────────────
        # Evolution Loop
        # ─────────────────────────────

        for gen in range(1, cycles + 1):

            parent = self._sample_parent()

            child_idx = self._mutate(
                parent.arch_index
            )

            child = self._make_individual(
                child_idx
            )

            self.population.append(child)

            # Discovery tracking

            if child_idx in tracker:

                t = tracker[child_idx]

                if t["first_discovery_gen"] is None:

                    t["first_discovery_gen"] = gen

                if not t["currently_alive"]:

                    if t["last_death_gen"] is not None:

                        gap = (
                            gen
                            - t["last_death_gen"]
                        )

                        t[
                            "rediscovery_gap"
                        ].append(gap)

                    t["currently_alive"] = True

                    t["last_death_gen"] = None


            # ─────────────────────
            # Aging Eviction
            # ─────────────────────

            acc_before = self._population_best_acc()

            removed = self.population.popleft()

            acc_after = self._population_best_acc()

            acc_drop = acc_before - acc_after


            if removed.arch_index in tracker:

                t = tracker[
                    removed.arch_index
                ]

                t["eviction_gen"].append(gen)

                t["acc_drop"].append(
                    acc_drop
                )

                t["currently_alive"] = False

                t["last_death_gen"] = gen

                eviction_events.append({

                    "gen": gen,

                    "arch_index":
                        removed.arch_index,

                    "acc_before":
                        acc_before,

                    "acc_after":
                        acc_after,

                    "acc_drop":
                        acc_drop

                })


            acc_history.append(
                self._population_best_acc()
            )


        return (
            tracker,
            acc_history,
            eviction_events
        )