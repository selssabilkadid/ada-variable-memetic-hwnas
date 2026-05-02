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


class Individual:

    def __init__(self, arch_index, ops):

        self.arch_index = arch_index
        self.ops = ops
        self.accuracy = None


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
                    "Accuracy column missing."
                )

        # Lookup

        self.lookup = {

            int(row["arch_index"]): row

            for _, row in self.df.iterrows()

        }

        print("[RE] Building ops lookup...")

        self.arch_ops = {

            idx: self._get_ops(idx)

            for idx in self.lookup

        }

        print("[RE] Building reverse map...")

        self.ops_to_index = {

            tuple(ops): idx

            for idx, ops in self.arch_ops.items()

        }

        print("[RE] Ready.")


    def reset(self):

        self.population.clear()


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


    def _evaluate(self, individual):

        row = self.lookup[
            individual.arch_index
        ]

        individual.accuracy = float(
            row["accuracy"]
        )


    def _make_individual(self, arch_index):

        ind = Individual(
            arch_index,
            self.arch_ops[arch_index]
        )

        self._evaluate(ind)

        return ind


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


    def _population_best_acc(self):

        return max(
            ind.accuracy
            for ind in self.population
        )


    def run(
        self,
        cycles,
        optimal_archs
    ):

        tracker = {

            g: {

                "eviction_gen": [],
                "acc_drop": []

            }

            for g in optimal_archs
        }

        best_alive_history = []
        best_ever_history  = []

        eviction_events = []

        all_indices = list(
            self.lookup.keys()
        )

        # ── Biased Initialization ──

        print("[RE] Biased initialization")

        for idx in optimal_archs:

            ind = self._make_individual(idx)

            self.population.append(ind)

        while len(self.population) < self.population_size:

            idx = random.choice(all_indices)

            ind = self._make_individual(idx)

            self.population.append(ind)

        best_ever = self._population_best_acc()

        # ── Evolution Loop ──

        for gen in range(1, cycles + 1):

            parent = self._sample_parent()

            child_idx = self._mutate(
                parent.arch_index
            )

            child = self._make_individual(
                child_idx
            )

            self.population.append(child)

            # Update best-ever

            best_alive = self._population_best_acc()

            if best_alive > best_ever:

                best_ever = best_alive

            # Aging eviction

            acc_before = best_alive

            removed = self.population.popleft()

            acc_after = self._population_best_acc()

            acc_drop = acc_before - acc_after

            # Track eviction

            if removed.arch_index in tracker:

                tracker[
                    removed.arch_index
                ]["eviction_gen"].append(gen)

                tracker[
                    removed.arch_index
                ]["acc_drop"].append(acc_drop)

                eviction_events.append({

                    "gen": gen,
                    "arch_index":
                        removed.arch_index,
                    "acc_drop":
                        acc_drop

                })

            best_alive_history.append(
                acc_after
            )

            best_ever_history.append(
                best_ever
            )

        return (

            tracker,
            best_alive_history,
            best_ever_history,
            eviction_events

        )