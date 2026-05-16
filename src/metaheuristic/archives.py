"""
src/metaheuristic/archives.py
FHArchive  : Fixed-Hypergrid Archive (Patil 2020).
CrowdingArchive : Bounded archive with crowding-distance elision.

Same public interface for both:
    archive.update(candidates)   # iterable of (arch_idx, acc, lat)
    archive.to_dataframe()       # → pd.DataFrame[arch_index, accuracy, latency]
    archive.size                 # → int
"""

import numpy as np
import pandas as pd


class FHArchive:
    """Fixed-Hypergrid Archive — 2-objective version.

    Parameters
    ----------
    ref_acc, ref_lat : float  Origin of each axis for cell indexing.
    delta_acc, delta_lat : float  Cell dimensions.
    max_cells : int  Max non-empty cells kept (default 2000).
    max_sols  : int  Max solutions per cell (default 5).
    """

    def __init__(self, ref_acc=0.0, ref_lat=0.0,
                 delta_acc=0.5, delta_lat=0.5,
                 max_cells=2000, max_sols=5):
        self.ref_acc   = ref_acc
        self.ref_lat   = ref_lat
        self.delta_acc = delta_acc
        self.delta_lat = delta_lat
        self.max_cells = max_cells
        self.max_sols  = max_sols
        self._cells:   dict = {}
        self._members: set  = set()

    def _cell_key(self, acc, lat):
        return (
            int(np.floor((acc - self.ref_acc) / self.delta_acc)),
            int(np.floor((lat - self.ref_lat) / self.delta_lat)),
        )

    @staticmethod
    def _dominates(acc_a, lat_a, acc_b, lat_b):
        return (acc_a >= acc_b and lat_a <= lat_b
                and (acc_a > acc_b or lat_a < lat_b))

    def update(self, candidates) -> None:
        for arch_idx, acc, lat in candidates:
            if arch_idx in self._members:
                continue
            if any(self._dominates(s[1], s[2], acc, lat)
                   for sols in self._cells.values() for s in sols):
                continue
            for key in list(self._cells):
                removed = [s for s in self._cells[key]
                           if self._dominates(acc, lat, s[1], s[2])]
                for s in removed:
                    self._members.discard(s[0])
                self._cells[key] = [s for s in self._cells[key]
                                    if not self._dominates(acc, lat, s[1], s[2])]
            key = self._cell_key(acc, lat)
            self._cells.setdefault(key, [])
            cell = self._cells[key]
            if len(cell) < self.max_sols:
                cell.append((arch_idx, acc, lat))
                self._members.add(arch_idx)
            else:
                worst_k = min(range(len(cell)), key=lambda k: cell[k][1])
                if acc > cell[worst_k][1]:
                    self._members.discard(cell[worst_k][0])
                    cell[worst_k] = (arch_idx, acc, lat)
                    self._members.add(arch_idx)
            if len(self._cells) > self.max_cells:
                self._cells = {k: v for k, v in self._cells.items() if v}

    def to_dataframe(self) -> pd.DataFrame:
        rows = [(arch_idx, acc, lat)
                for sols in self._cells.values()
                for arch_idx, acc, lat in sols]
        if not rows:
            return pd.DataFrame(columns=["arch_index", "accuracy", "latency"])
        return pd.DataFrame(rows, columns=["arch_index", "accuracy", "latency"])

    @property
    def size(self) -> int:
        return len(self._members)

    def __repr__(self):
        return (f"FHArchive(size={self.size}, "
                f"delta_acc={self.delta_acc}, delta_lat={self.delta_lat})")


class CrowdingArchive:
    """Bounded non-dominated archive with crowding-distance elision.

    Parameters
    ----------
    max_size : int  Maximum solutions kept (default 500).
    """

    def __init__(self, max_size: int = 500):
        self.max_size     = max_size
        self._archive:    list = []
        self._member_ids: set  = set()

    @staticmethod
    def _dominates(acc_a, lat_a, acc_b, lat_b):
        return (acc_a >= acc_b and lat_a <= lat_b
                and (acc_a > acc_b or lat_a < lat_b))

    def _crowding_distances(self) -> np.ndarray:
        n = len(self._archive)
        if n <= 2:
            return np.full(n, np.inf)
        cd   = np.zeros(n)
        data = np.array([(m[1], m[2]) for m in self._archive])
        for obj in range(2):
            order  = np.argsort(data[:, obj])
            spread = data[order[-1], obj] - data[order[0], obj]
            if spread == 0:
                continue
            cd[order[0]]  = np.inf
            cd[order[-1]] = np.inf
            for k in range(1, n - 1):
                cd[order[k]] += (data[order[k+1], obj] - data[order[k-1], obj]) / spread
        return cd

    def _prune(self) -> None:
        while len(self._archive) > self.max_size:
            cd = self._crowding_distances()
            finite_mask = np.isfinite(cd)
            worst = int(np.where(finite_mask, cd, np.inf).argmin()) \
                    if finite_mask.any() else 0
            evicted = self._archive.pop(worst)
            self._member_ids.discard(evicted[0])

    def update(self, candidates) -> None:
        for arch_idx, acc, lat in candidates:
            if arch_idx in self._member_ids:
                continue
            if any(self._dominates(m[1], m[2], acc, lat) for m in self._archive):
                continue
            kept = []
            for m in self._archive:
                if self._dominates(acc, lat, m[1], m[2]):
                    self._member_ids.discard(m[0])
                else:
                    kept.append(m)
            self._archive = kept
            self._archive.append((arch_idx, acc, lat))
            self._member_ids.add(arch_idx)
            if len(self._archive) > self.max_size:
                self._prune()

    def to_dataframe(self) -> pd.DataFrame:
        if not self._archive:
            return pd.DataFrame(columns=["arch_index", "accuracy", "latency"])
        return pd.DataFrame(self._archive,
                            columns=["arch_index", "accuracy", "latency"])

    @property
    def size(self) -> int:
        return len(self._archive)

    def __repr__(self):
        return f"CrowdingArchive(size={self.size}, max_size={self.max_size})"