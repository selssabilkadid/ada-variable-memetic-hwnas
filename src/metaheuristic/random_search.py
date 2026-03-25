import numpy as np
from tqdm import tqdm
from src.benchmark.hwnas_loader import HWNASLoader
from src.evaluation.metrics import compute_pareto_front


class RandomSearch:
    """
    Baseline: pure random search over the HW-NAS-Bench search space.
    Every metaheuristic you implement should match this interface.
    """

    def __init__(self, loader: HWNASLoader, dataset: str = "cifar10",
                 hw_platform: str = "raspi4"):
        self.loader = loader
        self.dataset = dataset
        self.hw_platform = hw_platform

    def run(self, n_evaluations: int = 500, seed: int = 42) -> dict:
        np.random.seed(seed)
        all_solutions = []

        for i in tqdm(range(n_evaluations), desc="Random Search"):
            idx = self.loader.random_architecture()
            result = self.loader.get_objectives(idx, self.dataset, self.hw_platform)
            all_solutions.append(result)

        pareto = compute_pareto_front(all_solutions)

        return {
            "all_solutions": all_solutions,
            "pareto_front": pareto,
            "n_evaluations": n_evaluations,
            "dataset": self.dataset,
            "hw_platform": self.hw_platform,
        }