import numpy as np

from src.utils.nasbench201_encoding import (
    vector_to_index
)

class HWNASProblem:

    def __init__(self,
                 acc_loader,
                 hw_loader,
                 dataset="cifar10",
                 platform="raspi4"):

        self.acc_loader = acc_loader
        self.hw_loader = hw_loader

        self.dataset = dataset
        self.platform = platform

        # NASBench201 structure
        self.n_var = 6

        self.n_obj = 2

        self.num_choices = np.array([5]*6)

        self.cache = {}


    def evaluate(self, x):

        # vector → index
        arch_index = vector_to_index(x)

        if arch_index in self.cache:
            return self.cache[arch_index]

        acc = self.acc_loader.get_accuracy(
            arch_index,
            dataset=self.dataset
        )

        latency = self.hw_loader.get_latency(
            arch_index,
            hw_platform=self.platform,
            dataset=self.dataset
        )

        # NSGA-II minimizes
        f1 = -acc
        f2 = latency

        result = np.array([f1, f2])

        self.cache[arch_index] = result

        return result