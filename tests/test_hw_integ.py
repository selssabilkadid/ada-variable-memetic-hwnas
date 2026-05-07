import numpy as np
from src.benchmark.hwnas_problem_def import HWNASProblem


class DummyAcc:
    def get_accuracy(self, arch_index, dataset):
        return 0.8


class DummyHW:
    def get_latency(self, arch_index, hw_platform, dataset):
        return 10.0


class DummyEncoder:
    def decode(self, x):
        return x

    def random_encoding(self):
        return [0, 1, 2, 3, 4, 0]


def test_hw_nas_evaluate():
    problem = HWNASProblem(
        acc_loader=DummyAcc(),
        hw_loader=DummyHW(),
        dataset="cifar10",
        platform="raspi4"
    )

    x = np.array([0, 1, 2, 3, 4, 0])
    f = problem.evaluate(x)

    assert len(f) == 2
    assert f[0] == -0.8
    assert f[1] == 10.0


def test_hw_nas_cache():
    problem = HWNASProblem(
        acc_loader=DummyAcc(),
        hw_loader=DummyHW(),
        dataset="cifar10",
        platform="raspi4"
    )

    x = np.array([0, 1, 2, 3, 4, 0])

    f1 = problem.evaluate(x)
    f2 = problem.evaluate(x)

    assert np.array_equal(f1, f2)
