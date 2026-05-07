import numpy as np

from src.utils.nasbench201_encoding import (
    vector_to_index,
    index_to_vector
)

from src.benchmark.hwnas_loader import HWNASLoader
from src.benchmark.nasbench201csv_loader import NASBench201CSVLoader  



def test_round_trip():

    print("\n=== Round-trip test ===")

    test_vectors = [
        np.array([0,0,0,0,0,0]),
        np.array([1,2,3,4,0,1]),
        np.array([4,4,4,4,4,4]),
        np.array([0,1,2,3,4,0])
    ]

    for v in test_vectors:

        idx = vector_to_index(v)

        v_back = index_to_vector(idx)

        print("Vector:", v)
        print("Index :", idx)
        print("Back  :", v_back)

        assert np.all(v == v_back), \
            "Round-trip failed!"

    print("✓ Round-trip test passed")


def test_random_vectors(n_tests=20):

    print("\n=== Random vector tests ===")

    for _ in range(n_tests):

        v = np.random.randint(
            0,
            5,
            size=6
        )

        idx = vector_to_index(v)

        v_back = index_to_vector(idx)

        assert np.all(v == v_back), \
            f"Mismatch for {v}"

    print(f"✓ {n_tests} random tests passed")


def test_index_bounds():

    print("\n=== Index bounds test ===")

    max_index = 5**6 - 1

    v = np.array([4,4,4,4,4,4])

    idx = vector_to_index(v)

    print("Expected max:", max_index)
    print("Computed max:", idx)

    assert idx == max_index, \
        "Max index mismatch"

    print("✓ Bounds test passed")


def test_hw_queries():

    print("\n=== HW query test ===")

    acc_loader = NASBench201CSVLoader(
        csv_path="data/nasbench201_accuracies.csv"
    )

    hw_loader = HWNASLoader()

    v = np.array([1,2,3,4,0,1])

    idx = vector_to_index(v)

    print("Vector:", v)
    print("Index :", idx)

    acc = acc_loader.get_accuracy(idx)

    lat = hw_loader.get_latency(idx)

    print("Accuracy:", acc)
    print("Latency :", lat)

    print("✓ HW query successful")


if __name__ == "__main__":

    print("\n==============================")
    print(" NASBench201 Encoding Tests ")
    print("==============================")

    test_round_trip()

    test_random_vectors()

    test_index_bounds()

    test_hw_queries()

    print("\nAll tests completed successfully.")