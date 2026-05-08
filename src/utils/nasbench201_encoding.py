import numpy as np


OPS = [
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3"
]


NUM_EDGES = 6
NUM_OPS = len(OPS)


def vector_to_index(x):
    """
    Convert 6-op vector → NASBench201 index.
    
    Base-5 encoding.
    """

    index = 0

    for i, op in enumerate(x):
        index += int(op) * (NUM_OPS ** i)

    return index


def index_to_vector(index):
    """
    Convert index → 6-op vector.
    """

    x = []

    for _ in range(NUM_EDGES):
        x.append(index % NUM_OPS)
        index //= NUM_OPS

    return np.array(x)