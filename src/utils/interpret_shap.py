# utils/interpret_shap.py
from src.benchmark.hwnas_loader import HWNASLoader

def map_feature_indices_to_ops(arch_index, feature_indices):
    """
    Given an architecture index and a list of SHAP feature indices,
    returns the corresponding operations in that architecture.
    
    Args:
        arch_index (int): Index of the architecture
        feature_indices (list[int]): List of feature indices from SHAP importance
        
    Returns:
        dict: mapping feature_index -> op_name
    """
    loader = HWNASLoader()
    config = loader.get_net_config(arch_index)
    arch_str = config["arch_str"]
    
    # Parse the arch_str into a flat list of ops
    ops = []
    nodes = arch_str.split('+')
    for node in nodes:
        parts = node.strip('|').split('|')
        for p in parts:
            op = p.split('~')[0]
            ops.append(op)
    
    # Map indices
    mapping = {}
    for idx in feature_indices:
        if idx < len(ops):
            mapping[idx] = ops[idx]
        else:
            mapping[idx] = "unknown"
    return mapping

# Example usage after computing SHAP:
# top_indices = [3, 0, 5, 4, 1, 2, 6]
# arch_index = 10
# mapping = map_feature_indices_to_ops(arch_index, top_indices)
# print(mapping)