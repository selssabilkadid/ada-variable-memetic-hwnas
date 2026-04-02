import pandas as pd

class NASBench201CSVLoader:
    """
    NAS-Bench-201 loader using a CSV file for fast access to accuracies.
    CSV path: data/nasbench201_accuracies.csv
    Columns expected: arch_index, acc_cifar10, acc_cifar100, acc_ImageNet16-120
    """
    
    # Mapping dataset names to CSV columns
    DATASET_KEYS = {
        "cifar10": "acc_cifar10",
        "cifar100": "acc_cifar100",
        "ImageNet16-120": "acc_ImageNet16-120",
    }
    
    def __init__(self, csv_path: str = "data/nasbench201_accuracies.csv"):
        self.df = pd.read_csv(csv_path)
        if "arch_index" not in self.df.columns:
            raise ValueError("CSV must contain 'arch_index' column.")
        self.df.set_index("arch_index", inplace=True)
        print(f"[NASBench201CSVLoader] Loaded {len(self.df)} architectures from CSV.")
        
    def get_accuracy(self, arch_index: int, dataset: str = "cifar10") -> float:
        """
        Returns test accuracy (%) for a given architecture.
        """
        col_name = self.DATASET_KEYS.get(dataset)
        if col_name is None:
            raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(self.DATASET_KEYS.keys())}")
        try:
            return float(self.df.loc[arch_index, col_name])
        except KeyError:
            raise KeyError(f"Architecture index {arch_index} not found in CSV.")
    
    def inspect(self, arch_index: int):
        """
        Prints all available accuracy info for a given architecture.
        """
        try:
            row = self.df.loc[arch_index]
            print(f"[Inspect] arch_index={arch_index}")
            for col in self.df.columns:
                print(f"  {col}: {row[col]}")
        except KeyError:
            print(f"Architecture index {arch_index} not found in CSV.")
    
    def get_arch_ops(self, arch_index: int):
        """
        Placeholder for architecture string parsing.
        Since CSV doesn't store arch_str, return None or raise NotImplementedError.
        """
        raise NotImplementedError("get_arch_ops is not available with CSV loader. "
                                  "Use NAS-Bench-201 .pth API for architecture strings.")