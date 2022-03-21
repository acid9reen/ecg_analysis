import torch

from ecg_analysis.dataset import PtbXlWrapper

# Hyperparameters
EPOCH_COUNT = 20
LR = 8e-4
BATCH_SIZE = 128
LOG_PATH = "./runs"

# Data configuration
DATA_DIR = "data"

# Hardware configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{DEVICE=}")


def main():
    dataset = PtbXlWrapper(
        r"data/raw",
        r"data/processed",
        "ptbxl_dataset.csv",
        "scp_statements.csv",
        "classes_mlb.pkl",
        "superclasses_mlb.pkl",
        "tabular.csv",
        "waves",
        threshold=100,
        sampling_rate=100,
        batch_size=64,
    )


if __name__ == "__main__":
    main()
