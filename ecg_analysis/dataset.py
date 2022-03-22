import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from ecg_analysis.load_data import prepare_tabular_data, prepare_waves


class PtbXlWrapper:
    """Store train/test/validation waves and tabular data """

    def __init__(
            self,
            raw_data_folder: str,
            processed_data_folder: str,
            ptbxl_dataset_filename: str,
            scp_statements_filename: str,
            classes_mlb_filename: str,
            supercalsses_mlb_filenames: str,
            tabular_filename: str,
            waves_filename: str,
            threshold: int,
            sampling_rate: int=100,
            batch_size: int = 64,
    ) -> None:
        """Separate data into train/test/val datasets"""

        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.threshold = threshold

        # Check for prepared data existence, if not, process
        if not (
            os.path.exists(os.path.join(processed_data_folder, tabular_filename))
            and os.path.exists(os.path.join(processed_data_folder, classes_mlb_filename))
            and os.path.exists(os.path.join(processed_data_folder, supercalsses_mlb_filenames))
        ):
            prepare_tabular_data(
                os.path.join(raw_data_folder, ptbxl_dataset_filename),
                os.path.join(raw_data_folder, scp_statements_filename),
                processed_data_folder,
                tabular_filename,
                classes_mlb_filename,
                supercalsses_mlb_filenames,
                self.threshold
            )

        if not (
            os.path.exists(os.path.join(processed_data_folder, f"{waves_filename}_train.npy"))
            and os.path.exists(os.path.join(processed_data_folder, f"{waves_filename}_test.npy"))
            and os.path.exists(os.path.join(processed_data_folder, f"{waves_filename}_validation.npy"))
        ):
            prepare_waves(
                os.path.join(processed_data_folder, tabular_filename),
                raw_data_folder,
                waves_filename,
                processed_data_folder,
                self.sampling_rate
            )

        self.classes_mlb: MultiLabelBinarizer
        self.superclasses_mlb: MultiLabelBinarizer

        with open(os.path.join(processed_data_folder, classes_mlb_filename), "rb") as f:
            self.classes_mlb = pickle.load(f)

        with open(os.path.join(processed_data_folder, supercalsses_mlb_filenames), "rb") as f:
            self.superclasses_mlb = pickle.load(f)

        self.waves_train = np.load(os.path.join(processed_data_folder, f"{waves_filename}_train.npy"))
        self.waves_val = np.load(os.path.join(processed_data_folder, f"{waves_filename}_validation.npy"))
        self.waves_test = np.load(os.path.join(processed_data_folder, f"{waves_filename}_test.npy"))
        tabular = pd.read_csv(os.path.join(processed_data_folder, tabular_filename))

        # Split labels
        # 1-8 for training
        self.y_train = tabular[tabular.strat_fold < 9]["mlb_diagnose"].to_numpy()

        # 9 for validation
        self.y_val = tabular[tabular.strat_fold == 9]["mlb_diagnose"].to_numpy()

        # 10 for test
        self.y_test = tabular[tabular.strat_fold == 10]["mlb_diagnose"].to_numpy()

    def make_train_dataloader(self) -> DataLoader:
        return DataLoader(
            PtbXl(self.waves_train, self.y_train),
            batch_size=self.batch_size
        )

    def make_val_dataloader(self) -> DataLoader:
        return DataLoader(
            PtbXl(self.waves_val, self.y_val),
            batch_size=self.batch_size
        )

    def make_test_dataloader(self) -> DataLoader:
        return DataLoader(
            PtbXl(self.waves_test, self.y_test),
            batch_size=self.batch_size
        )


class PtbXl(Dataset):
    """Implement ptbxl dataset"""
    features: torch.Tensor
    labels: torch.Tensor
    index: int
    length: int

    def __init__(
            self,
            features: torch.Tensor,
            labels: torch.Tensor,
    ) -> None:
        """Create dataset from user features and labels"""

        feat_len, labels_len = len(features), len(labels)

        if feat_len != labels_len:
            raise ValueError(
                f"Length of features and labels must be the same,"
                f"but it's {feat_len}, {labels_len} respectively"
            )

        self.length = feat_len
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        features = self.features[index]
        label = self.labels[index]

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )
