import ast
import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from ecg_analysis.load_data import prepare_tabular_data, prepare_waves


class PtbXlWrapper(ABC):
    """Store train/test/validation waves and tabular data """

    def __init__(
            self,
            raw_data_folder: str,
            processed_data_folder: str,
            ptbxl_dataset_filename: str,
            scp_statements_filename: str,
            classes_mlb_filename: str,
            superclasses_mlb_filenames: str,
            tabular_filename: str,
            waves_filename: str,
            threshold: int,
            sampling_rate: int=100,
            batch_size: int = 64,
            drop_last: bool = False,
    ) -> None:
        """Separate data into train/test/val datasets"""

        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.threshold = threshold
        self.processed_data_folder = processed_data_folder
        self.waves_filename = waves_filename
        self.drop_last = drop_last

        # Check for prepared data existence, if not, process
        if not (
            os.path.exists(os.path.join(processed_data_folder, tabular_filename))
            and os.path.exists(
                os.path.join(processed_data_folder, classes_mlb_filename)
            )
            and os.path.exists(
                os.path.join(processed_data_folder, superclasses_mlb_filenames)
            )
        ):
            prepare_tabular_data(
                os.path.join(raw_data_folder, ptbxl_dataset_filename),
                os.path.join(raw_data_folder, scp_statements_filename),
                processed_data_folder,
                tabular_filename,
                classes_mlb_filename,
                superclasses_mlb_filenames,
                self.threshold
            )

        if not (
            os.path.exists(os.path.join(
                processed_data_folder, f"{waves_filename}_train.npy")
            )
            and os.path.exists(os.path.join(
                processed_data_folder, f"{waves_filename}_test.npy")
            )
            and os.path.exists(os.path.join(
                processed_data_folder, f"{waves_filename}_validation.npy")
            )
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

        with open(os.path.join(
            processed_data_folder, classes_mlb_filename
        ), "rb") as f:
            self.classes_mlb = pickle.load(f)

        with open(os.path.join(
            processed_data_folder, superclasses_mlb_filenames
        ), "rb") as f:
            self.superclasses_mlb = pickle.load(f)

        tabular = pd.read_csv(os.path.join(processed_data_folder, tabular_filename))
        tabular["diagnose"] = tabular["diagnose"].apply(ast.literal_eval)
        tabular["superclass"] = tabular["superclass"].apply(ast.literal_eval)

        self.labels = self.prepare_labels(tabular)
        self.labels_indices_train = tabular["strat_fold"] < 9
        self.labels_indices_val = tabular["strat_fold"] == 9
        self.labels_indices_test = tabular["strat_fold"] == 10

    @property
    def _waves_train(self) -> np.ndarray:
        return np.load(
            os.path.join(self.processed_data_folder, f"{self.waves_filename}_train.npy")
        )

    @property
    def waves_val(self) -> np.ndarray:
        return np.load(
            os.path.join(
                self.processed_data_folder,
                f"{self.waves_filename}_validation.npy"
            )
        )

    @property
    def _waves_test(self) -> np.ndarray:
        return np.load(
            os.path.join(self.processed_data_folder, f"{self.waves_filename}_test.npy")
        )

    @abstractmethod
    def prepare_labels(self, tabular: pd.DataFrame) -> np.ndarray:
        ...

    @property
    def y_train(self) -> np.ndarray:
        return self.labels[self.labels_indices_train]

    @property
    def y_val(self) -> np.ndarray:
        return self.labels[self.labels_indices_val]

    @property
    def y_test(self) -> np.ndarray:
        return self.labels[self.labels_indices_test]

    def make_train_dataloader(self) -> DataLoader:
        return DataLoader(
            PtbXl(self._waves_train, self.y_train),
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )

    def make_val_dataloader(self) -> DataLoader:
        return DataLoader(
            PtbXl(self.waves_val, self.y_val),
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )

    def make_test_dataloader(self) -> DataLoader:
        return DataLoader(
            PtbXl(self._waves_test, self.y_test),
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )


class PtbXlClasses(PtbXlWrapper):
    def __init__(
            self,
            raw_data_folder: str,
            processed_data_folder: str,
            ptbxl_dataset_filename: str,
            scp_statements_filename: str,
            classes_mlb_filename: str,
            superclasses_mlb_filenames: str,
            tabular_filename: str,
            waves_filename: str,
            threshold: int,
            sampling_rate: int=100,
            batch_size: int = 64,
    ) -> None:
        super().__init__(
            raw_data_folder,
            processed_data_folder,
            ptbxl_dataset_filename,
            scp_statements_filename,
            classes_mlb_filename,
            superclasses_mlb_filenames,
            tabular_filename,
            waves_filename,
            threshold,
            sampling_rate,
            batch_size,
        )

    def prepare_labels(self, tabular: pd.DataFrame) -> np.ndarray:
        return self.classes_mlb.transform(tabular["diagnose"].to_numpy())


class PtbXlClassesSuperclasses(PtbXlWrapper):
    def __init__(
            self,
            raw_data_folder: str,
            processed_data_folder: str,
            ptbxl_dataset_filename: str,
            scp_statements_filename: str,
            classes_mlb_filename: str,
            superclasses_mlb_filenames: str,
            tabular_filename: str,
            waves_filename: str,
            threshold: int,
            sampling_rate: int=100,
            batch_size: int = 64,
    ) -> None:
        super().__init__(
            raw_data_folder,
            processed_data_folder,
            ptbxl_dataset_filename,
            scp_statements_filename,
            classes_mlb_filename,
            superclasses_mlb_filenames,
            tabular_filename,
            waves_filename,
            threshold,
            sampling_rate,
            batch_size,
            drop_last=True
        )

    def prepare_labels(self, tabular: pd.DataFrame) -> np.ndarray:
        return np.hstack((
            self.classes_mlb.transform(tabular["diagnose"].to_numpy()),
            self.superclasses_mlb.transform(tabular["superclass"].to_numpy())
        ))


class PtbXl(Dataset):
    """Implement ptbxl dataset"""
    features: np.ndarray
    labels: np.ndarray
    index: int
    length: int

    def __init__(
            self,
            features: np.ndarray,
            labels: np.ndarray,
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
