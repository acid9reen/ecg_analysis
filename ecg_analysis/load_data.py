import ast
import os
import pickle
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


def probs_to_tuple(probs: dict[str, int], threshold: int = 20) -> Optional[tuple[str]]:
    """
    Convert dict of diagnoses and their probabilities to
    tuple of diagnoses with probabilities >= given threshold.
    If result include diagnose with "NORM" or empty, return NA for later drop
    """

    result = tuple([key for key, value in probs.items() if value >= threshold])

    is_diagnose_with_norm = ("NORM" in result) and (len(result) > 1)

    if not result or is_diagnose_with_norm:
        return None

    return result


def aggregate_diagnostic(
        diagnoses: Optional[tuple[str]], mapping: dict[str, str]
) -> Optional[tuple[Optional[str]]]:
    """
    Return values of encountered keys from the given mapping.
    """

    if not diagnoses:
        return None

    superclasses = tuple({
        superclass
        if pd.notna(superclass := mapping.get(diagnose))
        else None
        for diagnose in diagnoses
    })

    if None in superclasses:
        return None

    return superclasses


def create_fit_classes_and_superclasses_mlbs(
        ptbxl: pd.DataFrame,
        scp_statements: pd.DataFrame,
) -> tuple[MultiLabelBinarizer, MultiLabelBinarizer]:
    """
    Fit mlbs, return them.
    """

    classes = tuple(scp_statements[scp_statements.diagnostic_class.notna()].index)
    superclasses = list(scp_statements.diagnostic_class.unique())
    superclasses = tuple(filter(
        lambda diagnose: isinstance(diagnose, str),
        superclasses
    ))

    classes_mlb = MultiLabelBinarizer()
    superclasses_mlb = MultiLabelBinarizer()
    classes_mlb.fit([classes])
    superclasses_mlb.fit([superclasses])

    ptbxl["mlb_diagnose"] = [
        np.array(diagnose)
        for diagnose in classes_mlb.transform(ptbxl.diagnose.to_numpy())
    ]
    ptbxl["mlb_superclass"] = [
        np.array(superclass)
        for superclass in superclasses_mlb.transform(ptbxl.superclass.to_numpy())
    ]

    return classes_mlb, superclasses_mlb


def prepare_tabular_data(
        database_path: str,
        scp_statements_path: str,
        output_folder_path: str,
        data_filename: str,
        classes_mlb_filename: str,
        superclasses_mlb_filename: str,
        threshold: int,
) -> None:
    """
    Preprocess tabular data, fit mlbs for classes and superclasses.
    Finally save processed tabular data and mlbs
    """

    # Read datasets
    ptbxl = pd.read_csv(database_path, index_col="ecg_id")
    scp_statements = pd.read_csv(scp_statements_path, index_col=0)

    # Convert dict string to dict object
    ptbxl.scp_codes = ptbxl.scp_codes.apply(ast.literal_eval)

    # Convert probs to tuple with given threshold
    probs_to_tuple_treshold = partial(probs_to_tuple, threshold=threshold)
    ptbxl["diagnose"] = ptbxl.scp_codes.apply(probs_to_tuple_treshold)

    # Map all classes to their superclasses
    class_to_superclass_mapping = dict(zip(
        scp_statements.index, scp_statements.diagnostic_class
    ))

    aggregate_diagnostic_class_to_superclass = partial(
        aggregate_diagnostic,
        mapping=class_to_superclass_mapping,
    )

    ptbxl["superclass"] = ptbxl.diagnose.apply(aggregate_diagnostic_class_to_superclass)

    ptbxl = ptbxl.dropna(axis="index", subset=["superclass", "diagnose"])

    # Create and fit mlbs
    classes_mlb, superclasses_mlb = create_fit_classes_and_superclasses_mlbs(
        ptbxl, scp_statements
    )

    # Save processed tabular data and mlbs
    os.makedirs(output_folder_path, exist_ok=True)
    ptbxl.to_csv(os.path.join(output_folder_path, data_filename))

    with open(os.path.join(output_folder_path, classes_mlb_filename), 'wb') as out:
        pickle.dump(classes_mlb, out)

    with open(os.path.join(output_folder_path, superclasses_mlb_filename), 'wb') as out:
        pickle.dump(superclasses_mlb, out)


def prepare_waves(
        path_to_processed_tabular_data: str,
        waves_folder_path: str,
        processed_data_name: str,
        out_folder: str,
        sampling_rate: int,
) -> None:
    """
    Collect waves from all over the dataset, split and save after all
    """

    if sampling_rate not in (100, 500):
        raise ValueError(
            f"Sampling rate must be 100 either 500, not {sampling_rate}!"
        )

    df_filename_column = (
        "filename_lr" if sampling_rate == 100
        else "filename_hr"
    )

    tabular = pd.read_csv(path_to_processed_tabular_data, index_col="ecg_id")
    train_tabular = tabular[tabular.strat_fold < 9]
    validation_tabular = tabular[tabular.strat_fold == 9]
    test_tabular = tabular[tabular.strat_fold == 10]

    tabular = {
        "train": train_tabular, "validation": validation_tabular, "test": test_tabular
    }

    for split_name, tabular_data in tabular.items():
        files = tabular_data[df_filename_column]
        data = [
            wfdb.rdsamp(os.path.join(waves_folder_path, file))
            for file in tqdm(files, f"Collecting {split_name} data")
        ]

        data = np.array([np.transpose(signal) for signal, __ in data])
        np.save(os.path.join(out_folder, f"{processed_data_name}_{split_name}"), data)
