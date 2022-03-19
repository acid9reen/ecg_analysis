import ast
import os
import pickle
from functools import partial

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def probs_to_tuple(probs: dict[str, int], threshold: int = 20) -> tuple[str]:
    """
    Convert dict of diagnoses and their probabilities to
    tuple of diagnoses with probabilities >= given threshold.
    Also if probabilities of diagnoses < probability of NORM, return ("NORM",)
    else tuple of diagnoses with probabilities >= probability of NORM.
    """

    norm_prob = probs.get("NORM", 0)

    result = [
        key for key, value in probs.items() if (
            value >= threshold and value >= norm_prob  and key != "NORM"
        )
    ]

    return res if (res := tuple(result)) else ("NORM",)


def aggregate_diagnostic(diagnoses: tuple[str], mapping: dict[str, str]):
    """
    Return values of encountered keys from the given mapping.
    """

    superclasses = {
        superclass
        if pd.notna(superclass := mapping.get(diagnose))
        else "NONE"
        for diagnose in diagnoses
    }

    return tuple(superclasses)


def create_fit_classes_and_superclasses_mlbs(
        ptbxl: pd.DataFrame,
        scp_statements: pd.DataFrame,
) -> tuple[MultiLabelBinarizer, MultiLabelBinarizer]:
    """
    Fit mlbs, return them.
    """

    classes = tuple(scp_statements.index)
    superclasses = list(scp_statements.diagnostic_class.unique()) + ["NONE"]
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
