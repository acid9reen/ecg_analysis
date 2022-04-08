from typing import Literal, Optional

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             multilabel_confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from ecg_analysis.metrics import EpochMetric
from ecg_analysis.tensorboard import TensorboardExperiment
from ecg_analysis.tracking import Stage


class Runner:
    def __init__(
            self,
            loader: DataLoader,
            model: torch.nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            threshold: float = 0.5,
            device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        self.run_count = 0
        self.loader = loader
        self.accuracy_metric = EpochMetric()
        self.loss = EpochMetric()
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.thresholder = torch.nn.Threshold(threshold, 0)

        # Objective (loss) function
        self.compute_loss = torch.nn.BCELoss(reduction="mean")
        self.y_true_batches = []
        self.y_pred_batches = []

        # Assume Stage based on presence of optimizer
        self.stage = Stage.VALIDATION if optimizer is None else Stage.TRAIN

    @property
    def avg_accuracy(self):
        return self.accuracy_metric.average

    @property
    def avg_loss(self):
        return self.loss.average

    def run(self, desc: str, experiment: TensorboardExperiment):
        self.model.train(self.stage is Stage.TRAIN)

        for x, y in tqdm(self.loader, desc=desc, ncols=80):
            x, y = x.to(self.device), y.to(self.device)
            loss, batch_accuracy = self._run_single(x, y)

            if self.optimizer:
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def _run_single(self, x: torch.Tensor, y: torch.Tensor):
        self.run_count += 1
        batch_size: int = x.shape[0]
        prediction = self.model(x)
        loss = self.compute_loss(prediction, y)

        # Compute Batch Test Metrics
        y_np = y.cpu().detach().numpy()
        prediction = self.thresholder(prediction).ceil()
        y_prediction_np = prediction.cpu().detach().numpy()

        batch_accuracy = accuracy_score(y_np, y_prediction_np)
        self.accuracy_metric.update(batch_accuracy, batch_size)
        loss_copy = torch.clone(loss)
        self.loss.update(loss_copy.cpu().detach().numpy(), batch_size)

        self.y_true_batches += [y_np]
        self.y_pred_batches += [y_prediction_np]
        return loss, batch_accuracy

    def reset(self):
        self.accuracy_metric = EpochMetric()
        self.loss = EpochMetric()
        self.y_true_batches = []
        self.y_pred_batches = []


def run_epoch(
        train_runner: Runner,
        val_runner: Runner,
        experiment: TensorboardExperiment,
        epoch_id: int,
) -> None:
    # Training Loop
    experiment.stage = Stage.TRAIN
    train_runner.run("Train Batches", experiment)

    # Log Training Epoch Metrics
    experiment.add_epoch_metric(
        "Accuracy",
        train_runner.avg_accuracy,
        epoch_id
    )

    experiment.add_epoch_metric(
        "Loss",
        train_runner.avg_loss,
        epoch_id
    )

    # Validation Loop
    experiment.stage = Stage.VALIDATION
    val_runner.run("Validation Batches", experiment)

    # Log Validation Epoch Metrics
    experiment.add_epoch_metric(
        "Loss",
        val_runner.avg_loss,
        epoch_id
    )

    experiment.add_epoch_metric("Accuracy", val_runner.avg_accuracy, epoch_id)
    precision, recall, f1_score, __ = precision_recall_fscore_support(
        np.concatenate(val_runner.y_true_batches),
        np.concatenate(val_runner.y_pred_batches),
        average="samples",
        zero_division=0,
    )
    experiment.add_epoch_metric("Precision", precision, epoch_id)
    experiment.add_epoch_metric("Recall", recall, epoch_id)
    experiment.add_epoch_metric("f1_score", f1_score, epoch_id)


def run_test(
        test_runner: Runner,
        experiment: TensorboardExperiment,
        classes: list[str],
) -> None:
    epoch_id = 0

    experiment.stage = Stage.TEST
    test_runner.run("Test Batches", experiment)
    fig = cf_img(
        test_runner.y_true_batches,
        test_runner.y_pred_batches,
        classes
    )

    experiment.add_epoch_img(fig)
    experiment.add_epoch_metric("Accuracy", test_runner.avg_accuracy, epoch_id)
    precision, recall, f1_score, __ = precision_recall_fscore_support(
        np.concatenate(test_runner.y_true_batches),
        np.concatenate(test_runner.y_pred_batches),
        average="samples",
        zero_division=0,
    )
    experiment.add_epoch_metric("Precision", precision, epoch_id)
    experiment.add_epoch_metric("Recall", recall, epoch_id)
    experiment.add_epoch_metric("f1_score", f1_score, epoch_id)


def cf_img(
        y_true_batches: list,
        y_pred_batches: list,
        classes: list[str],
) -> matplotlib.figure.Figure:
    confusion_matrices = multilabel_confusion_matrix(
        np.concatenate(y_true_batches),
        np.concatenate(y_pred_batches),
    )

    biggest_classes_conf_matrices = sorted(
        zip(classes, confusion_matrices),
        key=lambda x: x[1][1][0],
        reverse=True
    )

    biggest_classes_conf_matrices = (
        biggest_classes_conf_matrices[:2]
        + biggest_classes_conf_matrices[-2:]
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 7), dpi=87)
    axes = axes.ravel()

    for axe, (title, cf) in zip(axes, biggest_classes_conf_matrices):
        disp = ConfusionMatrixDisplay(cf)
        disp.plot(ax=axe)
        disp.im_.colorbar.remove()
        disp.ax_.set_title(title)

    return fig
