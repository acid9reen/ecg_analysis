import os
from datetime import datetime as dt

import matplotlib
from torch.utils.tensorboard.writer import SummaryWriter

from ecg_analysis.tracking import Stage


def create_experiment_log_dir(root: str) -> str:
    """
    Create folder under provided root folder with name formed from timestamp.
    Return created folder path
    """

    dirname = str(int(dt.timestamp(dt.now())))
    dirpath = os.path.join(root, dirname)
    os.makedirs(dirpath)

    return dirpath


class TensorboardExperiment:
    def __init__(self, log_path: str):
        self.log_dir = create_experiment_log_dir(root=log_path)
        self._stage = Stage.TRAIN
        self._writer = SummaryWriter(log_dir=self.log_dir)

    @property
    def stage(self) -> Stage:
        return self._stage

    @stage.setter
    def stage(self, stage: Stage) -> None:
        self._stage = stage

    def flush(self) -> None:
        self._writer.flush()

    def add_batch_metric(self, name: str, value: float, step: int) -> None:
        tag = f"{self.stage.name}/batch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_metric(self, name: str, value: float, step: int) -> None:
        tag = f"{self.stage.name}/epoch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_img(
            self,
            fig: matplotlib.figure.Figure
    ) -> None:
        tag = f"{self.stage.name}/epoch/confusion_matrix"
        self._writer.add_figure(tag, fig, 0)
