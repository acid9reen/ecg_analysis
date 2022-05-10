import os

import torch
from torchinfo import summary

from ecg_analysis.dataset import PtbXlClasses, PtbXlClassesSuperclasses
from ecg_analysis.models import (ResidualConvNet, ResidualConvNetMixed,
                                 SimpleConv)
from ecg_analysis.runner import Runner, run_epoch, run_test
from ecg_analysis.tensorboard import TensorboardExperiment

# Hyperparameters
EPOCH_COUNT = 150
LR = 4e-4
BATCH_SIZE = 128
LOG_PATH = "./runs"
CLASS_SUPERCLASS_PENALTY_RATIO = 0.5  # how many times the loss is less for classes than for superclasses

# Data configuration
DATA_DIR = "data"

# Hardware configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{DEVICE=}")


def main():
    # Model and optimizer
    model = ResidualConvNetMixed(
        channels_progression=[12, 64, 128, 256, 512],
        downsamples=[2, 2, 2, 2],
        kernels_sizes=[3, 3, 3, 3],
        dropout_probs=[0.5 for __ in range(4)],
        linear_layers_sizes_1=[64],
        linear_layers_sizes_2=[128],
        num_superclasses=5,
        num_classes=44,
        outer_dropout_prob=0.5,
    ).to(DEVICE)

    # model = ResidualConvNet(
    #     channels_progression=[12, 64, 128, 256, 512],
    #     downsamples=[2, 2, 2, 2],
    #     kernels_sizes=[3, 3, 3, 3],
    #     dropout_probs=[0.5 for __ in range(4)],
    #     linear_layers_sizes=[64],
    #     num_classes=44,
    #     outer_dropout_prob=0.5,
    # ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    weight = torch.ones([BATCH_SIZE, 49], dtype=torch.float32)
    weight[:, :-5] *= CLASS_SUPERCLASS_PENALTY_RATIO
    weight = weight.to(DEVICE)

    # Uncomment to get model summary
    # summary(model, input_size=(BATCH_SIZE, 12, 1000))
    # return

    # Data reading and dataloaders
    dataset = PtbXlClassesSuperclasses(
        r"data/raw",
        r"data/processed",
        "ptbxl_database.csv",
        "scp_statements.csv",
        "classes_mlb.pkl",
        "superclasses_mlb.pkl",
        "tabular.csv",
        "waves",
        threshold=100,
        sampling_rate=100,
        batch_size=BATCH_SIZE,
    )

    # Create the data loaders
    train_dl = dataset.make_train_dataloader()
    test_dl = dataset.make_test_dataloader()
    val_dl = dataset.make_val_dataloader()

    # Create the runners
    test_runner = Runner(test_dl, model, device=DEVICE)
    train_runner = Runner(train_dl, model, optimizer, device=DEVICE, weight=weight)
    val_runner = Runner(val_dl, model, device=DEVICE)

    # Setup the experiment tracker
    tracker = TensorboardExperiment(LOG_PATH)

    # Run the epochs
    for epoch_id in range(EPOCH_COUNT):
        run_epoch(
            train_runner,
            val_runner,
            tracker,
            epoch_id,
        )

        # Compute Average Epoch Metrics
        print(
            f"[Epoch: {epoch_id + 1}/{EPOCH_COUNT}]",
            f"Test Accuracy: {test_runner.avg_accuracy: 0.4f}",
            f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
            sep='/n',
            end='/n/n',
        )

        # Reset the runners
        train_runner.reset()
        test_runner.reset()

        # Flush the tracker after every epoch for live updates
        tracker.flush()

    classes = (
        list(dataset.classes_mlb.classes_)
        + list(dataset.superclasses_mlb.classes_)
    )

    run_test(test_runner, tracker, classes)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(tracker.log_dir, "model.pt"))


if __name__ == "__main__":
    main()
