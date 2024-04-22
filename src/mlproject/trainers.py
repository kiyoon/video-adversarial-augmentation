from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor, nn
from torch.utils.data import DataLoader
from wandb import wandb_sdk

from mlproject.callbacks import Interval

from .decorators import collect_metrics
from .utils import get_logger

logger = get_logger(__name__)


def get_dict_shapes(x):
    if not isinstance(x, dict):
        return get_dict_shapes(x.__dict__)
    return {
        key: value.shape if isinstance(value, torch.Tensor) else len(value)
        for key, value in x.items()
    }


@dataclass
class TrainerOutput:
    opt_loss: torch.Tensor
    step_idx: int
    metrics: dict[str, Any]
    phase_name: str


class Trainer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def training_step(
        self,
        model: nn.Module,
        batch: dict[str, Tensor] | list[Tensor],
        batch_idx: int,
        step_idx: int,
        epoch_idx: int,
        accelerator: Accelerator,
    ) -> TrainerOutput:
        pass

    @abstractmethod
    def start_training(
        self,
        model: nn.Module,
        epoch_idx: int,
        step_idx: int,
        train_dataloader: DataLoader | None = None,
    ) -> TrainerOutput:
        pass

    @abstractmethod
    def end_training(
        self,
        model: nn.Module,
        epoch_idx: int,
        step_idx: int,
        train_dataloader: DataLoader | None = None,
    ) -> TrainerOutput:
        pass


class ClassificationTrainer(Trainer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        scheduler_interval: str = Interval.STEP,
        experiment_tracker: wandb_sdk.wandb_run.Run | None = None,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.experiment_tracker = experiment_tracker
        self.epoch_metrics = {}

        if self.scheduler is not None:
            assert scheduler_interval in {"step", "epoch"}
            self.scheduler_interval = scheduler_interval

    def get_optimizer(self):
        return self.optimizer

    @collect_metrics
    def training_step(
        self,
        model: nn.Module,
        batch: dict[str, Tensor],
        batch_idx: int,
        step_idx: int,
        epoch_idx: int,
        accelerator: Accelerator,
    ) -> TrainerOutput:
        self.optimizer.zero_grad()
        logits = model(batch["pixel_values"])
        accuracy = (logits.argmax(dim=-1) == batch["labels"]).float().mean()
        opt_loss = F.cross_entropy(logits, batch["labels"])
        loss = opt_loss.detach()
        accelerator.backward(loss=opt_loss)
        self.optimizer.step()

        if self.scheduler is not None:
            if self.scheduler_interval == "step":
                self.scheduler.step(epoch=step_idx)
            elif self.scheduler_interval == "epoch" and batch_idx == 0:
                self.scheduler.step(epoch=epoch_idx)
        metrics = {"accuracy": accuracy, "loss": loss}
        for key, value in metrics.items():
            self.epoch_metrics.setdefault(key, []).append(value.detach().cpu())

        return TrainerOutput(
            phase_name="training",
            opt_loss=opt_loss,
            step_idx=step_idx,
            metrics={
                "accuracy": accuracy,
                "loss": loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            },
        )

    @collect_metrics
    def start_training(
        self,
        model: nn.Module,
        epoch_idx: int,
        step_idx: int,
        train_dataloader: DataLoader | None = None,
    ):
        model.train()
        self.epoch_metrics = {}
        return TrainerOutput(
            opt_loss=None, step_idx=step_idx, metrics={}, phase_name="training"
        )

    @collect_metrics
    def end_training(
        self,
        model: nn.Module,
        epoch_idx: int,
        step_idx: int,
        train_dataloader: DataLoader | None = None,
    ):
        epoch_metrics = {}
        for key, value in self.epoch_metrics.items():
            epoch_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            epoch_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        return TrainerOutput(
            opt_loss=None,
            step_idx=step_idx,
            metrics=epoch_metrics,
            phase_name="training",
        )
