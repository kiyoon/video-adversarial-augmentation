from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor, nn
from torch.utils.data import DataLoader
from wandb import wandb_sdk

from .decorators import collect_metrics
from .utils import get_logger

logger = get_logger(__name__)


def get_dict_shapes(x):
    return (
        {
            key: value.shape if isinstance(value, torch.Tensor) else len(value)
            for key, value in x.items()
        }
        if isinstance(x, dict)
        else get_dict_shapes(x.__dict__)
    )


@dataclass
class EvaluatorOutput:
    step_idx: int
    metrics: dict
    phase_name: str


class Evaluator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def validation_step(
        self,
        model: nn.Module,
        batch: dict[str, Tensor] | list[Tensor],
        batch_idx: int,
        step_idx: int,
        epoch_idx: int,
        accelerator: Accelerator,
    ) -> EvaluatorOutput:
        pass

    @abstractmethod
    def start_validation(
        self,
        model: nn.Module,
        epoch_idx: int,
        step_idx: int,
        val_dataloaders: list[DataLoader] | None = None,
    ) -> EvaluatorOutput:
        pass

    @abstractmethod
    def end_validation(
        self,
        model: nn.Module,
        epoch_idx: int,
        step_idx: int,
        val_dataloaders: list[DataLoader] | None = None,
    ) -> EvaluatorOutput:
        pass

    @abstractmethod
    def testing_step(
        self,
        model: nn.Module,
        batch: dict[str, Tensor] | list[Tensor],
        batch_idx: int,
        step_idx: int,
        epoch_idx: int,
        accelerator: Accelerator,
    ) -> EvaluatorOutput:
        pass

    @abstractmethod
    def start_testing(
        self,
        model: nn.Module,
        epoch_idx: int,
        step_idx: int,
        test_dataloaders: list[DataLoader] | None = None,
    ) -> EvaluatorOutput:
        pass

    @abstractmethod
    def end_testing(
        self,
        model: nn.Module,
        epoch_idx: int,
        step_idx: int,
        test_dataloaders: list[DataLoader] | None = None,
    ) -> EvaluatorOutput:
        pass


class ClassificationEvaluator(Evaluator):
    def __init__(self, experiment_tracker: wandb_sdk.wandb_run.Run | None = None):
        super().__init__()
        self.epoch_metrics = {}
        self.experiment_tracker = experiment_tracker

    def validation_step(
        self,
        model: nn.Module,
        batch: dict[str, Tensor] | list[Tensor],
        batch_idx: int,
        step_idx: int,
        epoch_idx: int,
        accelerator: Accelerator,
    ):
        logits = model(batch["pixel_values"])
        accuracy = (logits.argmax(dim=-1) == batch["labels"]).float().mean()
        loss = F.cross_entropy(logits, batch["labels"]).detach()
        metrics = {"accuracy": accuracy, "loss": loss}

        for key, value in metrics.items():
            self.epoch_metrics.setdefault(key, []).append(value.detach().cpu())

        return EvaluatorOutput(
            step_idx=step_idx,
            phase_name="validation",
            metrics=metrics,
        )

    def testing_step(
        self,
        model: nn.Module,
        batch: dict[str, Tensor] | list[Tensor],
        batch_idx: int,
        step_idx: int,
        epoch_idx: int,
        accelerator: Accelerator,
    ):
        logits = model(batch["pixel_values"])
        accuracy = (logits.argmax(dim=-1) == batch["labels"]).float().mean()
        loss = F.cross_entropy(logits, batch["labels"]).detach()

        metrics = {"accuracy": accuracy, "loss": loss}

        for key, value in metrics.items():
            self.epoch_metrics.setdefault(key, []).append(value.detach().cpu())

        return EvaluatorOutput(
            step_idx=step_idx,
            phase_name="test",
            metrics={"accuracy": accuracy, "loss": loss},
        )

    @collect_metrics
    def start_validation(
        self,
        model: nn.Module,
        epoch_idx: int,
        step_idx: int,
        val_dataloaders: list[DataLoader] | None = None,
    ):
        model.eval()
        self.epoch_metrics = {}
        return EvaluatorOutput(
            step_idx=step_idx,
            phase_name="validation",
            metrics=self.epoch_metrics,
        )

    @collect_metrics
    def start_testing(
        self,
        model: nn.Module,
        epoch_idx: int,
        step_idx: int,
        test_dataloaders: list[DataLoader] | None = None,
    ):
        model.eval()
        self.epoch_metrics = {}
        return EvaluatorOutput(
            step_idx=step_idx, phase_name="testing", metrics=self.epoch_metrics
        )

    @collect_metrics
    def end_validation(
        self,
        model: nn.Module,
        epoch_idx: int,
        step_idx: int,
        val_dataloaders: list[DataLoader] | None = None,
    ):
        epoch_metrics = {}
        for key, value in self.epoch_metrics.items():
            epoch_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            epoch_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        return EvaluatorOutput(
            step_idx=step_idx, phase_name="validation", metrics=epoch_metrics
        )

    @collect_metrics
    def end_testing(
        self,
        model: nn.Module,
        epoch_idx: int,
        step_idx: int,
        test_dataloaders: list[DataLoader] | None = None,
    ):
        epoch_metrics = {}
        for key, value in self.epoch_metrics.items():
            epoch_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            epoch_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        return EvaluatorOutput(
            step_idx=step_idx, phase_name="testing", metrics=epoch_metrics
        )
