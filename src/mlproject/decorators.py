import functools
from typing import Any, Callable

import torch


def collect_metrics(func: Callable) -> Callable:
    def collect_metrics(
        step_idx: int,
        metrics_dict: dict,
        phase_name: str,
        experiment_tracker: Any,
    ) -> None:
        for metric_key, computed_value in metrics_dict.items():
            if computed_value is not None:
                value = (
                    computed_value.detach()
                    if isinstance(computed_value, torch.Tensor)
                    else computed_value
                )
                experiment_tracker.log(
                    {f"{phase_name}/{metric_key}": value},
                    step=step_idx,
                )

                # print(f"{phase_name}/{metric_key} {value} {step_idx}")

    @functools.wraps(func)
    def wrapper_collect_metrics(*args, **kwargs):
        outputs = func(*args, **kwargs)
        experiment_tracker = args[0].experiment_tracker
        metrics_dict = outputs.metrics
        phase_name = outputs.phase_name
        step_idx = outputs.step_idx
        collect_metrics(
            step_idx=step_idx,
            metrics_dict=metrics_dict,
            phase_name=phase_name,
            experiment_tracker=experiment_tracker,
        )
        return outputs

    return wrapper_collect_metrics
