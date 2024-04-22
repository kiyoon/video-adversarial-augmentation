import torch.nn as nn
from hydra_zen import store

from . import zen_store
from .epic import tsm_resnet50


@zen_store(
    name="tsm_resnet50-pre=kinetics400",
    num_frames="${input_num_frames}",
    num_classes="${dataset.num_classes}",
    pretrained="kinetics400",
)
@zen_store(
    name="tsm_resnet50-pre=imagenet",
    num_frames="${input_num_frames}",
    num_classes="${dataset.num_classes}",
    pretrained="imagenet",
)
@zen_store(
    name="tsm_resnet50",
    num_frames="${input_num_frames}",
    num_classes="${dataset.num_classes}",
    pretrained="random",
)
def build_tsm_model(
    num_classes: int = 51,
    num_frames: int = 8,
    pretrained: str = "kinetics400",
):
    model: nn.Module = tsm_resnet50(
        num_classes=num_classes, input_frame_length=num_frames, pretrained=pretrained
    )

    return model
