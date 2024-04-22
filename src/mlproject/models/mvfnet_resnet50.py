import torch.nn as nn

from . import zen_store
from .epic.mvf import MVFNet


@zen_store(
    name="mvfnet_resnet50-pre=imagenet",
    num_frames="${input_num_frames}",
    num_classes="${dataset.num_classes}",
    pretrained="imagenet",
)
def build_mvfnet_model(
    num_classes: int = 51,
    num_frames: int = 8,
    pretrained: str = "imagenet",
):
    model: nn.Module = MVFNet(
        num_class=num_classes,
        num_segments=num_frames,
        modality="RGB",
        base_model="resnet50",
        pretrained=pretrained,
    )

    return model
