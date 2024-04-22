import torch

from ...utils.loader import model_load_state_dict_nostrict
from .tsm import TSM

base_model = "resnet50"


def tsm_resnet50(num_classes, input_frame_length, pretrained="imagenet"):
    assert pretrained in [
        "imagenet",
        "random",
        "imagenet1k_v1",
        "imagenet1k_v2",
        "default",
        "kinetics400",
    ]

    class_counts = num_classes
    segment_count = input_frame_length

    if pretrained == "kinetics400":
        assert input_frame_length == 8

        model = TSM(
            class_counts,
            segment_count,
            "RGB",
            base_model=base_model,
            consensus_type="avg",
            pretrained=None,
        )

        weights = torch.hub.load_state_dict_from_url(
            "https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_256p_1x1x8_50e_kinetics400_rgb/tsm_r50_256p_1x1x8_50e_kinetics400_rgb_20200726-020785e2.pth"
        )

        new_weights = {}
        # Convert mmaction2 keys to torchvision keys
        for k, v in weights["state_dict"].items():
            if k == "cls_head.fc_cls.weight":
                k = "new_fc.weight"
            elif k == "cls_head.fc_cls.bias":
                k = "new_fc.bias"
            else:
                if k.startswith("backbone."):
                    k = k.replace("backbone.", "base_model.")

                k = k.replace(".conv1.conv.", ".conv1.")
                k = k.replace(".conv2.conv.", ".conv2.")
                k = k.replace(".conv3.conv.", ".conv3.")
                k = k.replace(".conv1.bn.", ".bn1.")
                k = k.replace(".conv2.bn.", ".bn2.")
                k = k.replace(".conv3.bn.", ".bn3.")
                k = k.replace(".downsample.conv.", ".downsample.0.")
                k = k.replace(".downsample.bn.", ".downsample.1.")
                k = k.replace("cls_head.fc_cls.weight", ".downsample.1.")
                k = k.replace(".downsample.bn.", ".downsample.1.")
            new_weights[k] = v

        model_load_state_dict_nostrict(model, new_weights, partial=False)

    else:
        model = TSM(
            class_counts,
            segment_count,
            "RGB",
            base_model=base_model,
            consensus_type="avg",
            pretrained=None if pretrained == "random" else pretrained,
        )

    return model
