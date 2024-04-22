import rich
import rich.pretty
from dotenv import load_dotenv
from rich.traceback import install

load_dotenv()
install()  # beautiful and clean tracebacks for debugging

import ast
import io
import os
import pickle
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path

import git
import kornia as K
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sn
import torch
import torch.utils.data
import wandb
import wandb.plot
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs
from hydra_zen import instantiate, store
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from timm.scheduler import CosineLRScheduler
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from mlproject.models import store_configs as _
from mlproject.models.uniformer import uniformer_small
from mlproject.models.video_swin import video_swin_tiny_imagenet
from mlproject.torch_datasets import store_configs as _
from mlproject.utils import set_seed

run_start_time = time.time()

matplotlib.use("pdf")
t = torch.tensor
p = nn.Parameter

accelerator = Accelerator(
    log_with="wandb",
    kwargs_handlers=[
        # broadcast_buffers=False is needed for adversarial training
        # https://github.com/pytorch/pytorch/issues/22095
        # Alternatively, you can use SyncBatchNorm
        # https://stackoverflow.com/questions/66165366/trying-to-use-distributed-data-parallel-on-gans-but-getting-runtime-error-about
        DistributedDataParallelKwargs(
            find_unused_parameters=True,
            broadcast_buffers=False,
        ),
        InitProcessGroupKwargs(timeout=timedelta(hours=3)),
    ],
)


@accelerator.on_local_main_process
def print(*args, **kwargs):
    rich.print(*args, **kwargs)


@accelerator.on_local_main_process
def pprint(*args, **kwargs):
    rich.pretty.pprint(*args, **kwargs)


def plot_confusion_matrix(
    df_confusion_matrix, vmin=0, vmax=1, cmap="YlGnBu", center=None
):
    fig = plt.figure(figsize=(65, 50))
    ax = fig.add_subplot(111)
    # x label on top
    ax.xaxis.tick_top()

    sn.set(font_scale=10)  # for label size
    sn_plot = sn.heatmap(
        df_confusion_matrix,
        annot=False,
        annot_kws={"size": 20},
        cmap=cmap,
        square=True,
        xticklabels=1,
        yticklabels=1,
        vmin=vmin,
        vmax=vmax,
        center=center,
    )  # font size
    plt.xlabel("Predicted", fontsize=50)
    plt.ylabel("Target", fontsize=50)

    # This sets the yticks "upright" with 0, as opposed to sideways with 90.
    plt.yticks(fontsize=20, rotation=0)
    plt.xticks(fontsize=20, rotation=90)

    # here set the colorbar labelsize by 50
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=50)
    fig.set_tight_layout(True)

    return fig


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    return img


@dataclass
class BaseConfig:
    batch_size: int = 4
    num_epochs: int = 200
    num_frames: int = 8
    img_size: int = 224
    lr_aug: float = 0.1
    lr_model: float = 1e-4
    seed: int = 42
    aug_strategy: str = "none"  # "none", "random", "adversarial-from-clean", "adversarial-from-random", "curriculum"
    # curriculum: no aug -> random aug (20) -> adversarial aug with scheduler (40)
    # curriculum2: no aug -> random aug (20) -> adversarial aug with scheduler (40) -> random aug (60)
    # curriculum3: no aug -> random aug (20) -> adversarial aug with scheduler (40) -> random aug (60) -> no aug (80)
    aug_scheduler: str = "none"  # "none", "linear", "exponential", "triangular"
    # only for curriculum
    blend_clean_loss: bool = True
    # adversarial training, blend clean loss with adversarial loss (by stacking batch). It will consume double memory.
    max_entropy_loss_weight: float = 1.0
    # for adversarial training. Regularize with max entropy loss. 0.0 means no regularization.
    aug_severity: int = 1
    aug_severity_max: int = 2  # for adversarial, allow bigger range for clipping
    curriculum_epochs: list[int] = field(
        default_factory=lambda: [20, 40, 60, 80, 90, 100]
        # default_factory=lambda: [1, 2, 3, 80, 90, 100]
    )
    dataset: str = "hmdb51-gulprgb-squeezed-noaug"
    test_dataset: str = ""
    model: str = "tsm_resnet50-pre=imagenet"
    num_classes: int | None = None
    split_num: int = 1
    cos_sim_eval_datasets: list[str] = field(
        default_factory=lambda: [
            "ucf-101-gulprgb-squeezed-noaug",
        ]
    )
    input_normalize: bool | None = None
    mean: list[float] | None = None
    std: list[float] | None = None
    early_stopping_patience: int | None = None

    def update_based_on_env_vars(self):
        self.verify_unknown_env_vars()
        for key, value in asdict(self).items():
            env_var = os.getenv(f"ML_{key}")
            if env_var:
                print(f"Updating {key} from environment variable ML_{key}={env_var}")
                if type(value) == list:
                    setattr(self, key, ast.literal_eval(env_var))
                else:
                    setattr(self, key, type(value)(env_var))

    def verify_unknown_env_vars(self):
        # os.environ.keys() is always uppercase
        for name, value in os.environ.items():
            keys_lower = [k.lower() for k in asdict(self).keys()]
            if name.startswith("ML_") and name[3:].lower() not in keys_lower:
                raise ValueError(f"Unknown environment variable {name}={value}")


class VideoAugmentation(nn.Module):
    def __init__(self, batch_size: int, num_frames: int, img_height: int):
        super(VideoAugmentation, self).__init__()
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.img_height = img_height

        # range: (-img_height//2, img_height//2)
        # self.translations = p(t([50., 50.]).expand(batch_size, -1))
        self.rotation_center = p(
            t([img_height / 2, img_height / 2]).expand(batch_size * num_frames, -1),
            requires_grad=False,
        )
        # self.rotation_angle = p(t([20.0]).expand(batch_size))
        # self.scale = p(t([1., 1.]).expand(batch_size, -1))
        # self.shear_x = p(t([0.1]).expand(batch_size))
        # self.shear_y = p(t([0.1]).expand(batch_size))

        # only define the shape of the tensors here
        self.translations = p(torch.zeros(batch_size, 2))
        self.rotation_angle = p(torch.zeros(batch_size))
        self.scale = p(torch.zeros(batch_size, 2))
        self.shear_x = p(torch.zeros(batch_size))
        self.shear_y = p(torch.zeros(batch_size))

        self.severity = 2
        self.translations_range = (-img_height // 4, img_height // 4)
        self.rotation_angle_range = (-20, 20)
        self.scale_range = (0.7, 2.0)
        self.shear_range = (-0.2, 0.2)

        self.transform_types = ["translation", "rotation", "scale", "shear"]

    def set_severity(self, severity: int):
        assert severity in [1, 2]
        if severity == 1:
            self.translations_range = (-self.img_height // 8, self.img_height // 8)
            self.rotation_angle_range = (-10, 10)
            self.scale_range = (0.9, 1.5)
            self.shear_range = (-0.1, 0.1)
        elif severity == 2:
            self.translations_range = (-self.img_height // 4, self.img_height // 4)
            self.rotation_angle_range = (-20, 20)
            self.scale_range = (0.7, 2.0)
            self.shear_range = (-0.2, 0.2)
        self.severity = severity

    def uniform_(self, severity: int | None = None):
        with torch.no_grad():
            if severity is not None:
                orig_severity = self.severity
                self.set_severity(severity)

            self.translations.uniform_(*self.translations_range)
            self.rotation_angle.uniform_(*self.rotation_angle_range)
            self.scale.uniform_(*self.scale_range)
            self.shear_x.uniform_(*self.shear_range)
            self.shear_y.uniform_(*self.shear_range)

            if severity is not None:
                # restore the original ranges
                self.set_severity(orig_severity)

    def no_op_(self):
        """
        No operation for all augmentation
        """
        with torch.no_grad():
            self.translations.zero_()
            self.rotation_angle.zero_()
            self.scale = p(torch.ones_like(self.scale))
            self.shear_x.zero_()
            self.shear_y.zero_()

    def clamp_(self):
        with torch.no_grad():
            num_clamped = 0

            num_clamped += torch.sum(self.translations < self.translations_range[0])
            num_clamped += torch.sum(self.translations > self.translations_range[1])
            self.translations.clamp_(*self.translations_range)

            num_clamped += torch.sum(self.rotation_angle < self.rotation_angle_range[0])
            num_clamped += torch.sum(self.rotation_angle > self.rotation_angle_range[1])
            self.rotation_angle.clamp_(*self.rotation_angle_range)

            num_clamped += torch.sum(self.scale < self.scale_range[0])
            num_clamped += torch.sum(self.scale > self.scale_range[1])
            self.scale.clamp_(*self.scale_range)

            num_clamped += torch.sum(self.shear_x < self.shear_range[0])
            num_clamped += torch.sum(self.shear_x > self.shear_range[1])
            self.shear_x.clamp_(*self.shear_range)

            num_clamped += torch.sum(self.shear_y < self.shear_range[0])
            num_clamped += torch.sum(self.shear_y > self.shear_range[1])
            self.shear_y.clamp_(*self.shear_range)

            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            # print(f"Clamped {num_clamped} values out of {num_params} parameters.")
        return num_clamped, num_params

    def forward(
        self,
        x,
        video_ids: torch.Tensor | None = None,
        transform_type: str = "translation",
        severity: int = 1,
    ):
        if self.training:
            # x: (B, T, C, H, W)
            B, T, C, H, W = x.shape

            x = x.view(B * T, C, H, W)

            affine = K.geometry.transform.get_affine_matrix2d(
                translations=self.translations.repeat_interleave(
                    repeats=self.num_frames, dim=0
                ),
                center=self.rotation_center,
                angle=self.rotation_angle.repeat_interleave(
                    repeats=self.num_frames, dim=0
                ),
                scale=self.scale.repeat_interleave(repeats=self.num_frames, dim=0),
                sx=self.shear_x.repeat_interleave(repeats=self.num_frames, dim=0),
                sy=self.shear_y.repeat_interleave(repeats=self.num_frames, dim=0),
            )
            # You need to remove the last column of the affine matrix
            # see https://github.com/kornia/kornia/blob/82525ee082985f507d8126e3a3b5e60ed98e8e29/kornia/augmentation/_2d/geometric/affine.py#L115
            affine = affine[:, :2, :]

            x = K.geometry.transform.warp_affine(x, affine, (H, W))

            return x.view(B, T, C, H, W)
        else:
            assert transform_type in self.transform_types
            assert 1 <= severity <= 30
            assert x.shape[0] == video_ids.shape[0]
            assert video_ids is not None

            # x: (B, T, C, H, W)
            B, T, C, H, W = x.shape
            batch_size = B

            translations = torch.zeros(batch_size, 2, device=x.device, dtype=x.dtype)
            center = p(
                t([H / 2, H / 2]).expand(B * T, -1),
                requires_grad=False,
            ).to(x.device)
            angle = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
            scale = torch.ones(batch_size, 2, device=x.device, dtype=x.dtype)
            sx = None
            sy = None

            # video_ids float to int
            video_ids = video_ids.long()

            if transform_type == "translation":
                direction = video_ids % 4
                translations = torch.zeros(
                    batch_size, 2, device=x.device, dtype=x.dtype
                )
                # top left for batches with direction == 0
                # top right for batches with direction == 1
                # bottom left for batches with direction == 2
                # bottom right for batches with direction == 3
                translations[:, 0] = (
                    (direction % 2 == 0) * self.translations_range[0] / 5 * severity
                ) + ((direction % 2 == 1) * self.translations_range[1] / 5 * severity)
                translations[:, 1] = (
                    (direction // 2 == 0) * self.translations_range[0] / 5 * severity
                ) + ((direction // 2 == 1) * self.translations_range[1] / 5 * severity)

            elif transform_type == "rotation":
                direction = video_ids % 2
                angle = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
                # counter-clockwise for batches with direction == 0
                # clockwise for batches with direction == 1
                angle[:] = (
                    (direction == 0) * self.rotation_angle_range[0] / 5 * severity
                ) + ((direction == 1) * self.rotation_angle_range[1] / 5 * severity)
            elif transform_type == "scale":
                direction = video_ids % 2
                scale = torch.zeros(batch_size, 2, device=x.device, dtype=x.dtype)
                # zoom in for batches with direction == 0
                # zoom out for batches with direction == 1
                scale[:, 0] = (
                    (direction == 0) * (1 - (1 - self.scale_range[0]) / 5 * severity)
                ) + ((direction == 1) * (1 + (self.scale_range[1] - 1) / 5 * severity))
                scale[:, 1] = (
                    (direction == 0) * (1 - (1 - self.scale_range[0]) / 5 * severity)
                ) + ((direction == 1) * (1 + (self.scale_range[1] - 1) / 5 * severity))
            elif transform_type == "shear":
                direction = video_ids % 4
                sx = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
                sy = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
                # sx < 0, sy < 0 for samples with direction == 0
                # sx > 0, sy < 0 for samples with direction == 1
                # sx < 0, sy > 0 for samples with direction == 2
                # sx > 0, sy > 0 for samples with direction == 3
                sx = ((direction % 2 == 0) * self.shear_range[0] / 5 * severity) + (
                    (direction % 2 == 1) * self.shear_range[1] / 5 * severity
                )
                sy = ((direction // 2 == 0) * self.shear_range[0] / 5 * severity) + (
                    (direction // 2 == 1) * self.shear_range[1] / 5 * severity
                )
            else:
                raise NotImplementedError

            translations = translations.repeat_interleave(
                repeats=self.num_frames, dim=0
            )
            angle = angle.repeat_interleave(repeats=self.num_frames, dim=0)
            scale = scale.repeat_interleave(repeats=self.num_frames, dim=0)
            if sx is not None:
                sx = sx.repeat_interleave(repeats=self.num_frames, dim=0)
            if sy is not None:
                sy = sy.repeat_interleave(repeats=self.num_frames, dim=0)

            x = x.view(B * T, C, H, W)

            affine = K.geometry.transform.get_affine_matrix2d(
                translations=translations,
                center=center,
                angle=angle,
                scale=scale,
                sx=sx,
                sy=sy,
            )
            # You need to remove the last column of the affine matrix
            # see https://github.com/kornia/kornia/blob/82525ee082985f507d8126e3a3b5e60ed98e8e29/kornia/augmentation/_2d/geometric/affine.py#L115
            affine = affine[:, :2, :]

            x = K.geometry.transform.warp_affine(x, affine, (H, W))

            return x.view(B, T, C, H, W)


class Entropy(nn.Module):
    """
    CE loss will encourage over-confident predictions.
    This loss should be used for adversarial examples to balance it out.
    Those examples shouldn't be overly confident but should have even probabilities across classes.

    In order to maximise entropy, use -entropy as a loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        probs = F.softmax(x, -1)
        log_probs = F.log_softmax(x, -1)
        return -(probs * log_probs).mean()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FeatureGatherer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.features = {}

    def update(self, features: torch.Tensor, labels: torch.Tensor):
        assert len(features) == len(labels)
        features = features.double()
        labels = labels.cpu().numpy().astype(np.int64)
        for feature, label in zip(features, labels):
            if label not in self.features:
                self.features[label] = [feature]
            else:
                self.features[label].append(feature)

    def get_features(self):
        features = {}
        for label in self.features:
            features[label] = torch.stack(self.features[label])
        return features

    def get_features_labels(self):
        features = []
        labels = []
        for label in self.features:
            features.extend(self.features[label])
            labels.extend([label] * len(self.features[label]))

        # Convert to torch tensors
        features = torch.stack(features)
        labels = torch.tensor(labels)
        return features, labels


cfg = BaseConfig()
cfg.update_based_on_env_vars()
set_seed(cfg.seed)

if cfg.aug_strategy not in ["curriculum", "curriculum2", "curriculum3"]:
    assert (
        cfg.aug_scheduler == "none"
    ), f"Only cfg.aug_scheduler='none' is allowed when cfg.aug_strategy=='curriculum'. Got {cfg.aug_scheduler}"

datasets = store["torch_dataset", cfg.dataset](
    data_dir=Path(os.environ["DATASET_DIR"]),
    size=cfg.img_size,
    ensure_installed=True,
    accelerator=accelerator,
    split_num=cfg.split_num,
)
datasets, class_names = instantiate(datasets)

if cfg.num_classes is None:
    cfg.num_classes = len(class_names)

if cfg.input_normalize is None:
    if cfg.model == "swin_t-pre=imagenet":
        cfg.input_normalize = False
    else:
        cfg.input_normalize = True

if cfg.mean is None:
    if cfg.model == "swin_t-pre=imagenet":
        cfg.mean = [123.675, 116.28, 103.53]
    else:
        cfg.mean = [0.485, 0.456, 0.406]

if cfg.std is None:
    if cfg.model == "swin_t-pre=imagenet":
        cfg.std = [58.395, 57.12, 57.375]
    else:
        cfg.std = [0.229, 0.224, 0.225]

pprint(cfg)

aug_model = VideoAugmentation(cfg.batch_size, cfg.num_frames, cfg.img_size)
aug_model.set_severity(cfg.aug_severity)
for param in aug_model.parameters():
    print(type(param.data), param.size())

if cfg.model == "uniformer_s-pre=imagenet":
    model = uniformer_small(num_classes=cfg.num_classes, pretrained="imagenet")
elif cfg.model == "swin_t-pre=imagenet":
    model = video_swin_tiny_imagenet(num_classes=cfg.num_classes)
else:
    model_zen = store["model", cfg.model](
        num_classes=cfg.num_classes, num_frames=cfg.num_frames
    )
    model: nn.Module = instantiate(model_zen)

# model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
print(model)
train_dataset = datasets["train"]
val_dataset = datasets["test"]
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)


criterion = torch.nn.CrossEntropyLoss()
entropy_criterion = Entropy()

optimizer_aug = torch.optim.SGD(aug_model.parameters(), lr=cfg.lr_aug, momentum=0.9)
# optimizer_model = torch.optim.AdamW(
#     model.get_optim_policies(), lr=cfg.lr_model, weight_decay=1e-5
# )
optimizer_model = torch.optim.SGD(
    model.get_optim_policies(), lr=cfg.lr_model, momentum=0.9, weight_decay=5e-4
)
# scheduler_model = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer_model, mode="min", factor=0.1, patience=10, verbose=True
# )

(
    aug_model,
    model,
    optimizer_aug,
    optimizer_model,
    train_dataloader,
    val_dataloader,
) = accelerator.prepare(
    aug_model,
    model,
    optimizer_aug,
    optimizer_model,
    train_dataloader,
    val_dataloader,
)

if cfg.aug_scheduler == "linear":
    # Start with lr = cfg.lr_aug and ends with cfg.lr_aug * 10
    scheduler_aug = torch.optim.lr_scheduler.LambdaLR(
        optimizer_aug,
        lambda x: 1 + 9 * (x / (cfg.num_epochs * len(train_dataloader))),
    )

    # Start with lr = cfg.lr_aug * 0.1 and ends with cfg.lr_aug
    # scheduler_aug = torch.optim.lr_scheduler.LinearLR(
    #     optimizer_aug,
    #     start_factor=0.1,
    #     total_iters=cfg.num_epochs * len(train_dataloader),
    # )
elif cfg.aug_scheduler == "exponential":
    # Start with lr = cfg.lr_aug and ends with cfg.lr_aug * 20
    scheduler_aug = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_aug,
        gamma=20 ** (1.0 / (cfg.num_epochs * len(train_dataloader))),
    )
elif cfg.aug_scheduler == "triangular":
    # 0.1 to 1.0 for 700 iterations. It goes down for the next 700. Repeats.
    scheduler_aug = torch.optim.lr_scheduler.CyclicLR(
        optimizer_aug,
        mode="triangular",
        base_lr=cfg.lr_aug,
        max_lr=cfg.lr_aug * 10,
        step_size_up=700,
    )
else:
    assert cfg.aug_scheduler == "none"
    # no-op
    scheduler_aug = torch.optim.lr_scheduler.ConstantLR(
        optimizer_aug,
        factor=1.0,
        total_iters=1,
    )

# len(train_dataloader) becomes smaller after prepare() when using DDP
scheduler_model = CosineLRScheduler(
    optimizer_model, t_initial=cfg.num_epochs * len(train_dataloader)
)


# wandb.init(project="video-augmentation", settings=wandb.Settings(start_method="fork"))
# wandb.init(
#     project="video-augmentation",
#     config=asdict(cfg),
# )
accelerator.init_trackers("video-augmentation", config=asdict(cfg))

if accelerator.is_main_process:
    # Find project root directory with .git directory
    CURRENT_DIR = Path(__file__).parent
    repo = git.Repo(CURRENT_DIR, search_parent_directories=True)
    wandb.run.log_code(repo.working_tree_dir)


mean = torch.tensor(cfg.mean, device=accelerator.device).view(1, 1, 3, 1, 1)
std = torch.tensor(cfg.std, device=accelerator.device).view(1, 1, 3, 1, 1)


def run_train_val(cfg, model, accelerator, mean, std):
    train_loss_pre_meter = AverageMeter()
    train_loss_meter = AverageMeter()
    train_accuracy_meter = AverageMeter()
    train_num_clamped_meter = AverageMeter()
    train_num_clamped_ratio_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    val_accuracy_meter = AverageMeter()
    val_probs_gatherer = FeatureGatherer()
    df_cm = None

    val_losses = []  # for early stopping
    val_accuracies = []  # for early stopping
    if cfg.aug_strategy in ["curriculum", "curriculum2", "curriculum3"]:
        # this will change over iterations
        aug_strategy = "none"
    else:
        aug_strategy = cfg.aug_strategy

    for epoch in range(cfg.num_epochs):
        if cfg.aug_strategy in ["curriculum", "curriculum2", "curriculum3"]:
            if epoch == cfg.curriculum_epochs[0]:
                aug_strategy = "clean+random"
                accelerator.unwrap_model(aug_model).set_severity(cfg.aug_severity)
            elif epoch == cfg.curriculum_epochs[1]:
                aug_strategy = "adversarial-from-random"
            # elif epoch in cfg.curriculum_epochs[2:]:
            #     for g in optimizer_aug.param_groups:
            #         g["lr"] *= 2

        if cfg.aug_strategy in ["curriculum2", "curriculum3"]:
            # End with random augmentation
            if epoch == cfg.curriculum_epochs[2]:
                aug_strategy = "random"

        if cfg.aug_strategy in ["curriculum3"]:
            if epoch == cfg.curriculum_epochs[3]:
                aug_strategy = "none"

        train_loss_pre_meter.reset()
        train_loss_meter.reset()
        train_accuracy_meter.reset()
        val_loss_meter.reset()
        val_accuracy_meter.reset()
        train_num_clamped_meter.reset()
        train_num_clamped_ratio_meter.reset()
        val_probs_gatherer.reset()

        print(f"Epoch {epoch} / {cfg.num_epochs-1}")

        aug_model.train()
        model.train()
        for train_iter, data in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
        ):
            step = epoch * len(train_dataloader) + train_iter

            # B, T, C, H, W
            batch = data["pixel_values"]
            labels = data["labels"]

            example_images = []
            if train_iter == 0:
                for i in range(3):
                    example_images.append(batch[i, 0].cpu().numpy().transpose(1, 2, 0))

            if aug_strategy in ["random", "clean+random"]:
                with torch.no_grad():
                    accelerator.unwrap_model(aug_model).uniform_()
            elif aug_strategy in ["adversarial-from-random", "adversarial-from-clean"]:
                if aug_strategy == "adversarial-from-random":
                    accelerator.unwrap_model(aug_model).uniform_(
                        severity=cfg.aug_severity
                    )
                elif aug_strategy == "adversarial-from-clean":
                    accelerator.unwrap_model(aug_model).no_op_()
                accelerator.unwrap_model(aug_model).set_severity(cfg.aug_severity_max)
                auged = aug_model(batch)

                if train_iter == 0:
                    for i in range(3):
                        example_images[i] = np.concatenate(
                            (
                                example_images[i],
                                auged[i, 0].cpu().detach().numpy().transpose(1, 2, 0),
                            ),
                            axis=0,
                        )

                if not cfg.input_normalize:
                    auged = auged * 255.0
                auged = auged - mean
                auged = auged / std
                logits = model(auged)
                loss_pre = criterion(logits, labels)
                if (
                    torch.isnan(loss_pre).any().item()
                    or torch.isinf(loss_pre).any().item()
                ):
                    print("Loss is NaN or Inf")
                    continue
                optimizer_aug.zero_grad()

                # We need the graph of the clean sample for later
                retain_graph = cfg.aug_strategy == "adversarial-from-clean"
                accelerator.backward(-loss_pre, retain_graph=retain_graph)

                # print(aug.translations)
                # print(aug.translations.grad)

                optimizer_aug.step()
                scheduler_aug.step()
                # print(aug.translations)
                num_clamped, num_params = accelerator.unwrap_model(aug_model).clamp_()
                num_clamped = accelerator.gather_for_metrics(num_clamped).sum().item()
                num_params = torch.tensor(
                    num_params, dtype=torch.int, device=accelerator.device
                )
                num_params = accelerator.gather_for_metrics(num_params).sum().item()
                train_num_clamped_meter.update(num_clamped)
                train_num_clamped_ratio_meter.update(num_clamped / num_params)
                # print(aug.translations)

            if aug_strategy in ["random", "clean+random", "adversarial-from-random"]:
                with torch.no_grad():
                    batch_clean = batch.detach().clone()
                    # batch = aug_model(batch_clean)
                    batch = aug_model(batch)

                if train_iter == 0:
                    for i in range(3):
                        example_images[i] = np.concatenate(
                            (
                                example_images[i],
                                batch[i, 0].cpu().numpy().transpose(1, 2, 0),
                            ),
                            axis=0,
                        )

            if train_iter == 0:
                example_images = [
                    wandb.Image(image, caption=f"{class_names[labels[i]]}")
                    for i, image in enumerate(example_images)
                ]
                accelerator.log(
                    {"examples/train": example_images},
                    step=step,
                )

            # Display multiple images
            # plt.figure()
            # plt.imshow(auged[0, 0].permute(1, 2, 0).cpu().detach().numpy())
            # plt.figure()
            # plt.imshow(hard_auged[0, 0].permute(1, 2, 0).cpu().detach().numpy())

            # if aug_strategy in ["adversarial-from-random", "adversarial-from-clean"]:
            #     if cfg.blend_clean_loss:
            #         with torch.no_grad():
            #             batch = torch.stack((batch, batch_clean), dim=0)
            #             labels = torch.cat((labels, labels), dim=0)

            with torch.no_grad():
                if not cfg.input_normalize:
                    batch = batch * 255.0
                batch = batch - mean
                batch = batch / std
            logits = model(batch)
            loss = criterion(logits, labels)

            if cfg.blend_clean_loss:
                if aug_strategy in ["clean+random", "adversarial-from-random"]:
                    with torch.no_grad():
                        if not cfg.input_normalize:
                            batch_clean = batch_clean * 255.0
                        batch_clean = batch_clean - mean
                        batch_clean = batch_clean / std
                    logits_clean = model(batch_clean)
                    loss_clean = criterion(logits_clean, labels)
                    loss = (loss + loss_clean) / 2
                elif aug_strategy == "adversarial-from-clean":
                    # We already have the gradient computed.
                    loss = (loss + loss_pre) / 2

            if aug_strategy in [
                "clean+random",
                "adversarial-from-random",
                "adversarial-from-clean",
            ]:
                if cfg.max_entropy_loss_weight > 0.0:
                    loss_entropy = entropy_criterion(logits)
                    loss = loss - cfg.max_entropy_loss_weight * loss_entropy
                    accelerator.log(
                        {"loss/train_entropy": loss_entropy},
                        step=step,
                    )

            if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
                print("Loss is NaN or Inf")
                continue
            optimizer_model.zero_grad()
            accelerator.backward(loss)
            optimizer_model.step()
            scheduler_model.step(epoch=step)
            # print(logits.softmax(dim=1))

            if aug_strategy in ["adversarial-from-random", "adversarial-from-clean"]:
                loss_pre = accelerator.gather_for_metrics(loss_pre).mean().item()
                train_loss_pre_meter.update(loss_pre)

            loss = accelerator.gather_for_metrics(loss).mean().item()
            train_loss_meter.update(loss)

            acc = (logits.argmax(dim=1) == labels).float()
            acc = accelerator.gather_for_metrics(acc).mean().item()
            train_accuracy_meter.update(acc)

            if aug_strategy in ["adversarial-from-random", "adversarial-from-clean"]:
                # wandb.log(
                accelerator.log(
                    {
                        "acc/train": train_accuracy_meter.avg,
                        "loss/train_pre": train_loss_pre_meter.avg,
                        "loss/train": train_loss_meter.avg,
                        "lr_model/train": optimizer_model.param_groups[0]["lr"],
                        "lr_aug/train": optimizer_aug.param_groups[0]["lr"],
                        "num_clamped/train": train_num_clamped_meter.avg,
                        "num_clamped_ratio/train": train_num_clamped_ratio_meter.avg,
                    },
                    step=step,
                )
            else:
                accelerator.log(
                    {
                        "acc/train": train_accuracy_meter.avg,
                        "loss/train": train_loss_meter.avg,
                        "lr/train": optimizer_model.param_groups[0]["lr"],
                    },
                    step=step,
                )

        with torch.no_grad():
            model.eval()
            for val_iter, data in tqdm(
                enumerate(val_dataloader),
                total=len(val_dataloader),
                disable=not accelerator.is_local_main_process,
            ):
                # B, T, C, H, W
                batch = data["pixel_values"]
                labels = data["labels"]

                if not cfg.input_normalize:
                    batch = batch * 255.0
                batch = batch - mean
                batch = batch / std
                logits = model(batch)
                loss = criterion(logits, labels)

                labels_all_procs = accelerator.gather_for_metrics(labels)
                logits_all_procs = accelerator.gather_for_metrics(logits)
                batch_size = labels_all_procs.shape[0]

                val_loss_all_procs = accelerator.gather_for_metrics(loss)
                val_loss_all_procs = val_loss_all_procs.mean().item()
                val_loss_meter.update(val_loss_all_procs, n=batch_size)
                val_acc = (
                    (logits_all_procs.argmax(dim=1) == labels_all_procs)
                    .float()
                    .mean()
                    .item()
                )
                val_accuracy_meter.update(val_acc, n=batch_size)

                probs = torch.nn.functional.softmax(logits_all_procs, dim=1)
                val_probs_gatherer.update(probs, labels_all_procs)

            # For ReduceLROnPlateau
            # scheduler_model.step(val_loss_meter.avg)
            val_probs, val_labels = val_probs_gatherer.get_features_labels()
            val_probs = val_probs.cpu().numpy()
            val_labels = val_labels.cpu().numpy()

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "acc/val": val_accuracy_meter.avg,
                        "loss/val": val_loss_meter.avg,
                    },
                    step=(epoch + 1) * len(train_dataloader) - 1,
                )

                if epoch % 10 == 0 or epoch == cfg.num_epochs - 1:
                    cm = confusion_matrix(
                        val_labels,
                        val_probs.argmax(axis=1),
                        labels=range(len(class_names)),
                    )
                    cm = normalize(
                        cm, axis=1, norm="l1"
                    )  # row (true labels) will sum to 1.
                    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

                    cm_fig = plot_confusion_matrix(df_cm)
                    cm_img = fig2img(cm_fig)
                    plt.close(cm_fig)

                    wandb.log(
                        {
                            # "conf_mat/val": wandb.plot.confusion_matrix(
                            #     probs=val_probs,
                            #     y_true=val_labels,
                            #     preds=None,
                            #     class_names=class_names,
                            # ),
                            "confusion_matrix/val": wandb.Image(cm_img),
                        },
                        step=(epoch + 1) * len(train_dataloader) - 1,
                    )

            # Early stopping
            # if val_loss didn't improve for 20 epochs, stop training
            if cfg.early_stopping_patience is not None:
                val_losses.append(val_loss_meter.avg)
                val_accuracies.append(val_accuracy_meter.avg)
                if (
                    len(val_losses) > cfg.early_stopping_patience
                    and min(val_losses[-cfg.early_stopping_patience :])
                    > min(val_losses)
                    and max(val_accuracies[-cfg.early_stopping_patience :])
                    < max(val_accuracies)
                ):
                    print("Early stopping")
                    break

    return df_cm


def run_corruption_error(
    cfg,
    aug_model,
    model,
    accelerator,
    mean,
    std,
    step,
):
    val_accuracy_meter = AverageMeter()

    print("Evaluating with multiple corruption")
    with torch.no_grad():
        aug_model.eval()
        model.eval()
        for transform_type in accelerator.unwrap_model(aug_model).transform_types:
            for severity in range(1, 31):
                print(f"Transform type: {transform_type}, severity: {severity}")
                val_accuracy_meter.reset()

                for val_iter, data in tqdm(
                    enumerate(val_dataloader),
                    total=len(val_dataloader),
                    disable=not accelerator.is_local_main_process,
                ):
                    # B, T, C, H, W
                    batch = data["pixel_values"]
                    labels = data["labels"]
                    video_ids = data["video_ids"]

                    batch = aug_model(batch, video_ids, transform_type, severity)

                    if not cfg.input_normalize:
                        batch = batch * 255.0
                    batch = batch - mean
                    batch = batch / std
                    logits = model(batch)
                    loss = criterion(logits, labels)

                    labels_all_procs = accelerator.gather_for_metrics(labels)
                    logits_all_procs = accelerator.gather_for_metrics(logits)
                    batch_size = labels_all_procs.shape[0]

                    val_loss_all_procs = accelerator.gather_for_metrics(loss)
                    val_loss_all_procs = val_loss_all_procs.mean().item()
                    val_acc = (
                        (logits_all_procs.argmax(dim=1) == labels_all_procs)
                        .float()
                        .mean()
                        .item()
                    )
                    val_accuracy_meter.update(val_acc, n=batch_size)

                if accelerator.is_main_process:
                    wandb.run.summary[
                        f"acc/val_{transform_type}_severity={severity}"
                    ] = val_accuracy_meter.avg


class FeatureExtractModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.is_ddp = True
        else:
            self.is_ddp = False

    def forward(self, x):
        batch_size = x.shape[0]
        if self.is_ddp:
            imagenet_features = self.model.module.features(x)
        else:
            imagenet_features = self.model.features(x)
        # It considers frames are image batch. Disentangle so you get actual video batch and number of frames.
        # Average over frames
        # return torch.mean(imagenet_features.view(batch_size, imagenet_features.shape[0] // batch_size, *imagenet_features.shape[1:]), dim=1)
        return imagenet_features.view(
            batch_size,
            imagenet_features.shape[0] // batch_size,
            *imagenet_features.shape[1:],
        )


class FeatureAverageMeter:
    def __init__(self):
        self.feature_sums = {}
        self.feature_counts = {}

    def update(self, features: torch.Tensor, labels: torch.Tensor):
        assert len(features) == len(labels)
        features = features.double()
        labels = labels.cpu().numpy().astype(np.int64)
        for feature, label in zip(features, labels):
            if label not in self.feature_sums:
                self.feature_sums[label] = feature
                self.feature_counts[label] = 1
            else:
                self.feature_sums[label] += feature
                assert not torch.isnan(self.feature_sums[label]).any()

                self.feature_counts[label] += 1

    def get_feature_averages(self):
        feature_averages = {}
        for label in self.feature_sums:
            feature_averages[label] = (
                self.feature_sums[label] / self.feature_counts[label]
            )
        return feature_averages


def run_cos_sim_eval(cfg, model, accelerator, mean, std, step):
    for cos_sim_eval_dataset in cfg.cos_sim_eval_datasets:
        if cos_sim_eval_dataset.startswith("ucf-101"):
            split_nums = [1, 2, 3]
        elif cos_sim_eval_dataset.startswith("hmdb51"):
            split_nums = [1, 2, 3]
        else:
            split_nums = [None]
        for cos_sim_eval_split in split_nums:
            print(
                f"Evaluating cosine similarity with {cos_sim_eval_dataset} split {cos_sim_eval_split}"
            )
            feature_model = FeatureExtractModel(accelerator.unwrap_model(model))

            if cos_sim_eval_split is None:
                datasets = store["torch_dataset", cos_sim_eval_dataset](
                    data_dir=Path(os.environ["DATASET_DIR"]),
                    size=cfg.img_size,
                    ensure_installed=True,
                    accelerator=accelerator,
                )
            else:
                datasets = store["torch_dataset", cos_sim_eval_dataset](
                    data_dir=Path(os.environ["DATASET_DIR"]),
                    size=cfg.img_size,
                    ensure_installed=True,
                    accelerator=accelerator,
                    split_num=cos_sim_eval_split,
                )

            datasets, class_names = instantiate(datasets)
            train_dataset = datasets["train"]
            if "test" in datasets:
                val_dataset = datasets["test"]
            else:
                val_dataset = datasets["val"]
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=False,
            )

            feature_model, train_dataloader, val_dataloader = accelerator.prepare(
                feature_model, train_dataloader, val_dataloader
            )

            train_feature_meter = FeatureAverageMeter()
            val_feature_gatherer = FeatureGatherer()
            cosine_similarity_loss_meter = AverageMeter()
            cosine_similarity_accuracy_meter = AverageMeter()
            val_cosine_similarity_probs_gatherer = FeatureGatherer()

            with torch.no_grad():
                feature_model.eval()
                for train_iter, data in tqdm(
                    enumerate(train_dataloader),
                    total=len(train_dataloader),
                    disable=not accelerator.is_local_main_process,
                ):
                    # B, T, C, H, W
                    batch = data["pixel_values"]
                    labels = data["labels"]
                    video_ids = data["video_ids"]

                    if not cfg.input_normalize:
                        batch = batch * 255.0
                    batch = batch - mean
                    batch = batch / std
                    features = feature_model(batch)
                    # B, T, F -> B, F
                    features = torch.mean(features, dim=1)
                    if features.dim() != 2:
                        # B, F, T, H, W ?
                        features = torch.flatten(features, start_dim=2).mean(-1)
                    features = accelerator.gather_for_metrics(features)
                    labels = accelerator.gather_for_metrics(labels)
                    train_feature_meter.update(features, labels)

                train_feature_averages = train_feature_meter.get_feature_averages()

                for val_iter, data in tqdm(
                    enumerate(val_dataloader),
                    total=len(val_dataloader),
                    disable=not accelerator.is_local_main_process,
                ):
                    # B, T, C, H, W
                    batch = data["pixel_values"]
                    labels = data["labels"]
                    video_ids = data["video_ids"]

                    if not cfg.input_normalize:
                        batch = batch * 255.0
                    batch = batch - mean
                    batch = batch / std
                    features = feature_model(batch)
                    # B, T, F -> B, F
                    features = torch.mean(features, dim=1)
                    if features.dim() != 2:
                        # B, F, T, H, W ?
                        features = torch.flatten(features, start_dim=2).mean(-1)
                    features = accelerator.gather_for_metrics(features)
                    labels = accelerator.gather_for_metrics(labels)
                    val_feature_gatherer.update(features, labels)

                val_features = val_feature_gatherer.get_features()

                # Train and val data may not cover the entire label space.
                # Only validate using classes that exist in train data
                labels_to_delete = []
                for label in val_features.keys():
                    if label not in train_feature_averages:
                        labels_to_delete.append(label)

                for label in labels_to_delete:
                    del val_features[label]

                assert set(val_features.keys()).issubset(
                    set(train_feature_averages.keys())
                ), f"{set(train_feature_averages.keys())} not superset of {set(val_features.keys())}"

                # label can be sparse and not continuous. Map to continuous index
                label_to_idx = {
                    label: idx
                    for idx, label in enumerate(train_feature_averages.keys())
                }

                # stack over sorted dictionary keys
                train_feature_averages = torch.stack(
                    [
                        train_feature_averages[label]
                        for label in sorted(train_feature_averages.keys())
                    ]
                )

                # normalised features
                train_feature_averages_normalised = (
                    train_feature_averages
                    / train_feature_averages.norm(p=2, dim=-1, keepdim=True)
                )

                # compute softmax based on cosine similarity
                for label, features in val_features.items():
                    # features: batch_size, F
                    # train_feature: num_classes, F
                    # cosine_similarity: batch_size, num_classes

                    label = label_to_idx[label]

                    # normalized features
                    features_normalised = features / features.norm(
                        p=2, dim=-1, keepdim=True
                    )

                    # cosine similarity as logits
                    logits = torch.matmul(
                        features_normalised, train_feature_averages_normalised.t()
                    )
                    assert logits.shape == (
                        features.shape[0],
                        train_feature_averages.shape[0],
                    )

                    # accuracy
                    accuracy = torch.mean(
                        (torch.argmax(logits, dim=-1) == label).float()
                    ).item()
                    cosine_similarity_accuracy_meter.update(accuracy, len(features))

                    # loss
                    logits = logits.to(accelerator.device)
                    labels = (
                        (torch.ones(features.shape[0]) * label)
                        .long()
                        .to(accelerator.device)
                    )
                    loss = F.cross_entropy(logits, labels)
                    cosine_similarity_loss_meter.update(loss.item(), len(features))

                    # gather probs
                    probs = F.softmax(logits, dim=-1)
                    val_cosine_similarity_probs_gatherer.update(probs, labels)

                if accelerator.is_main_process:
                    if wandb.run is not None:
                        (
                            val_probs,
                            val_labels,
                        ) = val_cosine_similarity_probs_gatherer.get_features_labels()
                        val_probs = val_probs.cpu().numpy()
                        val_labels = val_labels.cpu().numpy()

                        cm = confusion_matrix(
                            val_labels,
                            val_probs.argmax(axis=1),
                            labels=range(len(class_names)),
                        )
                        cm = normalize(
                            cm, axis=1, norm="l1"
                        )  # row (true labels) will sum to 1.
                        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

                        cm_fig = plot_confusion_matrix(df_cm)
                        cm_img = fig2img(cm_fig)
                        plt.close(cm_fig)

                        if cos_sim_eval_split is None:
                            wandb.run.summary[
                                f"acc/{cos_sim_eval_dataset}-cosine_similarity"
                            ] = cosine_similarity_accuracy_meter.avg
                            wandb.run.summary[
                                f"loss/{cos_sim_eval_dataset}-cosine_similarity"
                            ] = cosine_similarity_loss_meter.avg
                            wandb.log(
                                {
                                    # f"conf_mat/{cos_sim_eval_dataset}-cosine_similarity": wandb.plot.confusion_matrix(
                                    #     probs=val_probs,
                                    #     y_true=val_labels,
                                    #     preds=None,
                                    #     class_names=class_names,
                                    # ),
                                    f"confusion_matrix/{cos_sim_eval_dataset}-cosine_similarity": wandb.Image(
                                        cm_img
                                    ),
                                },
                                step=step,
                            )
                        else:
                            wandb.run.summary[
                                f"acc/{cos_sim_eval_dataset}-split{cos_sim_eval_split}-cosine_similarity"
                            ] = cosine_similarity_accuracy_meter.avg
                            wandb.run.summary[
                                f"loss/{cos_sim_eval_dataset}-split{cos_sim_eval_split}-cosine_similarity"
                            ] = cosine_similarity_loss_meter.avg
                            wandb.log(
                                {
                                    # f"conf_mat/{cos_sim_eval_dataset}-split{cos_sim_eval_split}-cosine_similarity": wandb.plot.confusion_matrix(
                                    #     probs=val_probs,
                                    #     y_true=val_labels,
                                    #     preds=None,
                                    #     class_names=class_names,
                                    # ),
                                    f"confusion_matrix/{cos_sim_eval_dataset}-split{cos_sim_eval_split}-cosine_similarity": wandb.Image(
                                        cm_img
                                    ),
                                },
                                step=step,
                            )


# Test
def run_test(cfg, model, accelerator, mean, std, step):
    df_cm = None
    if cfg.test_dataset != "":
        print(f"Evaluating test accuracy with {cfg.test_dataset}")
        # datasets = store["torch_dataset", cfg.test_dataset](
        #     data_dir=Path(os.environ["DATASET_DIR"]),
        #     size=cfg.img_size,
        #     ensure_installed=True,
        #     accelerator=accelerator,
        #     sets_to_include=["test"],
        # )
        datasets = store["torch_dataset", cfg.test_dataset](
            data_dir=Path(os.environ["DATASET_DIR"]),
            crop_size=cfg.img_size,
            ensure_installed=True,
            accelerator=accelerator,
            sets_to_include=["test"],
            test_scale=cfg.img_size,
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
        )
        datasets, class_names = instantiate(datasets)
        test_dataset = datasets["test"]

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        test_dataloader = accelerator.prepare(test_dataloader)

        val_accuracy_meter = AverageMeter()
        val_loss_meter = AverageMeter()
        val_probs_gatherer = FeatureGatherer()
        with torch.no_grad():
            model.eval()
            for test_iter, data in tqdm(
                enumerate(test_dataloader),
                total=len(test_dataloader),
                disable=not accelerator.is_local_main_process,
            ):
                # B, T, C, H, W
                batch = data["pixel_values"]
                labels = data["labels"]

                if test_iter == 0:
                    example_images = []
                    for i in range(3):
                        example_images.append(
                            batch[i, 0].cpu().numpy().transpose(1, 2, 0)
                        )

                    example_images = [
                        wandb.Image(image, caption=f"{class_names[labels[i]]}")
                        for i, image in enumerate(example_images)
                    ]
                    accelerator.log(
                        {"examples/test": example_images},
                        step=step,
                    )

                if not cfg.input_normalize:
                    batch = batch * 255.0
                batch = batch - mean
                batch = batch / std
                logits = model(batch)
                loss = criterion(logits, labels)

                labels_all_procs = accelerator.gather_for_metrics(labels)
                logits_all_procs = accelerator.gather_for_metrics(logits)
                batch_size = labels_all_procs.shape[0]

                val_loss_all_procs = accelerator.gather_for_metrics(loss)
                val_loss_all_procs = val_loss_all_procs.mean().item()
                val_loss_meter.update(val_loss_all_procs, n=batch_size)
                val_acc = (
                    (logits_all_procs.argmax(dim=1) == labels_all_procs)
                    .float()
                    .mean()
                    .item()
                )
                val_accuracy_meter.update(val_acc, n=batch_size)

                probs = torch.nn.functional.softmax(logits_all_procs, dim=1)
                val_probs_gatherer.update(probs, labels_all_procs)

            val_probs, val_labels = val_probs_gatherer.get_features_labels()
            val_probs = val_probs.cpu().numpy()
            val_labels = val_labels.cpu().numpy()

            print(f"Test accuracy: {val_accuracy_meter.avg}")
            print(f"Test loss: {val_loss_meter.avg}")

            if accelerator.is_main_process:
                cm = confusion_matrix(
                    val_labels, val_probs.argmax(axis=1), labels=range(len(class_names))
                )
                cm = normalize(
                    cm, axis=1, norm="l1"
                )  # row (true labels) will sum to 1.
                df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

                cm_fig = plot_confusion_matrix(df_cm)
                cm_img = fig2img(cm_fig)
                plt.close(cm_fig)

                wandb.run.summary["acc/test"] = val_accuracy_meter.avg
                wandb.run.summary["loss/test"] = val_loss_meter.avg
                wandb.log(
                    {
                        "confusion_matrix/test": wandb.Image(cm_img),
                    },
                    step=step,
                )

    return df_cm


trainval_start_time = time.time()
val_df_cm = run_train_val(cfg, model, accelerator, mean, std)
trainval_duration = time.time() - trainval_start_time

cossim_start_time = time.time()
run_cos_sim_eval(
    cfg,
    model,
    accelerator,
    mean,
    std,
    step=(cfg.num_epochs) * len(train_dataloader) - 1,
)
cossim_duration = time.time() - cossim_start_time

corruption_test_start_time = time.time()
run_corruption_error(
    cfg,
    aug_model,
    model,
    accelerator,
    mean,
    std,
    step=(cfg.num_epochs) * len(train_dataloader) - 1,
)
corruption_test_duration = time.time() - corruption_test_start_time

test_start_time = time.time()
test_df_cm = run_test(
    cfg,
    model,
    accelerator,
    mean,
    std,
    step=(cfg.num_epochs) * len(train_dataloader) - 1,
)
test_duration = time.time() - test_start_time

run_duration = time.time() - run_start_time

if accelerator.is_main_process:
    assert wandb.run is not None
    wandb.run.summary["trainval_duration"] = trainval_duration
    wandb.run.summary["cossim_duration"] = cossim_duration
    wandb.run.summary["corruption_duration"] = corruption_test_duration
    wandb.run.summary["test_duration"] = test_duration
    wandb.run.summary["run_duration"] = run_duration

    confusion_matrix = {"val": val_df_cm, "test": test_df_cm}
    with open(os.path.join(wandb.run.dir, "confusion_matrix.pkl"), "wb") as f:
        pickle.dump(confusion_matrix, f)

    wandb.save(os.path.join(wandb.run.dir, "confusion_matrix.pkl"))

accelerator.end_training()
