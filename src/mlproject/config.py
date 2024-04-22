import logging
import multiprocessing
import os
from dataclasses import dataclass
from typing import Any

import wandb
from accelerate import Accelerator
from hydra.core.config_store import ConfigStore
from hydra_zen import MISSING, ZenField, builds, make_config, store
from torch.utils.data import DataLoader

# Importing `store_configs` populates the hydra-zen store.
# We're not actually using the module in this file, but we want to run the code.
from .aug_models import store_configs  # noqa: F811  # pyright: ignore
from .boilerplate import Learner
from .callbacks import UploadCheckpointsToHuggingFace
from .datasets import config as datasets_config
from .datasets import store_configs  # noqa: F811  # pyright: ignore
from .models import store_configs  # noqa: F811  # pyright: ignore
from .optimizers import store_configs  # noqa: F811  # pyright: ignore
from .schedulers import store_configs  # noqa: F811  # pyright: ignore
from .torch_datasets import store_configs  # noqa: F811  # pyright: ignore

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = "${hf_repo_dir}"
NUM_WORKERS = "${num_workers}"
HF_USERNAME = "${hf_username}"
CODE_DIR = "${code_dir}"
DATASET_DIR = "${data_dir}"
EXPERIMENT_NAME = "${exp_name}"
EXPERIMENTS_ROOT_DIR = "${root_experiment_dir}"
TRAIN_BATCH_SIZE = "${train_batch_size}"
CURRENT_EXPERIMENT_DIR = "${current_experiment_dir}"
TRAIN_ITERS = "${learner.train_iters}"
REPO_PATH = "${repo_path}"
EXP_NAME = "${exp_name}"
SEED = "${seed}"
RESUME = "${resume}"
INPUT_NUM_FRAMES = "${input_num_frames}"


wandb_args_config = builds(wandb.init, populate_full_signature=True)

wandb_args_default = wandb_args_config(
    project=os.environ.get("WANDB_PROJECT", "mlproject"),
    resume="allow",  # allow, True, False, must
    dir=CURRENT_EXPERIMENT_DIR,
    save_code=True,
)


HFModelUploadConfig = builds(
    UploadCheckpointsToHuggingFace, populate_full_signature=True
)

hf_upload = HFModelUploadConfig(repo_name=EXPERIMENT_NAME, repo_owner=HF_USERNAME)

accelerator_config = builds(Accelerator, populate_full_signature=True)

dataloader_config = builds(DataLoader, dataset=None, populate_full_signature=True)

learner_config = builds(Learner, populate_full_signature=True)

learner_config = learner_config(
    model=None,
    experiment_name=EXPERIMENT_NAME,
    experiment_dir=CHECKPOINT_DIR,
    resume=RESUME,
    evaluate_every_n_steps=500,
    checkpoint_after_validation=True,
    checkpoint_every_n_steps=500,
    train_iters=10000,
)

default_callbacks = dict(hf_uploader=hf_upload)


@dataclass
class BaseConfig:
    # Must be passed at command line -- neccesary arguments

    exp_name: str = MISSING

    # Defaults for these are provided in the collect_config_store method,
    # but will be often overridden at command line

    model: Any = MISSING
    aug_model: Any = MISSING
    dataset: datasets_config.DatasetConfig | Any = MISSING
    torch_dataset: Any = MISSING
    dataloader: Any = MISSING
    optimizer: Any = MISSING
    scheduler: Any = MISSING
    learner: Any = MISSING
    callbacks: Any = MISSING

    wandb_args: Any = MISSING

    input_num_frames: int = 8

    hf_username: str = (
        os.environ["HF_USERNAME"] if "HF_USERNAME" in os.environ else MISSING
    )

    seed: int = 42

    freeze_backbone: bool = False
    resume: bool = False
    resume_from_checkpoint: int | None = None
    print_config: bool = False
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = multiprocessing.cpu_count() // 2
    train: bool = True
    test: bool = False
    download_latest: bool = True
    download_checkpoint_with_name: str | None = None

    root_experiment_dir: str = (
        os.environ["EXPERIMENTS_DIR"]
        if "EXPERIMENTS_DIR" in os.environ
        else "/experiments"
    )

    data_dir: str = (
        os.environ["DATASET_DIR"] if "DATASET_DIR" in os.environ else "/data"
    )

    current_experiment_dir: str = "${root_experiment_dir}/${exp_name}"
    repo_path: str = "${hf_username}/${exp_name}"
    hf_repo_dir: str = "${current_experiment_dir}/repo"
    code_dir: str = (
        os.environ["CODE_DIR"] if "CODE_DIR" in os.environ else "${hydra:runtime.cwd}"
    )


# Using hydra might look a bit more verbose but it saves having to manually define
# future args, and makes it a lot easier to add whatever we need from the command line


def collect_config_store():
    config_store = ConfigStore.instance()

    # model, torch_dataset, optimizer, scheduler
    store.add_to_hydra_store()

    config_store.store(
        group="dataloader",
        name="default",
        node=dataloader_config(
            batch_size=TRAIN_BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            shuffle=True,
        ),
    )

    ###################################################################################
    config_store.store(
        group="learner",
        name="default",
        node=learner_config,
    )

    config_store.store(group="callbacks", name="default", node=default_callbacks)

    config_store.store(group="wandb_args", name="default", node=wandb_args_default)

    config_store.store(
        group="hydra",
        name="default",
        node=dict(
            job_logging=dict(
                version=1,
                formatters=dict(
                    simple=dict(
                        level="INFO",
                        format="%(message)s",
                        datefmt="[%X]",
                    )
                ),
                handlers=dict(
                    rich={
                        "class": "rich.logging.RichHandler",
                        "formatter": "simple",
                    }
                ),
                root={"handlers": ["rich"], "level": "INFO"},
                disable_existing_loggers=False,
            ),
            hydra_logging=dict(
                version=1,
                formatters=dict(
                    simple=dict(
                        level="INFO",
                        format="%(message)s",
                        datefmt="[%X]",
                    )
                ),
                handlers={
                    "rich": {
                        "class": "rich.logging.RichHandler",
                        "formatter": "simple",
                    }
                },
                root={"handlers": ["rich"], "level": "INFO"},
                disable_existing_loggers=False,
            ),
            run={"dir": "${current_experiment_dir}/hydra-run/${now:%Y-%m-%d_%H-%M-%S}"},
            sweep={
                "dir": "${current_experiment_dir}/hydra-multirun/${now:%Y-%m-%d_%H-%M-%S}",
                "subdir": "${hydra.job.num}",
            },
        ),
    )

    zen_config = []

    for value in BaseConfig.__dataclass_fields__.values():
        item = (
            ZenField(name=value.name, hint=value.type, default=value.default)
            if value.default is not MISSING
            else ZenField(name=value.name, hint=value.type)
        )
        zen_config.append(item)

    hydra_config = make_config(
        *zen_config,
        hydra_defaults=[
            "_self_",
            dict(learner="default"),
            dict(optimizer="SGD"),
            dict(scheduler="MultiStepLR"),
            dict(model="tsm_resnet50-pre=kinetics400"),
            dict(aug_model="image_augmentation"),
            dict(dataset="hmdb51"),
            dict(torch_dataset="hmdb51-gulprgb"),
            dict(dataloader="default"),
            dict(hydra="default"),
            dict(wandb_args="default"),
            dict(callbacks="default"),
        ],
    )

    # Config
    config_store.store(name="config", node=hydra_config)

    return hydra_config
