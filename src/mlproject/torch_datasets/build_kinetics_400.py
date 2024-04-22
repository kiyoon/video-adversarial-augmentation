import os
from pathlib import Path

from accelerate import Accelerator
from huggingface_hub import snapshot_download

from . import zen_store
from .kinetics_400 import prepare_kinetics_400, prepare_kinetics_400_test_set
from .loader.decord_sparsesample_dataset import DecordSparsesampleDataset
from .loader.decord_sparsesample_squeezed_dataset import (
    DecordSparsesampleSqueezedDataset,
)


def get_class_names(dataset_root, class_names_filename="class_names.txt"):
    dataset_root = Path(dataset_root)

    with open(dataset_root / "k400" / "metadata" / class_names_filename, "r") as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


@zen_store(name="kinetics-400")
@zen_store(
    name="kinetics-400-hmdb-overlap",
    split_dir="splits_hmdb_overlap_decord_videos",
    class_names_filename="class_names_hmdb_overlap.txt",
)
@zen_store(
    name="kinetics-400-ucf-overlap",
    split_dir="splits_ucf_overlap_decord_videos",
    class_names_filename="class_names_ucf_overlap.txt",
)
def build_kinetics_400_dataset(
    data_dir: str | Path,
    sets_to_include=None,
    split_dir="splits_decord_videos",
    train_jitter_min=224,
    train_jitter_max=336,
    train_horizontal_flip=True,
    test_scale=256,
    test_num_spatial_crops=1,
    crop_size=224,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    normalise=True,
    bgr=False,
    ensure_installed=True,
    accelerator: Accelerator | None = None,
    class_names_filename="class_names.txt",
):
    cache_dir = os.environ.get("HF_CACHE_DIR", None)
    assert cache_dir is not None

    data_dir = Path(data_dir)

    if data_dir.name != "kinetics-dataset":
        data_dir = data_dir / "kinetics-dataset"

    if sets_to_include is None:
        sets_to_include = ["train", "val", "test"]

    if ensure_installed:
        if accelerator is None or accelerator.is_local_main_process:
            if sets_to_include == ["test"]:
                # Test only
                prepare_kinetics_400_test_set(data_dir)
            else:
                prepare_kinetics_400(data_dir)
            snapshot_download(
                repo_id="kiyoonkim/kinetics-400-splits",
                repo_type="dataset",
                resume_download=True,
                local_dir=data_dir / "k400",
                cache_dir=cache_dir,
                # allow_patterns="splits_gulp_rgb/*",
            )

        if accelerator is not None:
            accelerator.wait_for_everyone()

    dataset = {}
    for set_name in sets_to_include:
        input_frame_length = 8
        videos_dir = data_dir / "k400" / "videos"

        if set_name == "train":
            mode = "train"
            csv_path = data_dir / "k400" / split_dir / "train.csv"
            videos_dir = videos_dir / "train"
        elif set_name == "val":
            mode = "test"
            csv_path = data_dir / "k400" / split_dir / "val.csv"
            videos_dir = videos_dir / "val"
        elif set_name == "test":
            mode = "test"
            csv_path = data_dir / "k400" / split_dir / "test.csv"
            videos_dir = videos_dir / "test"
        else:
            raise ValueError(f"Unknown set_name: {set_name}")

        data = DecordSparsesampleDataset(
            csv_path,
            mode,
            input_frame_length,
            train_jitter_min=train_jitter_min,
            train_jitter_max=train_jitter_max,
            train_horizontal_flip=train_horizontal_flip,
            test_scale=test_scale,
            test_num_spatial_crops=test_num_spatial_crops,
            crop_size=crop_size,
            mean=mean,
            std=std,
            normalise=normalise,
            bgr=bgr,
            sample_index_code="pyvideoai",
            path_prefix=videos_dir,
        )
        dataset[set_name] = data

    class_names = get_class_names(data_dir, class_names_filename=class_names_filename)

    return dataset, class_names


@zen_store(
    name="kinetics-400-squeezed-hmdb-overlap",
    split_dir="splits_hmdb_overlap_decord_videos",
    class_names_filename="class_names_hmdb_overlap.txt",
)
@zen_store(
    name="kinetics-400-squeezed-ucf-overlap",
    split_dir="splits_ucf_overlap_decord_videos",
    class_names_filename="class_names_ucf_overlap.txt",
)
def build_kinetics_400_squeezed_dataset(
    data_dir: str | Path,
    sets_to_include=None,
    split_dir="splits_decord_videos",
    size=224,
    ensure_installed=True,
    accelerator: Accelerator | None = None,
    class_names_filename="class_names.txt",
):
    cache_dir = os.environ.get("HF_CACHE_DIR", None)
    assert cache_dir is not None

    data_dir = Path(data_dir)

    if data_dir.name != "kinetics-dataset":
        data_dir = data_dir / "kinetics-dataset"

    if sets_to_include is None:
        sets_to_include = ["train", "val", "test"]

    if ensure_installed:
        if accelerator is None or accelerator.is_local_main_process:
            if sets_to_include == ["test"]:
                # Test only
                prepare_kinetics_400_test_set(data_dir)
            else:
                prepare_kinetics_400(data_dir)
            snapshot_download(
                repo_id="kiyoonkim/kinetics-400-splits",
                repo_type="dataset",
                resume_download=True,
                local_dir=data_dir / "k400",
                cache_dir=cache_dir,
                # allow_patterns="splits_gulp_rgb/*",
            )

        if accelerator is not None:
            accelerator.wait_for_everyone()

    dataset = {}
    for set_name in sets_to_include:
        input_frame_length = 8
        videos_dir = data_dir / "k400" / "videos"

        if set_name == "train":
            mode = "train"
            csv_path = data_dir / "k400" / split_dir / "train.csv"
            videos_dir = videos_dir / "train"
        elif set_name == "val":
            mode = "test"
            csv_path = data_dir / "k400" / split_dir / "val.csv"
            videos_dir = videos_dir / "val"
        elif set_name == "test":
            mode = "test"
            csv_path = data_dir / "k400" / split_dir / "test.csv"
            videos_dir = videos_dir / "test"
        else:
            raise ValueError(f"Unknown set_name: {set_name}")

        data = DecordSparsesampleSqueezedDataset(
            csv_path,
            mode,
            input_frame_length,
            size=size,
            path_prefix=videos_dir,
        )
        dataset[set_name] = data

    class_names = get_class_names(data_dir, class_names_filename=class_names_filename)

    return dataset, class_names
