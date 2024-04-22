import os
from pathlib import Path

from accelerate import Accelerator
from huggingface_hub import snapshot_download

from . import zen_store
from .loader.gulp_sparsesample_dataset import GulpSparsesampleDataset
from .loader.gulp_sparsesample_squeezed_dataset import GulpSparsesampleSqueezedDataset


def get_class_names(dataset_root, class_names_filename="class_names.txt"):
    dataset_root = Path(dataset_root)

    with open(dataset_root / "metadata" / class_names_filename, "r") as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


@zen_store(
    name="hmdb51-gulprgb", dataset_name="hmdb51-gulprgb", data_dir="${data_dir}/hmdb51"
)
@zen_store(
    name="hmdb51-gulprgb-noaug",
    dataset_name="hmdb51-gulprgb",
    data_dir="${data_dir}/hmdb51",
    train_jitter_min=224,
    train_jitter_max=224,
    train_horizontal_flip=False,
    test_scale=224,
)
def build_gulp_dataset(
    dataset_name: str,
    data_dir: str | Path,
    sets_to_include=None,
    train_jitter_min=224,
    train_jitter_max=336,
    train_horizontal_flip=True,
    test_scale=256,
    test_num_spatial_crops=1,
    split_num: int = 1,
    crop_size=224,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    normalise=True,
    bgr=False,
    ensure_installed=True,
    accelerator: Accelerator | None = None,
    class_names_filename="class_names.txt",
):
    assert dataset_name in ["hmdb51-gulprgb", "epic-kitchens-100-gulprgb"]
    cache_dir = os.environ.get("HF_CACHE_DIR", None)
    assert cache_dir is not None

    data_dir = Path(data_dir)

    if dataset_name == "hmdb51-gulprgb":
        assert split_num in [1, 2, 3]
        if data_dir.name != "hmdb51":
            data_dir = data_dir / "hmdb51"
        if sets_to_include is None:
            sets_to_include = ["train", "test"]
    elif dataset_name == "epic-kitchens-100-gulprgb":
        assert split_num == 1
        if data_dir.name != "epic-kitchens-100":
            data_dir = data_dir / "epic-kitchens-100"
        if sets_to_include is None:
            sets_to_include = ["train", "val"]
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    assert sets_to_include is not None

    if ensure_installed:
        if accelerator is None or accelerator.is_local_main_process:
            snapshot_download(
                repo_id=f"kiyoonkim/{dataset_name}",
                repo_type="dataset",
                resume_download=True,
                local_dir=data_dir,
                cache_dir=cache_dir,
                # allow_patterns="splits_gulp_rgb/*",
            )

        if accelerator is not None:
            accelerator.wait_for_everyone()

    dataset = {}
    for set_name in sets_to_include:
        input_frame_length = 8
        gulp_dir_path = data_dir / "gulp_rgb"

        if dataset_name == "hmdb51-gulprgb":
            if set_name == "train":
                mode = "train"
                csv_path = data_dir / "splits_gulp_rgb" / f"train{split_num}.csv"
            else:
                mode = "test"
                csv_path = data_dir / "splits_gulp_rgb" / f"test{split_num}.csv"
        elif dataset_name == "epic-kitchens-100-gulprgb":
            if set_name == "train":
                mode = "train"
                csv_path = data_dir / "verb_splits_gulp_rgb" / "train.csv"
            elif set_name == "val":
                mode = "test"
                csv_path = data_dir / "verb_splits_gulp_rgb" / "val.csv"
            else:
                raise ValueError(f"Unknown set_name: {set_name}")
        else:
            raise NotImplementedError

        data = GulpSparsesampleDataset(
            csv_path,
            mode,
            input_frame_length,
            gulp_dir_path,
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
            greyscale=False,
            sample_index_code="pyvideoai",
            processing_backend="pil",
            frame_neighbours=1,
            pil_transforms_after=None,
        )
        dataset[set_name] = data

    class_names = get_class_names(data_dir, class_names_filename)

    return dataset, class_names


@zen_store(
    name="hmdb51-gulprgb-squeezed-noaug",
    dataset_name="hmdb51-gulprgb",
    data_dir="${data_dir}/hmdb51",
    size=224,
)
@zen_store(
    name="ucf-101-gulprgb-squeezed-noaug",
    dataset_name="ucf-101-gulprgb",
    data_dir="${data_dir}",
    size=224,
)
@zen_store(
    name="hmdb51-kinetics-overlap-gulprgb-squeezed-noaug",
    dataset_name="hmdb51-gulprgb",
    data_dir="${data_dir}/hmdb51",
    size=224,
    split_dir="splits_kinetics_overlap_gulp_rgb",
    class_names_filename="class_names_kinetics_overlap.txt",
)
@zen_store(
    name="ucf-101-kinetics-overlap-gulprgb-squeezed-noaug",
    dataset_name="ucf-101-gulprgb",
    data_dir="${data_dir}",
    size=224,
    split_dir="splits_kinetics_overlap_gulp_rgb",
    class_names_filename="class_names_kinetics_overlap.txt",
)
@zen_store(
    name="epic-kitchens-100-gulprgb-squeezed-noaug",
    dataset_name="epic-kitchens-100-gulprgb",
    data_dir="${data_dir}",
    size=224,
)
@zen_store(
    name="something-something-v1-gulprgb-squeezed-noaug",
    dataset_name="something-something-v1-gulprgb",
    data_dir="${data_dir}",
    size=224,
)
def build_squeezed_gulp_dataset(
    dataset_name: str,
    data_dir: str | Path,
    sets_to_include=None,
    split_dir: str | Path = "splits_gulp_rgb",
    split_num: int = 1,
    size=224,
    data_format="BTCHW",
    ensure_installed=True,
    accelerator: Accelerator | None = None,
    class_names_filename="class_names.txt",
):
    assert dataset_name in [
        "hmdb51-gulprgb",
        "ucf-101-gulprgb",
        "epic-kitchens-100-gulprgb",
        "something-something-v1-gulprgb",
    ]
    cache_dir = os.environ.get("HF_CACHE_DIR", None)
    assert cache_dir is not None

    data_dir = Path(data_dir)

    if dataset_name == "hmdb51-gulprgb":
        assert split_num in [1, 2, 3]
        if data_dir.name != "hmdb51":
            data_dir = data_dir / "hmdb51"
        if sets_to_include is None:
            sets_to_include = ["train", "test"]
    elif dataset_name == "ucf-101-gulprgb":
        assert split_num in [1, 2, 3]
        if data_dir.name != "ucf101":
            data_dir = data_dir / "ucf101"
        if sets_to_include is None:
            sets_to_include = ["train", "test"]
    elif dataset_name == "epic-kitchens-100-gulprgb":
        assert split_num == 1
        if data_dir.name != "epic-kitchens-100":
            data_dir = data_dir / "epic-kitchens-100"
        if sets_to_include is None:
            sets_to_include = ["train", "val"]
    elif dataset_name == "something-something-v1-gulprgb":
        assert split_num == 1
        if data_dir.name != "something-something-v1":
            data_dir = data_dir / "something-something-v1"
        if sets_to_include is None:
            sets_to_include = ["train", "val"]
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    assert sets_to_include is not None

    if ensure_installed:
        if accelerator is None or accelerator.is_local_main_process:
            snapshot_download(
                repo_id=f"kiyoonkim/{dataset_name}",
                repo_type="dataset",
                resume_download=True,
                local_dir=data_dir,
                cache_dir=cache_dir,
                # allow_patterns="splits_gulp_rgb/*",
            )

        if accelerator is not None:
            accelerator.wait_for_everyone()

    dataset = {}
    for set_name in sets_to_include:
        input_frame_length = 8
        gulp_dir_path = data_dir / "gulp_rgb"
        if dataset_name == "hmdb51-gulprgb":
            if set_name == "train":
                mode = "train"
                csv_path = data_dir / split_dir / f"train{split_num}.csv"
            else:
                mode = "test"
                csv_path = data_dir / split_dir / f"test{split_num}.csv"
        elif dataset_name == "ucf-101-gulprgb":
            if set_name == "train":
                mode = "train"
                csv_path = data_dir / split_dir / f"trainlist{split_num:02d}.txt"
            elif set_name == "test":
                mode = "test"
                csv_path = data_dir / split_dir / f"testlist{split_num:02d}.txt"
            else:
                raise ValueError(f"Unknown set_name: {set_name}")
        elif dataset_name == "epic-kitchens-100-gulprgb":
            if set_name == "train":
                gulp_dir_path = gulp_dir_path / "train"
                mode = "train"
                csv_path = data_dir / split_dir / "train.csv"
            elif set_name == "val":
                gulp_dir_path = gulp_dir_path / "val"
                mode = "test"
                csv_path = data_dir / split_dir / "val.csv"
            else:
                raise ValueError(f"Unknown set_name: {set_name}")
        elif dataset_name == "something-something-v1-gulprgb":
            if set_name == "train":
                mode = "train"
                csv_path = data_dir / split_dir / "train.csv"
            elif set_name == "val":
                mode = "test"
                csv_path = data_dir / split_dir / "val.csv"
            else:
                raise ValueError(f"Unknown set_name: {set_name}")
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        data = GulpSparsesampleSqueezedDataset(
            csv_path,
            mode,
            input_frame_length,
            gulp_dir_path,
            size=size,
            data_format=data_format,
            pil_transforms_after=None,
        )
        dataset[set_name] = data

    class_names = get_class_names(data_dir, class_names_filename)

    return dataset, class_names
