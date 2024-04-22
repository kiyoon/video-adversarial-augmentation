import os
from pathlib import Path

import numpy as np
import pytest
import simplejpeg
from gulpio2 import GulpDirectory

SOURCE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize(
    "key,frame_size,num_frames",
    [
        (
            "hit/Collins_getting_hit_with_a_stick_on_the_head_hit_f_cm_np1_le_bad_4",
            [240, 320],
            43,
        ),
        ("hit/Destruction_of_a_TV_hit_f_cm_np1_ri_bad_2", [240, 360], 39),
        ("run/50_FIRST_DATES_run_f_cm_np1_ba_med_12", [240, 320], 49),
        ("run/BLACK_HAWK_DOWN_run_f_cm_np1_le_med_15", [240, 560], 47),
    ],
)
def test_gulp_dataloader(
    gulp_hmdb51_rgb: GulpDirectory, key: str, frame_size: list[int], num_frames: int
):
    video, metadata = gulp_hmdb51_rgb[key]
    assert metadata["frame_size"] == frame_size
    assert metadata["num_frames"] == num_frames
    assert video[0].shape == (*frame_size, 3)
    assert len(video) == num_frames


@pytest.mark.xfail(raises=KeyError)
@pytest.mark.parametrize("key", ["aaa", "run", "walk", "bbb"])
def test_gulp_dataloader_wrong_key(gulp_hmdb51_rgb: GulpDirectory, key):
    gulp_hmdb51_rgb[key]


@pytest.mark.parametrize(
    "key,frame_idx",
    [
        (
            "hit/Collins_getting_hit_with_a_stick_on_the_head_hit_f_cm_np1_le_bad_4",
            0,
        ),
        ("hit/Destruction_of_a_TV_hit_f_cm_np1_ri_bad_2", 0),
        ("run/50_FIRST_DATES_run_f_cm_np1_ba_med_12", 0),
        ("run/BLACK_HAWK_DOWN_run_f_cm_np1_le_med_15", 0),
    ],
)
def test_gulp_equals_jpeg(gulp_hmdb51_rgb: GulpDirectory, key, frame_idx):
    video, metadata = gulp_hmdb51_rgb[key]

    # read bytes from jpeg
    jpeg_path = Path(
        SOURCE_DIR / "res" / "hmdb51_tests" / "frames_q5" / key / f"{frame_idx:05d}.jpg"
    )
    with open(jpeg_path, "rb") as f:
        jpeg_bytes = f.read()

    assert np.all(
        simplejpeg.decode_jpeg(
            jpeg_bytes, fastdct=True, fastupsample=True, colorspace="RGB"
        )
        == video[frame_idx]
    )
