import numpy as np
import torch
from PIL import Image

import mlproject

TEST_VIDEO_PATHS = [
    mlproject.ROOT_DIR
    / "tests"
    / "res"
    / "hmdb51_tests"
    / "frames_q5"
    / "hit"
    / "Collins_getting_hit_with_a_stick_on_the_head_hit_f_cm_np1_le_bad_4",
    mlproject.ROOT_DIR
    / "tests"
    / "res"
    / "hmdb51_tests"
    / "frames_q5"
    / "hit"
    / "Destruction_of_a_TV_hit_f_cm_np1_ri_bad_2",
    mlproject.ROOT_DIR
    / "tests"
    / "res"
    / "hmdb51_tests"
    / "frames_q5"
    / "run"
    / "50_FIRST_DATES_run_f_cm_np1_ba_med_12",
    mlproject.ROOT_DIR
    / "tests"
    / "res"
    / "hmdb51_tests"
    / "frames_q5"
    / "run"
    / "BLACK_HAWK_DOWN_run_f_cm_np1_le_med_15",
]


def get_test_video_batch() -> torch.Tensor:
    """Get a batch of test videos."""
    video_batch = []
    for video_path in TEST_VIDEO_PATHS:
        video = []
        image_paths = [
            video_path / "00000.jpg",
            video_path / "00005.jpg",
            video_path / "00010.jpg",
            video_path / "00015.jpg",
            video_path / "00020.jpg",
            video_path / "00025.jpg",
            video_path / "00030.jpg",
            video_path / "00035.jpg",
        ]
        for image_path in image_paths:
            image = Image.open(image_path)
            image = image.resize((224, 224))
            video.append(np.array(image))
        video_batch.append(video)

    video_batch = torch.tensor(np.array(video_batch))  # B, T, H, W, C
    video_batch = video_batch.permute(0, 1, 4, 2, 3)  # B, T, C, H, W
    return video_batch
