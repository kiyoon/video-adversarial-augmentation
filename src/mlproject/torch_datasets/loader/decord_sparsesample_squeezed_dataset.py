# Code inspired from https://github.com/facebookresearch/SlowFast
import logging
import os
from pathlib import Path

import decord
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from decord import VideoReader

from . import transform as transform
from . import utils as utils

logger = logging.getLogger(__name__)


class DecordSparsesampleSqueezedDataset(torch.utils.data.Dataset):
    """
    Minimal dataloader that has almost no transforms except squeezing to the desired size.
    """

    def __init__(
        self,
        csv_file,
        mode,
        num_frames,
        path_prefix: str | Path,
        size=224,
        data_format="BTCHW",  # BTCHW, BCTHW
        num_decord_threads=1,
    ):
        """
        Construct the video loader with a given csv file. The format of
        the csv file is:
        ```
        num_classes     # set it to zero for single label. Only needed for multilabel.
        path_to_video_1.mp4 video_id_1 label_1 start_frame_1 end_frame_1 width_1 height_2
        path_to_video_2.mp4 video_id_2 label_2 start_frame_2 end_frame_2 width_2 height_2
        ...
        path_to_video_3.mp4 video_id_N label_N start_frame_N end_frame_N width_N height_N
        ```
        Args:
            mode (str): Options includes `train`, or `test` mode.
                For the train, the data loader will take data
                from the train set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
        """
        # Only support train, and test mode.
        assert mode in [
            "train",
            "test",
        ], "Split '{}' not supported".format(mode)
        self._csv_file = csv_file
        self._path_prefix = Path(path_prefix)
        self._num_decord_threads = num_decord_threads
        self.mode = mode

        self.size = size
        self.num_frames = num_frames

        assert data_format in ["BTCHW", "BCTHW"]
        self.data_format = data_format

        logger.info(f"Constructing decord video dataset {mode=}...")
        self._construct_loader()

        decord.bridge.set_bridge("torch")

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        assert os.path.exists(self._csv_file), "{} not found".format(self._csv_file)

        self._path_to_videos = []
        self._video_ids = []
        self._labels = []
        self._start_frames = []  # number of sample video frames
        self._end_frames = []  # number of sample video frames
        self._widths = []
        self._heights = []
        with open(self._csv_file, "r") as f:
            self.num_classes = int(f.readline())
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split()) == 7
                (
                    path,
                    video_id,
                    label,
                    start_frame,
                    end_frame,
                    width,
                    height,
                ) = path_label.split()

                if self.num_classes > 0:
                    label_list = label.split(",")
                    label = np.zeros(self.num_classes, dtype=np.float32)
                    for label_idx in label_list:
                        label[int(label_idx)] = 1.0  # one hot encoding
                else:
                    label = int(label)

                self._path_to_videos.append(os.path.join(self._path_prefix, path))
                self._video_ids.append(int(video_id))
                self._labels.append(label)
                self._start_frames.append(int(start_frame))
                self._end_frames.append(int(end_frame))
                self._widths.append(int(width))
                self._heights.append(int(height))
        assert (
            len(self._path_to_videos) > 0
        ), f"Failed to load video loader from {self._csv_file}"
        logger.info(
            "Constructing video dataloader (size: {}) from {}".format(
                len(self._path_to_videos), self._csv_file
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            pixel_values (tensor): the frames of sampled from the video.
            video_id (int): the ID of the current video.
            label (int): the label of the current video.
        """

        size = self.size

        if self.mode == "train":
            sample_uniform = False
        else:
            sample_uniform = True

        # Decode video. Meta info is used to perform selective decoding.
        #        frame_indices = utils.TRN_sample_indices(self._num_sample_frames[index], self.num_frames, mode = self.mode)
        num_video_frames = self._end_frames[index] - self._start_frames[index] + 1
        frame_indices = utils.sparse_frame_indices(
            num_video_frames,
            self.num_frames,
            uniform=sample_uniform,
            # num_neighbours=self.frame_neighbours,
        )

        frame_indices = [
            idx + self._start_frames[index] for idx in frame_indices
        ]  # add offset (frame number start)

        new_width, new_height = transform.get_size_random_short_side_scale_jitter(
            self._widths[index], self._heights[index], size, size
        )
        vr = VideoReader(
            self._path_to_videos[index],
            width=new_width,
            height=new_height,
            num_threads=self._num_decord_threads,
        )
        frames = vr.get_batch(frame_indices)

        frames = frames / 255.0

        # T, H, W, C -> T, C, H, W
        frames = frames.permute(0, 3, 1, 2)
        frames = F.interpolate(frames, size=(size, size), mode="bilinear")
        if self.data_format == "BCTHW":
            # T, C, H, W -> C, T, H, W
            frames = frames.permute(1, 0, 2, 3)

        video_id = self._video_ids[index]
        label = self._labels[index]

        return {
            "pixel_values": frames,
            "video_ids": video_id,
            "labels": label,
            "indices": index,
            "frame_indices": np.array(frame_indices),
        }

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
