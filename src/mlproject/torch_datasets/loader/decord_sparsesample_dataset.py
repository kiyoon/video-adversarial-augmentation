# Code inspired from https://github.com/facebookresearch/SlowFast
import logging
import os
from pathlib import Path

import decord
import numpy as np
import torch
import torch.utils.data
from decord import VideoReader

from . import transform as transform
from . import utils as utils

logger = logging.getLogger(__name__)


class DecordSparsesampleDataset(torch.utils.data.Dataset):
    """
    Video loader. Construct the video loader, then sample
    clips from the videos. For training, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the center
    and four corners.
    """

    def __init__(
        self,
        csv_file,
        mode,
        num_frames,
        train_jitter_min=256,
        train_jitter_max=320,
        train_horizontal_flip=True,
        test_scale=256,
        test_num_spatial_crops=10,
        crop_size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        normalise=True,  # divide pixels by 255
        bgr=False,
        path_prefix: str | Path = "",
        sample_index_code="pyvideoai",
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
            sample_index_code (str): Options include `pyvideoai`, `TSN` and `TDN`.
                Slightly different implementation of how video is sampled (pyvideoai and TSN),
                and for the TDN, it is completely different as it samples num_frames*5 frames.
        """
        # Only support train, and test mode.
        assert mode in [
            "train",
            "test",
        ], "Split '{}' not supported".format(mode)
        self._csv_file = csv_file
        self._path_prefix = path_prefix
        self._num_decord_threads = num_decord_threads
        self.mode = mode
        self.sample_index_code = sample_index_code.lower()

        self.train_jitter_min = train_jitter_min
        self.train_jitter_max = train_jitter_max
        self.test_scale = test_scale

        self.train_horizontal_flip = train_horizontal_flip

        self.num_frames = num_frames

        self.crop_size = crop_size

        assert len(mean) in [1, 3]
        assert len(std) in [1, 3]
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

        self.normalise = normalise
        self.bgr = bgr

        # For training mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = test_num_spatial_crops

        assert test_num_spatial_crops in [
            1,
            5,
            10,
        ], "1 for centre, 5 for centre and four corners, 10 for their horizontal flips"
        self.test_num_spatial_crops = test_num_spatial_crops

        logger.info("Constructing video dataset {}...".format(mode))
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
        self._spatial_temporal_idx = []
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

                for idx in range(self._num_clips):
                    self._path_to_videos.append(os.path.join(self._path_prefix, path))
                    self._video_ids.append(int(video_id))
                    self._labels.append(label)
                    self._start_frames.append(int(start_frame))
                    self._end_frames.append(int(end_frame))
                    self._widths.append(int(width))
                    self._heights.append(int(height))
                    self._spatial_temporal_idx.append(idx)
        assert (
            len(self._path_to_videos) > 0
        ), f"Failed to load video loader from {self._csv_file}"
        logger.info(
            "Constructing video dataloader (size: {}) from {}".format(
                len(self._path_to_videos), self._csv_file
            )
        )

    def filter_samples(self, video_ids: list):
        """Given a video_ids list, filter the samples.
        Used for visualisation.
        """
        indices_of_video_ids = [
            x for x, v in enumerate(self._video_ids) if v in video_ids
        ]

        self._path_to_videos = [self._path_to_videos[x] for x in indices_of_video_ids]
        self._video_ids = [self._video_ids[x] for x in indices_of_video_ids]
        self._labels = [self._labels[x] for x in indices_of_video_ids]
        self._start_frames = [self._start_frames[x] for x in indices_of_video_ids]
        self._end_frames = [self._end_frames[x] for x in indices_of_video_ids]
        self._widths = [self._widths[x] for x in indices_of_video_ids]
        self._heights = [self._heights[x] for x in indices_of_video_ids]
        self._spatial_temporal_idx = [
            self._spatial_temporal_idx[x] for x in indices_of_video_ids
        ]

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            video_id (int): the ID of the current video.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        crop_size = self.crop_size
        if self.mode in ["train"]:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.train_jitter_min
            max_scale = self.train_jitter_max
            sample_uniform = False
        elif self.mode in ["test"]:
            # spatial_sample_index is in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
            spatial_sample_index = (
                self._spatial_temporal_idx[index] % self.test_num_spatial_crops
            )
            min_scale, max_scale = [self.test_scale] * 2
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale are expect to be the same.
            assert len({min_scale, max_scale}) == 1
            sample_uniform = True
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        # Decode video. Meta info is used to perform selective decoding.
        #        frame_indices = utils.TRN_sample_indices(self._num_sample_frames[index], self.num_frames, mode = self.mode)
        num_video_frames = self._end_frames[index] - self._start_frames[index] + 1
        if self.sample_index_code == "pyvideoai":
            frame_indices = utils.sparse_frame_indices(
                num_video_frames, self.num_frames, uniform=sample_uniform
            )
        elif self.sample_index_code == "tsn":
            frame_indices = utils.TSN_sample_indices(
                num_video_frames, self.num_frames, mode=self.mode
            )
        else:
            raise ValueError(
                f"Wrong self.sample_index_code: {self.sample_index_code}. Should be"
                " pyvideoai, TSN"
            )
        frame_indices = [
            idx + self._start_frames[index] for idx in frame_indices
        ]  # add offset (frame number start)

        new_width, new_height = transform.get_size_random_short_side_scale_jitter(
            self._widths[index], self._heights[index], min_scale, max_scale
        )
        vr = VideoReader(
            self._path_to_videos[index],
            width=new_width,
            height=new_height,
            num_threads=self._num_decord_threads,
        )
        frames = vr.get_batch(frame_indices)
        if self.bgr:
            frames = frames[..., ::-1]

        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.mean, self.std, normalise=self.normalise
        )

        # T, H, W, C -> C, T, H, W
        frames = frames.permute(3, 0, 1, 2)

        # Perform data augmentation.
        (frames, _, _, x_offset, y_offset, is_flipped,) = utils.spatial_sampling_5(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=None,  # Already rescaled using decord
            max_scale=None,
            crop_size=crop_size,
            random_horizontal_flip=self.train_horizontal_flip,
        )

        # C, T, H, W -> T, C, H, W
        frames = frames.permute(1, 0, 2, 3)

        video_id = self._video_ids[index]
        label = self._labels[index]

        return {
            "pixel_values": frames,
            "video_ids": video_id,
            "labels": label,
            "spatial_sample_indices": spatial_sample_index,
            "indices": index,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "is_flipped": is_flipped,
            "frame_indices": np.array(frame_indices),
        }

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
