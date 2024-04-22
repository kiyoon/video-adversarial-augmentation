import os
import tempfile
from multiprocessing import cpu_count
from pathlib import Path

import pytest
from gulpio2 import GulpDirectory, GulpIngestor

from mlproject.torch_datasets.loader.generic_gulp_adaptor import (
    GenericJpegDatasetAdapter,
)

SOURCE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def gulp_hmdb51_rgb():
    in_folder = SOURCE_DIR / "res" / "hmdb51_tests" / "frames_q5"
    gulp_adapter = GenericJpegDatasetAdapter(
        video_segment_dir=str(in_folder), frame_size=-1, class_folder=True
    )

    with tempfile.TemporaryDirectory() as tmp_out_folder:
        ingestor = GulpIngestor(
            adapter=gulp_adapter,
            output_folder=tmp_out_folder,
            videos_per_chunk=100,
            num_workers=cpu_count(),
        )
        ingestor()

        yield GulpDirectory(tmp_out_folder)
