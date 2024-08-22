import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from loguru import logger
from webdataset import TarWriter

from xares.audiowebdataset import create_rawaudio_webdataset, write_audio_tar
from xares.models import Mlp
from xares.task_base import TaskBase
from xares.utils import download_file, mkdir_if_not_exists, untar_file


@dataclass
class LibriSpeechMaleFemaleTask(TaskBase):
    save_encoded_per_batches = 1000  # If OOM, reduce this number
    batch_size = 32
    trim_length = 100_320  # 33440*3s
    output_dim = 2
    metric = "accuracy"

    def __post_init__(self):
        self.ori_data_root = self.env_dir / "LibriSpeech"
        self.wds_audio_paths_train = self.env_dir / f"wds-audio-train-*.tar"
        self.wds_audio_paths_dev = self.env_dir / "wds-audio-dev-*.tar"
        self.wds_audio_paths_test_clean = self.env_dir / "wds-audio-test-clean-*.tar"
        self.wds_audio_paths_test_other = self.env_dir / "wds-audio-test-other-*.tar"

        self.wds_encoded_paths_train = self.env_dir / f"wds-encoded-train-*.tar"
        self.wds_encoded_paths_dev = self.env_dir / "wds-encoded-dev-*.tar"
        self.wds_encoded_paths_test_clean = self.env_dir / "wds-encoded-test-clean-*.tar"
        self.wds_encoded_paths_test_other = self.env_dir / "wds-encoded-test-other-*.tar"

        self.wds_audio_paths = [
            self.wds_audio_paths_train,
            self.wds_audio_paths_dev,
            self.wds_audio_paths_test_clean,
            self.wds_audio_paths_test_other,
        ]
        self.wds_encoded_paths = [
            self.wds_encoded_paths_train,
            self.wds_encoded_paths_dev,
            self.wds_encoded_paths_test_clean,
            self.wds_encoded_paths_test_other,
        ]

        self.model = Mlp(in_features=self.encoder.output_dim, out_features=self.output_dim)
        self.checkpoint_dir = self.env_dir / "checkpoints"

    def make_audio_tar(self):
        if not self.force_generate_audio_tar and self.audio_tar_ready_file.exists():
            logger.info(f"Skip making audio tar. {self.audio_tar_ready_file} already exists.")
            return

        # Download and extract LibriSpeech dataset
        mkdir_if_not_exists(self.env_dir)
        download_file(
            "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
            self.env_dir / "train.tar.gz",
            force=self.force_download,
        )
        download_file(
            "https://www.openslr.org/resources/12/dev-clean.tar.gz",
            self.env_dir / "dev.tar.gz",
            force=self.force_download,
        )
        download_file(
            "https://www.openslr.org/resources/12/test-clean.tar.gz",
            self.env_dir / "test_clean.tar.gz",
            force=self.force_download,
        )
        download_file(
            "https://www.openslr.org/resources/12/test-other.tar.gz",
            self.env_dir / "test_other.tar.gz",
            force=self.force_download,
        )

        if not self.ori_data_root.exists():
            logger.info(f"Extracting {self.env_dir / 'train.tar.gz'} to {self.env_dir}...")
            untar_file(self.env_dir / "train.tar.gz", self.env_dir)
            logger.info(f"Extracting {self.env_dir / 'dev.tar.gz'} to {self.env_dir}...")
            untar_file(self.env_dir / "dev.tar.gz", self.env_dir)
            logger.info(f"Extracting {self.env_dir / 'test_clean.tar.gz'} to {self.env_dir}...")
            untar_file(self.env_dir / "test_clean.tar.gz", self.env_dir)
            logger.info(f"Extracting {self.env_dir / 'test_other.tar.gz'} to {self.env_dir}...")
            untar_file(self.env_dir / "test_other.tar.gz", self.env_dir)
        else:
            logger.info(f"Directory {self.ori_data_root} already exists. Skip.")

        # Create tar file with audio files
        with open(self.ori_data_root / "SPEAKERS.TXT", "r") as f:
            id_gender = {
                int(line.split("|")[0]): 1 if line.split("|")[1].strip() == "F" else 0
                for line in f
                if not line.strip().startswith(";")
            }

        datasets = ["train-clean-100", "dev-clean", "test-clean", "test-other"]

        for dataset, wds_audio_path in zip(datasets, self.wds_audio_paths):
            paths = os.walk(self.ori_data_root / dataset)
            file_paths = []
            speakers = []
            for path, dir_lst, file_lst in paths:
                for file_name in file_lst:
                    file_path = Path(os.path.join(path, file_name))
                    if str(file_path).endswith(".flac"):
                        file_paths.append(file_path.as_posix())
                        speaker = int(str(file_path.as_posix()).split("/")[4])
                        speakers.append(speaker)

            audio_speaker = dict(zip(file_paths, speakers))
            targets = [id_gender[key] for key in audio_speaker.values()]
            write_audio_tar(
                file_paths,
                targets,
                wds_audio_path.as_posix(),
                force=self.force_generate_audio_tar,
            )

        self.audio_tar_ready_file.touch()

    def make_encoded_tar(self, num_shards: int = 20):
        if not self.force_generate_encoded_tar and self.encoded_tar_ready_file.exists():
            logger.info(f"Skip making encoded tar. {self.encoded_tar_ready_file} already exists.")
            return

        def write_encoded_batches_to_wds(encoded_batches: List, ostream: TarWriter):
            for batch, label, keys in encoded_batches:
                for example, label, key in zip(batch, label, keys):
                    sample = {
                        "pth": example,
                        "json": json.dumps({"target": label["label"]}).encode("utf-8"),
                        "__key__": key,
                    }
                    ostream.write(sample)

        for wds_audio_path, wds_encoded_path in zip(self.wds_audio_paths, self.wds_encoded_paths):
            logger.info(f"Encoding audio for {wds_audio_path} ...")
            for shard in range(num_shards):
                sharded_tar_path = wds_audio_path.as_posix().replace("*", f"0{shard:05d}")
                dl = create_rawaudio_webdataset(
                    [sharded_tar_path],
                    batch_size=self.batch_size,
                    num_workers=self.num_encoder_workers,
                    crop_size=self.trim_length,
                    with_json=True,
                    remain_random_one_crop=True,
                )

                batch_buf = []
                sharded_encoded_tar_path = wds_encoded_path.as_posix().replace("*", f"0{shard:05d}")
                with TarWriter(sharded_encoded_tar_path) as ostream:
                    for batch, length, label, keys in dl:
                        encoded_batch = self.encoder(batch, 33_440)
                        batch_buf.append([encoded_batch, label, keys])

                        if len(batch_buf) >= self.save_encoded_per_batches:
                            write_encoded_batches_to_wds(batch_buf, ostream)
                            batch_buf.clear()
                    if len(batch_buf) > 0:
                        write_encoded_batches_to_wds(batch_buf, ostream)

        self.encoded_tar_ready_file.touch()

    def run_all(self) -> float:
        self.make_audio_tar()
        self.make_encoded_tar()

        self.ckpt_name = f"best_model.pt"
        self.ckpt_path = self.checkpoint_dir / self.ckpt_name
        self.train_mlp(
            [self.wds_encoded_paths_train.as_posix()],
            [self.wds_encoded_paths_dev.as_posix()],
        )
        acc_clean = self.evaluate_mlp(
            [self.wds_encoded_paths_test_clean.as_posix()], metric=self.metric, load_ckpt=True
        )
        logger.info(f"The test_clean accuracy is: {acc_clean}")

        acc_other = self.evaluate_mlp(
            [self.wds_encoded_paths_test_other.as_posix()], metric=self.metric, load_ckpt=True
        )
        logger.info(f"The test_other accuracy is: {acc_other}")

        return acc_clean
