import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from xares.audiowebdataset import write_audio_tar
from xares.common import XaresSettings
from xares.utils import download_file, mkdir_if_not_exists, unzip_file


def make_audio_tar(env_root: Optional[str] = None, force_download=False, force_generate_audio_tar=True, num_shards=4):
    settings = XaresSettings()
    target_dir = Path(env_root) if env_root else Path(settings.env_root) / "esc50"
    audio_tar_ready_file_path = target_dir / settings.audio_ready_filename

    if not force_generate_audio_tar and audio_tar_ready_file_path.exists():
        logger.info(f"{audio_tar_ready_file_path} already exists.")
        return

    # Download and extract ESC-50 dataset
    mkdir_if_not_exists(target_dir)
    zip_path = target_dir / "master.zip"
    download_file(
        "https://github.com/karoldvl/ESC-50/archive/master.zip",
        zip_path,
        force=force_download,
    )

    ori_data_root = target_dir / "ESC-50-master"
    if not ori_data_root.exists():
        logger.info(f"Extracting {target_dir / 'master.zip'} to {target_dir}...")
        unzip_file(target_dir / "master.zip", target_dir)
    else:
        logger.info(f"Directory {ori_data_root} already exists. Skip.")

    # Create tar file with audio files
    df = pd.read_csv(ori_data_root / "meta/esc50.csv", usecols=["filename", "fold", "target"])
    df.filename = df.filename.apply(lambda x: (ori_data_root / "audio" / x).as_posix())
    splits = df.fold.unique().tolist()
    for fold in splits:
        df_split = df[df.fold == fold].drop(columns=["fold"])
        tar_path = target_dir / f"wds-audio-fold-{fold}-*.tar"
        write_audio_tar(
            audio_paths=df_split.filename.tolist(),
            labels=df_split.target.tolist(),
            tar_path=tar_path.as_posix(),
            force=force_generate_audio_tar,
            num_shards=num_shards,
        )
    audio_tar_ready_file_path.touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make audio tar files for ESC-50 dataset.")
    parser.add_argument("--env_root", type=str, default=None, help="Root directory for the environment.")
    parser.add_argument("--force_download", action="store_true", help="Force download of the dataset.")
    parser.add_argument(
        "--force_generate_audio_tar", action="store_true", help="Force generation of the audio tar files."
    )
    parser.add_argument("--num_shards", type=int, default=4, help="Number of shards.")
    args = parser.parse_args()

    make_audio_tar(
        env_root=args.env_root,
        force_download=args.force_download,
        force_generate_audio_tar=args.force_generate_audio_tar,
        num_shards=args.num_shards,
    )
