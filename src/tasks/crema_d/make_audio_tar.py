import argparse
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from xares.audiowebdataset import write_audio_tar
from xares.common import XaresSettings
from xares.utils import mkdir_if_not_exists


def make_audio_tar(env_root: Optional[str] = None, force_download=False, force_generate_audio_tar=True, num_shards=4):
    settings = XaresSettings()
    target_dir = Path(env_root) if env_root else Path(settings.env_root) / "crema-d"
    audio_tar_ready_file_path = target_dir / settings.audio_ready_filename

    if not force_generate_audio_tar and audio_tar_ready_file_path.exists():
        logger.info(f"Skip making audio tar. {audio_tar_ready_file_path} already exists.")
        return

    # Download CREMA-D dataset
    mkdir_if_not_exists(target_dir)
    ori_data_root = target_dir / "CREMA-D"

    if not ori_data_root.exists():
        git_lfs_command = [
            "git",
            "lfs",
            "clone",
            "https://github.com/CheyneyComputerScience/CREMA-D.git",
            ori_data_root,
        ]
        try:
            subprocess.run(git_lfs_command, check=True)
            logger.info(f"Repository cloned to {ori_data_root}")
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Error cloning repository: {e}. "
                f"You may want to 'git lfs clone https://github.com/CheyneyComputerScience/CREMA-D.git' "
                f"to clone the dataset to {ori_data_root} manually."
            )
    else:
        logger.info(f"Directory {ori_data_root} already exists. Skip.")

    # Create tar file with audio files
    df = pd.read_csv(ori_data_root / "finishedResponses.csv", usecols=["clipName", "dispEmo", "queryType"])
    df = df[df["queryType"] == 1].drop(columns=["queryType"])  # Keep data with audio feedback only
    df = df.drop_duplicates(subset="clipName")
    df.clipName = df.clipName.apply(lambda x: (ori_data_root / "AudioWAV" / f"{x}.wav").as_posix())
    df = df.sample(frac=1).reset_index(drop=True)

    # 15% for test；15% for valid；70% for train
    test_size = int(len(df) * 0.15)
    split_df_dic = {
        "test": df.iloc[:test_size],
        "valid": df.iloc[test_size : 2 * test_size],
        "train": df.iloc[2 * test_size :],
    }
    for split in split_df_dic:
        df_split = split_df_dic[split]
        write_audio_tar(
            audio_paths=df_split.clipName.tolist(),
            labels=df_split.dispEmo.tolist(),
            tar_path=(target_dir / f"wds-audio-{split}-*.tar").as_posix(),
            force=force_generate_audio_tar,
        )

    audio_tar_ready_file_path.touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make audio tar files.")
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
