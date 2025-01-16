import argparse
import os
from pathlib import Path

import pandas as pd
from loguru import logger

from xares.audiowebdataset import write_audio_tar
from xares.common import XaresSettings
from xares.utils import download_file, mkdir_if_not_exists, unzip_file


def make_audio_tar(env_root: None | str = None, force_download=False, force_generate_audio_tar=True, num_shards=4):
    settings = XaresSettings()
    target_dir = Path(env_root) if env_root else Path(settings.env_root) / "vocalsound"
    audio_tar_ready_file_path = target_dir / settings.audio_ready_filename

    if not force_generate_audio_tar and audio_tar_ready_file_path.exists():
        logger.info(f"Skip making audio tar. {audio_tar_ready_file_path} already exists.")
        return

    mkdir_if_not_exists(target_dir)
    ori_data_root = target_dir / "vs_release_16k"

    try:
        download_file(
            "https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1",
            target_dir / "vs_release_16k.zip",
            force=force_download,
        )
    except Exception as e:
        logger.error(
            f"Failed to download vocalsound: {e}. Try download it (vs_release_16k.zip) manually from https://github.com/YuanGongND/vocalsound."
        )
        return

    if not ori_data_root.exists():
        logger.info(f"Extracting {target_dir / 'vs_release_16k.zip'} to {ori_data_root}...")
        unzip_file(target_dir / "vs_release_16k.zip", ori_data_root)
    else:
        logger.info(f"Directory {ori_data_root} already exists. Skip.")

    spk_to_split = {}
    for split, csv_path in zip(["train", "valid", "test"], ["tr_meta.csv", "val_meta.csv", "te_meta.csv"]):
        spk_to_split.update(
            {row[0]: split for row in pd.read_csv(ori_data_root / f"meta/{csv_path}", header=None).values}
        )

    data = []
    for filename in os.listdir(ori_data_root / "audio_16k"):
        if (ori_data_root / "audio_16k" / filename).is_file():
            spk, *_, target = filename.strip(".wav").split("_")
            split = spk_to_split[spk]
            target = os.path.splitext(target)[0]
            data.append({"filename": filename, "split": split, "target": target})

    df = pd.DataFrame(data)
    df.filename = df.filename.apply(lambda x: (ori_data_root / "audio_16k" / x).as_posix())

    for split in df.split.unique():
        df_split = df[df.split == split].drop(columns="split")
        write_audio_tar(
            audio_paths=df_split.filename.tolist(),
            labels=df_split.target.tolist(),
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
