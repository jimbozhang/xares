import argparse
import os
from pathlib import Path

from loguru import logger

from xares.audiowebdataset import write_audio_tar
from xares.common import XaresSettings
from xares.utils import download_file, mkdir_if_not_exists, untar_file


def make_audio_tar(env_root: None | str = None, force_download=False, force_generate_audio_tar=True, num_shards=4):
    settings = XaresSettings()
    target_dir = Path(env_root) if env_root else Path(settings.env_root) / "librispeechmalefemale"
    audio_tar_ready_file_path = target_dir / settings.audio_ready_filename

    if not force_generate_audio_tar and audio_tar_ready_file_path.exists():
        logger.info(f"{audio_tar_ready_file_path} already exists.")
        return

    splits = ["train-clean-100", "dev-clean", "test-clean"]

    # Download and extract LibriSpeech dataset
    mkdir_if_not_exists(target_dir)
    for split in splits:
        download_file(
            f"https://www.openslr.org/resources/12/{split}.tar.gz",
            target_dir / f"{split}.tar.gz",
            force=force_download,
        )

    ori_data_root = target_dir / "LibriSpeech"

    if not ori_data_root.exists():
        for split in splits:
            logger.info(f"Extracting {target_dir / f'{split}.tar.gz'} to {target_dir}...")
            untar_file(target_dir / f"{split}.tar.gz", target_dir)
    else:
        logger.info(f"Directory {ori_data_root} already exists. Skip.")

    # Make speaker to gender dict
    with open(ori_data_root / "SPEAKERS.TXT", "r") as f:
        spk_gender = {
            int(line.split("|")[0]): line.split("|")[1].strip() for line in f if not line.strip().startswith(";")
        }

    # Make transcription dict
    id_trans = {
        line.split()[0]: " ".join(line.split()[1:])
        for split in splits
        for path, _, file_lst in os.walk(ori_data_root / split)
        for file_name in file_lst
        if file_name.endswith(".trans.txt")
        for line in open(os.path.join(path, file_name))
    }

    # Create tar file with audio files
    for split in splits:
        file_paths, targets = zip(
            *[
                (
                    Path(os.path.join(path, file_name)).as_posix(),
                    {
                        "speaker": int(str(Path(os.path.join(path, file_name)).as_posix()).split("/")[4]),
                        "gender": spk_gender[int(str(Path(os.path.join(path, file_name)).as_posix()).split("/")[4])],
                        "trans": id_trans[file_name.split(".")[0]],
                    },
                )
                for path, _, file_lst in os.walk(ori_data_root / split)
                for file_name in file_lst
                if file_name.endswith(".flac")
            ]
        )
        tar_path = target_dir / f"wds-audio-{split}-*.tar"
        write_audio_tar(
            audio_paths=file_paths,
            labels=targets,
            tar_path=tar_path.as_posix(),
            suffix="flac",
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
