import os
from pathlib import Path

from loguru import logger


def mkdir_if_not_exists(dir:Path, main_process: bool = True):
    if not dir.exists() and main_process:
        logger.info(f"Creating directory {dir}...")
        dir.mkdir(parents=True, exist_ok=True)


def download_file(url, filename, force=False):
    import urllib.request

    if force or not os.path.exists(filename):
        logger.info(f"Downloading {url} to {filename}...")
        urllib.request.urlretrieve(url, filename)


def unzip_file(zip_file, dest_dir):
    import zipfile

    mkdir_if_not_exists(dest_dir)
    logger.info(f"Unzipping {zip_file} to {dest_dir}...")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(dest_dir)


def untar_file(tar_file, dest_dir):
    import tarfile

    mkdir_if_not_exists(dest_dir)
    logger.info(f"Extracting  {tar_file} to {dest_dir}...")
    with tarfile.open(tar_file, "r|gz") as tar_ref:
        tar_ref.extractall(dest_dir)


def download_zenodo_record(zenodo_id: str, target_dir: str, force_download=False):
    zenodo_archive = f"https://zenodo.org/api/records/{zenodo_id}/files-archive"
    target_path = f"{target_dir}/{zenodo_id}.zip"

    if not force_download and Path(target_path).exists():
        logger.info(f"{target_path} already exists.")
    else:
        download_file(zenodo_archive, target_path, force=force_download)

    unzip_file(f"{target_dir}/{zenodo_id}.zip", target_dir)
