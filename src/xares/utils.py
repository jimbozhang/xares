import os
import tarfile
import urllib.request
import zipfile

from loguru import logger


def mkdir_if_not_exists(dir):
    if not os.path.exists(dir):
        logger.info(f"Creating directory {dir}...")
        os.makedirs(dir)


def download_file(url, filename, force=False):
    if force or not os.path.exists(filename):
        logger.info(f"Downloading {url} to {filename}...")
        urllib.request.urlretrieve(url, filename)


def unzip_file(zip_file, dest_dir):
    mkdir_if_not_exists(dest_dir)
    logger.info(f"Unzipping {zip_file} to {dest_dir}...")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(dest_dir)


def untar_file(tar_file, dest_dir):
    mkdir_if_not_exists(dest_dir)
    logger.info(f"Extracting  {tar_file} to {dest_dir}...")
    with tarfile.open(tar_file, "r|gz") as tar_ref:
        tar_ref.extractall(dest_dir)
