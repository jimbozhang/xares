from pathlib import Path

from loguru import logger


def mkdir_if_not_exists(dir: Path, main_process: bool = True):
    dir = Path(dir)
    if not dir.exists() and main_process:
        logger.info(f"Creating directory {dir} ...")
        dir.mkdir(parents=True, exist_ok=True)


def download_file(url, filename):
    import urllib.request

    logger.info(f"Downloading {url} to {filename} ...")
    urllib.request.urlretrieve(url, filename)


def unzip_file(zip_file, dest_dir):
    import zipfile

    mkdir_if_not_exists(dest_dir)
    logger.info(f"Unzipping {zip_file} to {dest_dir} ...")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
    logger.info(f"Unzipping completed: {zip_file} extracted to {dest_dir}.")


def untar_file(tar_file, dest_dir):
    import tarfile

    mkdir_if_not_exists(dest_dir)
    logger.info(f"Extracting  {tar_file} to {dest_dir} ...")
    with tarfile.open(tar_file, "r|gz") as tar_ref:
        tar_ref.extractall(dest_dir)
    logger.info(f"Extracting completed: {tar_file} extracted to {dest_dir}.")


def download_zenodo_record(zenodo_id: str, target_dir: str, force_download: bool = False, temp_dir: None | str = None):
    import shutil
    import tempfile

    target_zip_path = Path(target_dir) / f"{zenodo_id}.zip"
    if not force_download and target_zip_path.exists():
        logger.info(f"{target_dir}/{zenodo_id}.zip already exists, skipping download.")
    else:
        temp_zip_path = Path(tempfile.gettempdir()) / f"{zenodo_id}.zip"
        zenodo_archive_url = f"https://zenodo.org/api/records/{zenodo_id}/files-archive"
        download_file(zenodo_archive_url, temp_zip_path)
        shutil.move(temp_zip_path, target_zip_path)
        logger.info(f"Downloading completed: {zenodo_id} saved to {target_zip_path}.")

    try:
        unzipped_flag = Path(target_dir) / f".unzipped"
        if not unzipped_flag.exists():
            unzip_file(target_zip_path, target_dir)
            unzipped_flag.touch()
        else:
            logger.info(f"{target_zip_path} already unzipped, skipping unzip.")
    except Exception as e:
        err_msg = f"Failed to unzip {target_zip_path} to {target_dir}: {e}, try to remove {target_zip_path} and retry."
        logger.error(err_msg)
        raise e


def attr_from_py_path(path: str, endswith: str | None = None) -> type:
    from importlib import import_module

    module_name = path.replace("/", ".").strip(".py")

    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        raise ValueError(f"Module not found: {module_name}")

    attr_list = [m for m in dir(module) if not endswith or m.endswith(endswith)]
    if len(attr_list) != 1:
        raise ValueError(f"Expected 1 class with endswith={endswith}, got {len(attr_list)}")

    return getattr(module, attr_list[0])
