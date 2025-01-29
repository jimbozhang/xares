import argparse
from functools import partial

import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger

from xares.audio_encoder_checker import check_audio_encoder
from xares.common import setup_global_logger
from xares.task import XaresTask
from xares.utils import attr_from_py_path


def worker(
    encoder_py: str,
    task_py: str,
    do_download: bool = False,
    do_encode: bool = False,
    do_mlp: bool = False,
    do_knn: bool = False,
):
    # Encoder setup
    encoder = attr_from_py_path(encoder_py, endswith="Encoder")()
    assert check_audio_encoder(encoder), "Invalid encoder"

    # Task setup
    config = attr_from_py_path(task_py, endswith="_config")(encoder)
    task = XaresTask(config=config)

    # Run the task
    if do_download:
        logger.info(f"Downloading data for task {config.name} ...")
        task.download_audio_tar()
        logger.info(f"Task {config.name} data ready.")

    if do_encode:
        logger.info(f"Running make_encoded_tar for task {config.name} ...")
        task.make_encoded_tar()
        logger.info(f"Task {config.name} encoded.")

    if config.private and not (task.encoded_tar_dir / task.config.xares_settings.encoded_ready_filename).exists():
        logger.warning(f"Task {config.name} is private and not ready, skipping.")
        do_mlp = do_knn = False

    mlp_score = 0
    if do_mlp:
        logger.info(f"Running run_mlp for task {config.name} ...")
        mlp_score = task.run_mlp()
        logger.info(f"MLP score of {config.name}: {mlp_score}")

    knn_score = 0
    if do_knn:
        logger.info(f"Running KNN for task {config.name} ...")
        knn_score = task.run_knn()
        logger.info(f"KNN score of {config.name}: {knn_score}")

    return mlp_score, knn_score


def main(args):
    setup_global_logger()
    torch.multiprocessing.set_start_method("spawn")

    # Stage 0: Download all datasets
    stage_0 = partial(worker, do_download=True)
    if args.from_stage <= 0:
        try:
            with mp.Pool(processes=args.max_jobs) as pool:
                pool.starmap(stage_0, [(args.encoder_py, task_py) for task_py in args.tasks_py])
            logger.info("Stage 0 completed: All data downloaded.")
        except Exception as e:
            logger.error(f"Error in stage 0 (download): {e}. Must fix it before proceeding.")
            raise e

    # Stage 1: Execute make_encoded_tar
    stage_1 = partial(worker, do_encode=True)
    if args.from_stage <= 1:
        try:
            with mp.Pool(processes=args.max_jobs) as pool:
                pool.starmap(stage_1, [(args.encoder_py, task_py) for task_py in args.tasks_py])
            logger.info("Stage 1 completed: All tasks encoded.")
        except RuntimeError as e:
            logger.error(f"Error in stage 1 (encode): {e}. Must fix it before proceeding.")
            raise e

    # Stage 2: Execute mlp and knn scoring
    stage_2 = lambda encoder_py, task_py, result: result.update(
        {task_py: worker(encoder_py, task_py, do_mlp=True, do_knn=True)}
    )
    if args.from_stage <= 2:
        manager = mp.Manager()
        return_dict = manager.dict()
        with mp.Pool(processes=args.max_jobs) as pool:
            pool.starmap(
                stage_2,
                [(args.encoder_py, task_py, return_dict) for task_py in args.tasks_py],
            )
        logger.info("Stage 2 completed: All tasks scored.")

    # Output results in a table and calculate the average value
    df = pd.DataFrame(return_dict.items(), columns=["Task", "Value"])
    df["Task"] = df["Task"].str.replace(r".*\.", "", regex=True).str.replace(r"_task$", "", regex=True)

    print(f"\nMLP result: \n{df}")
    print("\nAverage Value:", df["Value"].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a task")
    parser.add_argument("encoder_py", type=str, help="Encoder path. eg: example/dasheng/dasheng_encoder.py")
    parser.add_argument(
        "tasks_py",
        type=str,
        help="Tasks path. eg: src/tasks/*.py",
        nargs="+",
    )
    parser.add_argument("--max-jobs", type=int, default=1, help="Maximum number of concurrent tasks.")
    parser.add_argument("--from-stage", default=0, type=int, help="First stage to run.")
    args = parser.parse_args()
    main(args)
