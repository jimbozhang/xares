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


def stage_2(encoder_py, task_py, result: dict):
    result.update({task_py: worker(encoder_py, task_py, do_mlp=True, do_knn=True)})


def main(args):
    setup_global_logger()
    torch.multiprocessing.set_start_method("spawn")

    # Check Encoder and download the pretrained weights
    encoder = attr_from_py_path(args.encoder_py, endswith="Encoder")()
    if not check_audio_encoder(encoder):
        raise ValueError("Invalid encoder")
    del encoder

    # Stage 0: Download all datasets
    stage_0 = partial(worker, do_download=True)
    if args.from_stage <= 0:
        try:
            with mp.Pool(processes=args.max_jobs) as pool:
                pool.starmap(stage_0, [(args.encoder_py, task_py) for task_py in args.tasks_py])
            logger.info("Stage 0 completed: All data downloaded.")
        except Exception as e:
            if "Max retries exceeded with url" in str(e):
                logger.error(e)
                logger.error("This may be caused by Zenodo temporarily banning your connection.")
                logger.error("You may need to wait for a few hours and retry.")
                logger.error("Alternatively, you can download manually using `tools/download_manually.sh`.")
                return
            else:
                logger.error(f"Error in stage 0 (encode): {e}. Must fix it before proceeding.")
                raise e

    # Stage 1: Execute make_encoded_tar
    stage_1 = partial(worker, do_encode=True)
    if args.from_stage <= 1 and args.to_stage >= 1:
        try:
            with mp.Pool(processes=args.max_jobs) as pool:
                pool.starmap(stage_1, [(args.encoder_py, task_py) for task_py in args.tasks_py])
            logger.info("Stage 1 completed: All tasks encoded.")
        except RuntimeError as e:
            logger.error(f"Error in stage 1 (encode): {e}. Must fix it before proceeding.")
            raise e

    # Stage 2: Execute mlp and knn scoring
    if args.from_stage <= 2 and args.to_stage >= 2:
        manager = mp.Manager()
        return_dict = manager.dict()
        with mp.Pool(processes=args.max_jobs) as pool:
            pool.starmap(
                partial(stage_2, result=return_dict),
                [(args.encoder_py, task_py) for task_py in args.tasks_py],
            )
        logger.info("Scoring completed: All tasks scored.")

        # Print results
        df = pd.DataFrame(return_dict.items(), columns=["Task", "Scores"])
        df["MLP_Score"] = df["Scores"].apply(lambda x: x[0])
        df["KNN_Score"] = df["Scores"].apply(lambda x: x[1])
        df["Task"] = df["Task"].str.replace(r".*/", "", regex=True).str.strip("_task.py")
        df.drop(columns=["Scores"], inplace=True)
        df.sort_values(by="Task", inplace=True)

        print(f"\nResults: \n{df}")
        print("\nAverage MLP Score:", df["MLP_Score"].mean())
        print("Average KNN Score:", df["KNN_Score"].mean())


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
    parser.add_argument("--to-stage", default=2, type=int, help="Last stage to run.")
    args = parser.parse_args()
    main(args)
