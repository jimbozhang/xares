import argparse
from functools import partial

import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger

from xares.audio_encoder_checker import check_audio_encoder
from xares.common import setup_global_logger
from xares.metrics import weighted_average
from xares.task import XaresTask
from xares.utils import attr_from_py_path


def worker(
    encoder_py: None | str,
    task_py: str,
    do_download: bool = False,
    do_encode: bool = False,
    do_mlp: bool = False,
    do_knn: bool = False,
):
    # Encoder setup
    encoder = attr_from_py_path(encoder_py, endswith="Encoder")() if encoder_py else None

    # Task setup
    config = attr_from_py_path(task_py, endswith="_config")(encoder)
    if config.disabled:
        logger.warning(f"Task {config.name} is disabled, skipping")
        return (0, 0), (0, 0)
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

    mlp_score = (0, 0)
    if do_mlp:
        logger.info(f"Running run_mlp for task {config.name} ...")
        mlp_score = task.run_mlp()
        logger.info(f"MLP score of {config.name}: {mlp_score}")

    knn_score = (0, 0)
    if do_knn and task.config.do_knn:
        logger.info(f"Running KNN for task {config.name} ...")
        knn_score = task.run_knn()
        logger.info(f"KNN score of {config.name}: {knn_score}")

    torch.cuda.empty_cache()
    return mlp_score, knn_score


def stage_1(encoder_py, task_py, gpu_id):
    torch.cuda.set_device(gpu_id)
    return worker(encoder_py, task_py, do_encode=True)


def stage_2(encoder_py, task_py, result: dict):
    result.update({task_py: worker(encoder_py, task_py, do_mlp=True, do_knn=True)})


def main(args):
    setup_global_logger()
    enable_multiprocessing = args.max_jobs > 0
    torch.multiprocessing.set_start_method("spawn")

    # Stage 0: Download all datasets
    stage_0 = partial(worker, do_download=True)
    if args.from_stage <= 0:
        try:
            if enable_multiprocessing:
                with mp.Pool(processes=args.max_jobs) as pool:
                    pool.starmap(stage_0, [(None, task_py) for task_py in args.tasks_py])
            else:
                for task_py in args.tasks_py:
                    stage_0(None, task_py)
            logger.info("Stage 0 completed: All data downloaded.")
        except Exception as e:
            if "Max retries exceeded with url" in str(e):
                logger.error(e)
                logger.error("This may be caused by Zenodo temporarily banning your connection.")
                logger.error("You may need to wait for a few hours and retry.")
                logger.error("Alternatively, you can download manually using `tools/download_manually.sh`.")
                return
            else:
                logger.error(f"Error in stage 0 (download): {e} Must fix it before proceeding.")
                return
    else:
        # Ensure pretrained model has been saved at local if stage 0 is skipped
        for task_py in args.tasks_py:
            worker(None, task_py)

    if args.to_stage == 0:
        return

    # Check if the encoder supports the multiprocessing
    if enable_multiprocessing:
        try:
            with mp.Pool(processes=1) as pool:
                pool.starmap(worker, [(args.encoder_py, args.tasks_py[0])])
        except Exception as e:
            logger.warning("Multiprocessing is not supported for the encoder. Falling back to a single process.")
            logger.warning("If single processing is too slow, you can manually parallelize tasks with a shell script.")
            logger.warning("For models from Hugging Face, try save locally, which might fix for multiprocessing.")
            enable_multiprocessing = False

    # Double check the encoder and download the pretrained weights
    encoder = attr_from_py_path(args.encoder_py, endswith="Encoder")()
    if not check_audio_encoder(encoder):
        raise ValueError("Invalid encoder")
    del encoder

    # Stage 1: Execute make_encoded_tar
    if args.from_stage <= 1:
        try:
            if enable_multiprocessing:
                num_gpus = torch.cuda.device_count()
                with mp.Pool(processes=args.max_jobs) as pool:
                    pool.starmap(
                        stage_1,
                        [(args.encoder_py, task_py, i % num_gpus) for i, task_py in enumerate(args.tasks_py)],
                    )
            else:
                for task_py in args.tasks_py:
                    worker(args.encoder_py, task_py, do_encode=True)

            logger.info("Stage 1 completed: All tasks encoded.")
        except FileNotFoundError as e:
            logger.error(f"Task filename pattern error: {e}")
            return
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("CUDA out of memory. Try reducing `config.batch_size_encode` of tasks.")
            else:
                logger.error(f"Error in stage 1 (encode): {e} Must fix it before proceeding.")
                return
        logger.info("Stage 1 completed: All tasks encoded.")
    if args.to_stage == 1:
        return

    # Stage 2: Execute mlp and knn scoring
    if args.from_stage <= 2 and args.to_stage >= 2:
        try:
            if enable_multiprocessing:
                manager = mp.Manager()
                return_dict = manager.dict()
                with mp.Pool(processes=args.max_jobs) as pool:
                    pool.starmap(
                        partial(stage_2, result=return_dict),
                        [(args.encoder_py, task_py) for task_py in args.tasks_py],
                    )
            else:
                return_dict = {}
                for task_py in args.tasks_py:
                    return_dict[task_py] = worker(args.encoder_py, task_py, do_mlp=True, do_knn=True)
        except RuntimeError as e:
            logger.error(f"Error in stage 2 (scoring): {e} Must fix it before proceeding.")
            return
        logger.info("Scoring completed: All tasks scored.")

        # Print results
        df = pd.DataFrame(return_dict.items(), columns=["Task", "Scores"])
        df["MLP_Score"] = df["Scores"].apply(lambda x: x[0][0])
        df["KNN_Score"] = df["Scores"].apply(lambda x: x[1][0])
        df["Task"] = df["Task"].str.replace(r".*/|_task\.py$", "", regex=True)
        df.drop(columns=["Scores"], inplace=True)
        df.sort_values(by="Task", inplace=True)

        print(f"\nResults:\n{df.to_string(index=False)}")

        avg_mlp, avg_knn = weighted_average(return_dict)
        print("\nWeighted Average MLP Score:", avg_mlp)
        print("Weighted Average KNN Score:", avg_knn)


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
