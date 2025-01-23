import argparse
from functools import partial

import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger

from xares.audio_encoder_checker import check_audio_encoder
from xares.task import XaresTask
from xares.utils import attr_from_py_path


def scoring(encoder_py: str, task_py: str, do_encode: bool = True, do_mlp: bool = True, **kwargs):
    # Encoder setup
    encoder = attr_from_py_path(encoder_py, endswith="Encoder")()
    assert check_audio_encoder(encoder), "Invalid encoder"

    # Task setup
    config = attr_from_py_path(task_py, endswith="_config")(encoder)
    task = XaresTask(config=config)

    # Run the task
    logger.info(f"Running make_encoded_tar for task {config.name}...")
    do_encode and task.make_encoded_tar()

    logger.info(f"Running run_mlp for task {config.name}...")
    mlp_score = task.run_mlp() if do_mlp else 0

    logger.info(f"MLP score of {config.name}: {mlp_score}")
    logger.info(f"Running KNN for task {config.name}...")
    knn_score = task.run_knn() if do_mlp else 0

    logger.info(f"KNN score of {config.name}: {knn_score}")
    return mlp_score


def main(args):
    task_files = args.tasks_py
    single_worker_scoring = partial(scoring, num_encoder_workers=0, num_training_workers=0, num_validation_workers=0)
    stage_1 = partial(single_worker_scoring, do_encode=True, do_mlp=False)
    stage_2 = partial(single_worker_scoring, do_encode=False, do_mlp=True)

    torch.multiprocessing.set_start_method("spawn")

    def stage_2_return_dict(encoder_py, task_py, return_dict):
        return_dict[task_py] = stage_2(encoder_py, task_py)

    # Step 1: Execute make_encoded_tar
    with mp.Pool(processes=args.max_jobs) as pool:
        pool.starmap(stage_1, [(args.encoder_py, task_py) for task_py in task_files])

    # Step 2: Execute mlp scoring
    manager = mp.Manager()
    return_dict = manager.dict()
    with mp.Pool(processes=args.max_jobs) as pool:
        pool.starmap(
            stage_2_return_dict,
            [(args.encoder_py, task_py, return_dict) for task_py in task_files],
        )

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
    args = parser.parse_args()
    main(args)
