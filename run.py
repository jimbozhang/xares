import argparse
import importlib
import os

import pandas as pd
import torch.multiprocessing as mp


def make_encoded_tar(task_module, encoder):
    module = importlib.import_module(task_module)
    task_class = getattr(module, [cls for cls in dir(module) if cls.endswith("Task")][0])
    task_instance = task_class(encoder=encoder)
    task_instance.config.num_encoder_workers = 0  # Avoid daemonic processes
    task_instance.make_encoded_tar()


def run_mlp(task_module, encoder, return_dict):
    module = importlib.import_module(task_module)
    task_class = getattr(module, [cls for cls in dir(module) if cls.endswith("Task")][0])
    task_instance = task_class(encoder=encoder)
    task_instance.config.num_training_workers = 0
    task_instance.config.num_validation_workers = 0
    result = task_instance.run()
    return_dict[task_module] = result


def main(args):
    # Collect task modules
    task_modules = []
    if args.task_list:
        with open(args.task_list, "r") as f:
            task_modules = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    else:
        tasks_dir = "src/tasks"
        for root, _, files in os.walk(tasks_dir):
            for file in files:
                if file.endswith("_task.py"):
                    module_path = os.path.join(root, file).replace("/", ".").replace("\\", ".")[:-3]
                    task_modules.append(module_path)

    # Encoder setup
    encoder_module = importlib.import_module(args.encoder_module)
    encoder = getattr(encoder_module, args.encoder_class)()

    # Step 1: Execute make_encoded_tar for all tasks
    with mp.Pool(processes=args.max_jobs) as pool:
        pool.starmap(make_encoded_tar, [(task_module, encoder) for task_module in task_modules])

    # Step 2: Execute run for all tasks
    manager = mp.Manager()
    return_dict = manager.dict()
    with mp.Pool(processes=args.max_jobs) as pool:
        pool.starmap(run_mlp, [(task_module, encoder, return_dict) for task_module in task_modules])

    # Output results in a table and calculate the average value
    df = pd.DataFrame(return_dict.items(), columns=["Task", "Value"])
    df["Task"] = df["Task"].str.replace(r".*\.", "", regex=True).str.replace(r"_task$", "", regex=True)

    print(f"\nMLP result: \n{df}")
    print("\nAverage Value:", df["Value"].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tasks with a maximum concurrency limit.")
    parser.add_argument("encoder_module", type=str, help="Encoder module. eg: example.dasheng.dasheng_encoder")
    parser.add_argument("encoder_class", type=str, help="Encoder classname. eg: DashengEncoder")
    parser.add_argument("--max-jobs", type=int, default=1, help="Maximum number of concurrent tasks.")
    parser.add_argument("--task-list", type=str, help="File containing a list of task modules to execute.")
    args = parser.parse_args()
    main(args)
