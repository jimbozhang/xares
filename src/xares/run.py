import argparse

from xares.audio_encoder_checker import check_audio_encoder
from xares.utils import py_path_to_class


def main(args):
    # Encoder setup
    encoder = py_path_to_class(args.encoder_py, endswith="Encoder")()
    assert check_audio_encoder(encoder), "Invalid encoder"

    # Task setup
    task = py_path_to_class(args.task_py, endswith="Task")(encoder=encoder)

    # Run the task
    task.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a task")
    parser.add_argument("encoder_py", type=str, help="Encoder path. eg: example/dasheng/dasheng_encoder.py")
    parser.add_argument("task_py", type=str, help="Task path. eg: src/tasks/esc50/esc50_task.py")
    args = parser.parse_args()
    main(args)
