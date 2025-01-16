from pathlib import Path

from xares.task_base import TaskBase, TaskConfig


class SpeechCommandsV1Task(TaskBase):
    def __init__(self, encoder):
        self.class_label_maps = {
            "sheila": 0,
            "up": 1,
            "bird": 2,
            "right": 3,
            "left": 4,
            "six": 5,
            "yes": 6,
            "on": 7,
            "one": 8,
            "off": 9,
            "zero": 10,
            "marvin": 11,
            "seven": 12,
            "wow": 13,
            "five": 14,
            "down": 15,
            "three": 16,
            "nine": 17,
            "no": 18,
            "happy": 19,
            "cat": 20,
            "go": 21,
            "bed": 22,
            "house": 23,
            "stop": 24,
            "four": 25,
            "tree": 26,
            "dog": 27,
            "two": 28,
            "eight": 29,
        }
        task_config = TaskConfig(
            batch_size_train=64,
            learning_rate=1e-3,
            train_split="wds-audio-train",
            valid_split="wds-audio-valid",
            test_split="wds-audio-test",
            zenodo_id="TODO",
            output_dim=len(self.class_label_maps),
            label_processor=lambda x: self.class_label_maps[x[data_key]],
            epochs=50,
        )
        super().__init__(encoder, config=task_config)
        data_key = "labels"

    def run(self) -> float:
        return self.default_run()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
