from pathlib import Path

from xares.task_base import TaskBase, TaskConfig


class LibriSpeechMaleFemaleTask(TaskBase):
    def __init__(self, encoder):
        config = TaskConfig(
            zenodo_id="14641593",
            output_dim=2,
            train_split="train-clean-100",
            valid_split="dev-clean",
            test_split="test-clean",
            label_processor=lambda x: 0 if x["gender"] == "M" else 1,
        )
        super().__init__(encoder, config=config)

    def run(self) -> float:
        return self.default_run()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
