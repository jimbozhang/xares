from pathlib import Path

from xares.task_base import TaskBase


class LibriSpeechMaleFemaleTask(TaskBase):
    def __init__(self, encoder):
        super().__init__(encoder)

        self.env_dir = Path(self.config.env_root) / "librispeech_clean_100"

        self.config.zenodo_id = "14641593"
        self.config.output_dim = 2

        self.config.train_split = "train-clean-100"
        self.config.valid_split = "dev-clean"
        self.config.test_split = "test-clean"
        self.config.update_tar_name_of_split()

        self.label_processor = lambda x: 0 if x["gender"] == "M" else 1

    def run(self) -> float:
        return self.default_run()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
