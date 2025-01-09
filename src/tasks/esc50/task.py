from pathlib import Path

from xares.task_base import TaskBase


class ESC50Task(TaskBase):
    def __init__(self, encoder):
        super().__init__(encoder)

        self.config.zenodo_id = "14614287"
        self.config.k_fold_splits = range(1, 6)

        self.config.audio_tar_name_of_split = {
            fold: f"wds-audio-fold-{fold}-*.tar" for fold in self.config.k_fold_splits
        }
        self.config.encoded_tar_path_of_split = {
            fold: self.env_dir / f"wds-encoded-fold-{fold}-*.tar" for fold in self.config.k_fold_splits
        }

        self.config.trim_length = 220_500
        self.config.output_dim = 50

    def run(self) -> float:
        return self.default_run_k_fold()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()