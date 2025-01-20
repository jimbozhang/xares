from xares.task_base import TaskBase, TaskConfig


class ESC50Task(TaskBase):

    def __init__(self, encoder):
        config = TaskConfig(
            zenodo_id="14614287", k_fold_splits=list(range(1, 6)), output_dim=50, label_processor=lambda x: x["label"], 
        )
        super().__init__(encoder, config=config)

        self.config.audio_tar_name_of_split = {
            fold: f"wds-audio-fold-{fold}-*.tar" for fold in self.config.k_fold_splits
        }
        self.config.encoded_tar_name_of_split = {
            fold: f"wds-encoded-fold-{fold}-*.tar" for fold in self.config.k_fold_splits
        }

    def run(self):
        return self.default_run_k_fold()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
