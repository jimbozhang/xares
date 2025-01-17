from xares.task_base import TaskBase, TaskConfig


class RavdessTask(TaskBase):
    def __init__(self, encoder):
        task = "emotion"
        self.class_label_maps = {
            "neutral": 0,
            "calm": 1,
            "happy": 2,
            "sad": 3,
            "angry": 4,
            "fearful": 5,
            "disgust": 6,
            "surprised": 7,
        }
        config = TaskConfig(
            zenodo_id="TODO",
            k_fold_splits=list(range(0, 4)),
            output_dim=len(self.class_label_maps),
            label_processor=lambda x: self.class_label_maps[x[task]],
        )
        super().__init__(encoder, config=config)

        self.config.audio_tar_name_of_split = {
            fold: f"ravdess_fold_{fold}_0000000.tar" for fold in self.config.k_fold_splits
        }
        self.config.encoded_tar_name_of_split = {
            fold: f"ravdess-wds-encoded-fold-{fold}-*.tar" for fold in self.config.k_fold_splits
        }

    def run(self) -> float:
        return self.default_run_k_fold()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
