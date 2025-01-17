from xares.task_base import TaskBase, TaskConfig


class LibriCountTask(TaskBase):
    def __init__(self, encoder):
        task = "num_speakers"
        self.class_label_maps = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}
        config = TaskConfig(
            zenodo_id="TODO",
            learning_rate=1e-3,
            batch_size_train=64,
            k_fold_splits=list(range(0, 5)),
            output_dim=len(self.class_label_maps),
            label_processor=lambda x: self.class_label_maps[x[task]],
        )
        super().__init__(encoder, config=config)

        self.config.audio_tar_name_of_split = {
            fold: f"libricount_fold{fold:02d}.tar" for fold in self.config.k_fold_splits
        }
        self.config.encoded_tar_name_of_split = {
            fold: f"libricount-wds-encoded-fold-{fold}-*.tar" for fold in self.config.k_fold_splits
        }

    def run(self) -> float:
        return self.default_run_k_fold()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
