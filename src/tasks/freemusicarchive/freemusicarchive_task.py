from xares.task_base import TaskBase, TaskConfig


class FreeMusicArchiveTask(TaskBase):
    def __init__(self, encoder):
        data_key = "genre"
        self.class_label_maps = {
            "Hip-Hop": 0,
            "Pop": 1,
            "Folk": 2,
            "Experimental": 3,
            "Rock": 4,
            "International": 5,
            "Electronic": 6,
            "Instrumental": 7,
        }
        task_config = TaskConfig(
            batch_size_train=64,
            learning_rate=1e-3,
            train_split="fma_small_train",
            valid_split="fma_small_valid",
            test_split="fma_small_test",
            zenodo_id="TODO",
            output_dim=len(self.class_label_maps),
            label_processor=lambda x: self.class_label_maps[x[data_key]],
            crop_length=10, #10s
        )
        super().__init__(encoder, config=task_config)

    def run(self) -> float:
        return self.default_run()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
