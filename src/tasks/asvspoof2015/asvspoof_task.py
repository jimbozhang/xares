from xares.task_base import TaskBase, TaskConfig


class ASVSpoof2015Task(TaskBase):
    def __init__(self, encoder):
        data_key = "binary_spoof"
        self.class_label_maps = {"human": 0, "spoof": 1}
        task_config = TaskConfig(
            batch_size_train=64,
            learning_rate=1e-3,
            train_split="asvspoof_train",
            valid_split="asvspoof_valid",
            test_split="asvspoof_eval",
            zenodo_id="TODO",
            output_dim=len(self.class_label_maps),
            label_processor=lambda x: self.class_label_maps[x[data_key]],
            epochs=50,
        )
        super().__init__(encoder, config=task_config)

    def run(self) -> float:
        return self.default_run()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
