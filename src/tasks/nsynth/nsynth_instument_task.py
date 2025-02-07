from xares.task_base import TaskBase, TaskConfig


class NSynthInstumentTask(TaskBase):
    def __init__(self, encoder):
        data_key = "instrument_family_str"
        self.class_label_maps = {
            "guitar": 0,
            "bass": 1,
            "organ": 2,
            "keyboard": 3,
            "vocal": 4,
            "string": 5,
            "reed": 6,
            "flute": 7,
            "mallet": 8,
            "brass": 9,
            "synth_lead": 10,
        }
        task_config = TaskConfig(
            batch_size_train=64,
            learning_rate=1e-3,
            train_split="nsynth_train",
            test_split="nsynth_test",
            valid_split="nsynth_valid",
            zenodo_id="TODO",
            output_dim=len(self.class_label_maps),
            epochs=50,
        )
        super().__init__(encoder, config=task_config)
        self.label_processor = lambda x: self.class_label_maps[x[data_key]]

    def run(self) -> float:
        return self.default_run()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
