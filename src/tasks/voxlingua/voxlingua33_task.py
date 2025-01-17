from xares.task_base import TaskBase, TaskConfig


class VoxLingua33Task(TaskBase):
    def __init__(self, encoder):
        data_key = "labels"
        self.class_label_maps = {
            "uk": 0,
            "zh": 1,
            "nn": 2,
            "hr": 3,
            "nl": 4,
            "fa": 5,
            "el": 6,
            "sv": 7,
            "hy": 8,
            "pl": 9,
            "hu": 10,
            "mk": 11,
            "is": 12,
            "pt": 13,
            "sl": 14,
            "ar": 15,
            "en": 16,
            "lv": 17,
            "ru": 18,
            "az": 19,
            "no": 20,
            "lt": 21,
            "es": 22,
            "tr": 23,
            "fr": 24,
            "de": 25,
            "et": 26,
            "ur": 27,
            "da": 28,
            "fi": 29,
            "it": 30,
            "ja": 31,
            "sr": 32,
        }
        task_config = TaskConfig(
            batch_size_train=64,
            learning_rate=1e-3,
            train_split="train_subset",
            test_split="dev",
            valid_split="dev",
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
