from xares.task_base import TaskBase, TaskConfig


class VocalSoundTask(TaskBase):
    def __init__(self, encoder):
        data_key = "label"
        class_label_maps = {
            "laughter": 0,
            "sigh": 1,
            "throatclearing": 2,
            "cough": 3,
            "sneeze": 4,
            "sniff": 5,
            }

        task_config = TaskConfig(
            train_split="voxceleb1_train",
            test_split="voxceleb1_test",
            valid_split="voxceleb1_valid",
            zenodo_id="14641593",
            output_dim=len(class_label_maps),
            crop_length=6, # 6s 
            label_processor=lambda x: class_label_maps[x[data_key]]

        )
        super().__init__(encoder, config=task_config)

    def run(self) -> float:
        return self.default_run()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
