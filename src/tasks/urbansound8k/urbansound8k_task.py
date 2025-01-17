from xares.task_base import TaskBase, TaskConfig


class UrbanSound8KTask(TaskBase):

    def __init__(self, encoder):
        task = "soundevent"
        self.class_label_maps = {
            "air_conditioner": 0,
            "car_horn": 1,
            "children_playing": 2,
            "dog_bark": 3,
            "drilling": 4,
            "engine_idling": 5,
            "gun_shot": 6,
            "jackhammer": 7,
            "siren": 8,
            "street_music": 9,
        }
        config = TaskConfig(
            zenodo_id="TODO",
            k_fold_splits=list(range(1, 11)),
            output_dim=len(self.class_label_maps),
            label_processor=lambda x: self.class_label_maps[x[task]],
            num_training_workers=3,
            num_validation_workers=0,
        )
        super().__init__(encoder, config=config)
        self.config.audio_tar_name_of_split = {
            fold: f"urbansound_fold{fold}_0000000.tar" for fold in self.config.k_fold_splits
        }
        self.config.encoded_tar_name_of_split = {
            fold: f"urbansound-wds-encoded-fold-{fold}-*.tar" for fold in self.config.k_fold_splits
        }

    def run(self) -> float:
        return self.default_run_k_fold()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
