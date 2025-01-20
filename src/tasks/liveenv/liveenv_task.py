from xares.task_base import TaskBase, TaskConfig


class LiveEnvTask(TaskBase):
    def __init__(self, encoder):
        data_key = "soundevent"
        self.class_label_maps = {
            "Airplane": 0,
            "AirportHall": 1,
            "Alarm": 2,
            "BabyCry": 3,
            "BabyLaugh": 4,
            "BabySound ": 5,
            "Car": 6,
            "Cat": 7,
            "DiningArea": 8,
            "Dog": 9,
            "InsideTrain": 10,
            "Keyboard": 11,
            "KnockDoor": 12,
            "Metro": 13,
            "MetroWaitingArea": 14,
            "Motorbike": 15,
            "Ocean": 16,
            "Office": 17,
            "Park": 18,
            "Poultry": 19,
            "PublicSquareDance": 20,
            "Restaurant": 21,
            "Sheep": 22,
            "ShoppingMall": 23,
            "VaccumCleaner": 24,
            "WaitingRoom": 25,
            "Washmashine": 26,
            "Water": 27,
        }
        task_config = TaskConfig(
            batch_size_train=64,
            train_split="liveenv_train",
            valid_split="liveenv_test",
            test_split="liveenv_test",
            zenodo_id="TODO",
            output_dim=len(self.class_label_maps),
            label_processor=lambda x: self.class_label_maps[x[data_key]],
        )
        super().__init__(encoder, config=task_config)

    def run(self) -> float:
        return self.default_run()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
