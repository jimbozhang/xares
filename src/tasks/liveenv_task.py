from xares.task import TaskConfig


def liveenv_config(encoder) -> TaskConfig:
    data_key = "soundevent"
    class_label_maps = {
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

    return TaskConfig(
        do_knn=False,
        disabled=True,
        encoder=encoder,
        epochs=50,
        label_processor=lambda x: class_label_maps[x[data_key]],
        name="liveenv",
        output_dim=len(class_label_maps),
        private=True,
        test_split="liveenv_test",
        train_split="liveenv_train",
        valid_split="liveenv_test",
        eval_weight=5000,
    )
