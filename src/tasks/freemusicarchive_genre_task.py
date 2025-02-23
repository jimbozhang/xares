from xares.task import TaskConfig


def fma_genre_config(encoder) -> TaskConfig:
    data_key = "genre"
    class_label_maps = {
        "Hip-Hop": 0,
        "Pop": 1,
        "Folk": 2,
        "Experimental": 3,
        "Rock": 4,
        "International": 5,
        "Electronic": 6,
        "Instrumental": 7,
    }
    return TaskConfig(
        crop_length=10,  # 10s
        encoder=encoder,
        epochs=3,
        evalset_size=800,
        formal_name="Free Music Archive Small",
        label_processor=lambda x: class_label_maps[x[data_key]],
        name="freemusicarchive",
        output_dim=len(class_label_maps),
        test_split="fma_small_test",
        train_split="fma_small_train",
        valid_split="fma_small_valid",
        zenodo_id="14725056",
    )
