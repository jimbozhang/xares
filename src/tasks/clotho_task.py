from xares.task import TaskConfig


def data_merge_function_clotho(data_stream):
    for sample in data_stream:
        audio, captions, *extra = sample
        for caption in captions.values():
            yield audio, caption, *extra


def clotho_config(encoder) -> TaskConfig:
    task_config = TaskConfig(
        batch_size_train=128,
        encoder=encoder,
        name="clotho",
        criterion="AudioTextContrastiveLoss",
        train_split="clotho_development",
        test_split="clotho_validation",  # Worst naming scheme in a dataset
        valid_split="clotho_evaluation",
        zenodo_id="TODO",
        metric="recallatk_r1",
        task_type="contrastive",
        merge_processor=data_merge_function_clotho,
        num_training_workers=4,
        epochs=20,
        crop_length=30,
        do_knn=False,
        save_encoded_per_batches=500,
        pretrained_dependencies=["bert-base-uncased:tokenizer"],
    )
    return task_config
