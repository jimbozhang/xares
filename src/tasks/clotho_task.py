from xares.task import TaskConfig


def data_merge_function_clotho(data_stream):
    for sample in data_stream:
        audio, captions, *extra = sample
        for caption in captions.values():
            yield audio, caption, *extra


def clotho_config(encoder) -> TaskConfig:
    task_config = TaskConfig(
        batch_size_train=128,
        criterion="AudioTextContrastiveLoss",
        crop_length=30,
        do_knn=False,
        encoder=encoder,
        epochs=20,
        merge_processor=data_merge_function_clotho,
        metric="recallatk_r1",
        name="clotho",
        pretrained_dependencies=["bert-base-uncased:tokenizer"],
        save_encoded_per_batches=500,
        task_type="contrastive",
        test_split="clotho_validation",  # Worst naming scheme in a dataset
        train_split="clotho_development",
        valid_split="clotho_evaluation",
        zenodo_id="14856454",
    )
    return task_config
