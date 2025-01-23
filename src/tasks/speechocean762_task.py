from xares.task import TaskConfig


def speechocean762_config(encoder, **kwargs) -> TaskConfig:
    data_key = "accuracy"
    task_config = TaskConfig(
        name="speechocean762",
        train_split="speechocean762_train",
        valid_split="speechocean762_test",
        test_split="speechocean762_test",
        zenodo_id="TODO",
        output_dim=1,
        metric="MSE",
        criterion="MSELoss",
        batch_size_encode=1,  # Just avoid padding for this task
        label_processor=lambda x: float(x[data_key]),
        epochs=25,
        **kwargs,
    )
    return task_config
