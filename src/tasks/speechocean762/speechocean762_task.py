from xares.task_base import TaskBase, TaskConfig


class Speechocean762Task(TaskBase):
    def __init__(self, encoder):
        data_key = "accuracy"
        task_config = TaskConfig(
            batch_size_train=64,
            train_split="speechocean762_train",
            valid_split="speechocean762_test",
            test_split="speechocean762_test",
            zenodo_id="TODO",
            output_dim=1,
            metric="MSE",
            criterion="MSELoss",
            epochs=20,
            batch_size_encode=1,  # Just avoid padding for this task
            label_processor=lambda x: float(x[data_key]),
        )
        super().__init__(encoder, config=task_config)

    def run(self):
        return self.default_run()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
