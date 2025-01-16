from xares.task_base import TaskBase


class CremaDTask(TaskBase):
    def __init__(self, encoder):
        super().__init__(encoder)

        self.config.zenodo_id = "14646870"
        self.config.output_dim = 6

        self.label_processor = lambda x: {"H": 0, "S": 1, "A": 2, "F": 3, "D": 4, "N": 5}.get(x["label"], -1)

    def run(self) -> float:
        return self.default_run()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
