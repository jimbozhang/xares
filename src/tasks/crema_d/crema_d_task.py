from pathlib import Path

from xares.task_base import TaskBase, TaskConfig


class CremaDTask(TaskBase):
    def __init__(self, encoder):
        config = TaskConfig(
            zenodo_id="14646870",
            output_dim=6,
            label_processor=lambda x: {"H": 0, "S": 1, "A": 2, "F": 3, "D": 4, "N": 5}.get(x["label"], -1),
        )
        config.env_dir = (Path(config.env_root) / "cremad",)
        super().__init__(encoder, config=config)

    def run(self) -> float:
        return self.default_run()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
