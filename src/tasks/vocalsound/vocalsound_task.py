from xares.task_base import TaskBase


class VocalSoundTask(TaskBase):
    def __init__(self, encoder):
        super().__init__(encoder)

        self.config.zenodo_id = "14641593"
        self.config.output_dim = 6

        self.label_processor = lambda x: {
            "laughter": 0,
            "sigh": 1,
            "throatclearing": 2,
            "cough": 3,
            "sneeze": 4,
            "sniff": 5,
        }.get(x["label"], -1)

    def run(self) -> float:
        return self.default_run()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
