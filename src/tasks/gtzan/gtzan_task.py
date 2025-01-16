from xares.task_base import TaskBase



class GTZAN_GenreTask(TaskBase):
    def __init__(self, encoder):
        super().__init__(encoder)
        task = "genre"
        self.class_label_maps = {
                "blues": 0,
                "classical": 1,
                "country": 2,
                "disco": 3,
                "hiphop": 4,
                "jazz": 5,
                "metal": 6,
                "pop": 7,
                "reggae": 8,
                "rock": 9
        }


        self.config.zenodo_id = "TODO"
        self.config.k_fold_splits = list(range(0, 10))

        self.config.audio_tar_name_of_split = {
            fold: f"gtzan_fold_{fold}_0000000.tar" for fold in self.config.k_fold_splits
        }
        self.config.encoded_tar_name_of_split = {
            fold: f"wds-encoded-fold-{fold}-*.tar" for fold in self.config.k_fold_splits
        }

        self.label_processor = lambda x: self.class_label_maps[x[task]]
        self.config.output_dim = len(self.class_label_maps)

    def run(self) -> float:
        return self.default_run_k_fold()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
