from typing import List, Optional

import webdataset as wds


class EmbeddingWebdataset(wds.DataPipeline):
    def __init__(
        self,
        urls,
        shuffle: Optional[int] = None,
        resample: bool = False,
        batch_size: Optional[int] = None,
    ):
        pipeline: List = [wds.SimpleShardList(urls) if resample is False else wds.ResampledShards(urls)]
        if shuffle is not None:
            # Tar wise shuffle
            pipeline.extend(
                [
                    wds.detshuffle(
                        bufsize=shuffle,
                        initial=shuffle // 4,
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                    # at this point, we have an iterator over the shards assigned to each worker at each node
                    wds.tarfile_to_samples(handler=wds.warn_and_continue),
                    wds.shuffle(
                        bufsize=shuffle,
                        initial=shuffle // 4,
                    ),
                ]
            )
        else:
            pipeline.extend([wds.split_by_worker, wds.tarfile_to_samples()])
        pipeline.extend([wds.decode(), wds.to_tuple("pth", "json", "__key__")])
        if batch_size is not None:
            pipeline.append(wds.batched(batch_size))
        super().__init__(pipeline)
