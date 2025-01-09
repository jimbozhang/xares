# Reduced from `xiaomitts/common/audiowebdataset.py`

import json
import multiprocessing
import random
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union  # type: ignore

import numpy as np
import torch
import torchaudio
import webdataset as wds
import warnings
from loguru import logger


def fast_warn_and_continue(exn):
    warnings.warn(repr(exn))
    return True

def crop_or_pad_audio(wav: torch.Tensor, crop_size: int, pad_last: bool = False):
    n_samples, *_ = wav.shape
    available_crops = n_samples // crop_size
    for i in range(available_crops):
        crop = wav[i * crop_size : (i + 1) * crop_size, ...]
        yield crop

    if (available_crops == 0) or (pad_last):
        last_crop = wav[available_crops * crop_size :, ...]
        padded = torch.zeros((crop_size, *last_crop.shape[1:]))
        padded[: last_crop.shape[0]] = last_crop
        yield padded


def _seq_crop_audio(
    data,
    crop_size: Optional[int],
    mono: bool = True,
    drop_clipped: bool = True,
    remain_random_one_crop: bool = False,
    handler=None,
):
    """WebDataset crop filter, yields sequential crops"""
    for sample in data:
        audio, *extra = sample
        audio, sr = audio
        if mono and audio.ndim == 2:
            audio = audio.mean(0)
        if audio.abs().max() >= 0.99 and drop_clipped:
            continue
        if crop_size is not None:
            crops = crop_or_pad_audio(audio.float(), crop_size=crop_size, pad_last=False)
            if remain_random_one_crop:
                crops = list(crops)
                crops = [crops[random.randint(0, len(crops) - 1)]]
        else:
            crops = [audio.float()]

        for crop in crops:
            yield (crop, *extra)


class Audiowebdataset(wds.DataPipeline):

    def __init__(
        self,
        urls,
        tar_shuffle: Optional[int] = None,
        resample: bool = False,
        target_sample_rate: Optional[int] = None,
        batch_size: Optional[int] = None,
        filter_function: Optional[Callable] = None,
        rename_keys: Dict[str, str] = dict(audio="flac;mp3;sox;wav;m4a;ogg;wma", filename="__key__"),
        map_kwargs: Optional[Dict[str, Callable]] = None,
        merge_function: Optional[
            Callable
        ] = None,  # merge function is called before batching. In the merge function we can operate on the data in form of a tuple
        handler=fast_warn_and_continue,
    ):
        pipeline: List = [wds.ResampledShards(urls) if resample else wds.SimpleShardList(urls)]

        if tar_shuffle is not None:
            # Tar wise shuffle
            pipeline.extend(
                [
                    wds.detshuffle(
                        bufsize=tar_shuffle,
                        initial=tar_shuffle // 4,
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                    # at this point, we have an iterator over the shards assigned to each worker at each node
                    wds.tarfile_to_samples(handler=handler),
                    wds.shuffle(
                        bufsize=tar_shuffle,
                        initial=tar_shuffle // 4,
                    ),
                ]
            )
        else:
            pipeline.extend([wds.split_by_node, wds.split_by_worker, wds.tarfile_to_samples(handler=handler)])

        # Decode i.e., bytes object to a python-accessible obj.
        pipeline.extend([wds.decode(wds.torch_audio, handler=handler), wds.rename(**rename_keys, handler=handler)])

        if map_kwargs:
            pipeline.extend([wds.map_dict(**map_kwargs)])
        # Filter function takes a sample (key: value) as input and returns True for valid samples, otherwise false
        if filter_function:
            pipeline.extend([wds.select(filter_function)])

        # Resample audio, useful when dataset is not monotonous in sampling rate
        if target_sample_rate:
            assert "audio" in rename_keys.keys(), "target_sample_rate requires key_maps=dict(audio='flac;mp3;wav')"

            def resample_audio(audio_sr: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
                audio, sr = audio_sr
                audio = torchaudio.functional.resample(audio, sr, target_sample_rate)
                return (audio, target_sample_rate)

            pipeline.extend([wds.map_dict(audio=resample_audio)])

        # Webdataset support batching and parallel reading using
        # num_workers only with tuples, not dicts
        pipeline.extend(
            [
                wds.to_tuple(*rename_keys.keys()),
            ]
        )

        if merge_function is not None:
            pipeline.extend([merge_function])

        if batch_size is not None:
            pipeline.append(
                wds.batched(batch_size, collation_fn=partial(wds.filters.default_collation_fn, combine_tensors=False))
            )
        super().__init__(pipeline)


# Can also replace with wds.Randomix
class BalancedDatasetSampler(wds.DataPipeline, wds.compat.FluidInterface):

    def __init__(self, **datasets):

        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        sources = {k: iter(ds) for k, ds in self.datasets.items()}
        while True:
            for k, source in sources.items():
                try:
                    yield next(source)
                except StopIteration:
                    break

def expand_with_brace(lists: List[str]):
    import braceexpand
    r = []
    for l in lists:
        if "*" in l:
            # Expand using "posix" based *
            r.extend([str(f) for f in Path(l).parent.glob(Path(l).name)])
        else:
            r.extend(braceexpand.braceexpand(l))
    return r


def pad(tensorlist: Sequence[torch.Tensor], padding_value: float = 0.0):
    # Tensors are expected to be B, ..., T
    lengths = [f.shape[-1] for f in tensorlist]
    dims = tensorlist[0].shape
    trailing_dims = dims[:-1]
    batch_dim = len(lengths)
    num_raw_samples = max(lengths)
    out_dims = (batch_dim,) + trailing_dims + (num_raw_samples,)
    out_tensor = torch.full(out_dims, fill_value=padding_value, dtype=tensorlist[0].dtype)
    for i, tensor in enumerate(tensorlist):
        length = tensor.shape[-1]
        out_tensor[i, ..., :length] = tensor[..., :length]
    return out_tensor, torch.as_tensor(lengths)


def collate_with_lengths_wds(
        samples: List[Iterable], combine_scalars:bool=True, flatten: bool = True, combine_tensors:bool=True
):
    batched = list(zip(*samples))
    result = []
    for b in batched:
        if isinstance(b[0], (int, float)):
            if combine_scalars:
                b = np.array(list(b))
        elif isinstance(b[0], torch.Tensor):
            if combine_tensors:
                b = pad(list(b))
        elif isinstance(b[0], np.ndarray):
            if combine_tensors:
                b = np.array(list(b))
        else:
            b = list(b)
        # Do not flatten lists, i.e., some filenames
        if flatten and not isinstance(b, list):
            result.extend(b)
        else:
            result.append(b)
    return result


def create_rawaudio_webdataset(
    urls: Union[List[str], Dict[str, List[str]]],
    tar_shuffle: Optional[int] = None,
    resample: Optional[bool] = None,
    batch_size: int = 16,
    drop_clipped: bool = False,
    target_sample_rate: Optional[int] = None,
    crop_size: Optional[int] = None,
    with_json: bool = False,
    balanced_sampler: Optional[bool] = False,
    num_workers: int = 4,
    training: bool = False,
    remain_random_one_crop: bool = False,
    **kwargs,
):
    dataset_kwargs = dict(
        tar_shuffle=tar_shuffle,
        target_sample_rate=target_sample_rate,
        resample=resample,
        batch_size=batch_size,
        rename_keys=(
            dict(audio="flac;mp3;sox;wav;m4a;ogg;wma", json="json", filename="__key__")
            if with_json
            else dict(audio="flac;mp3;sox;wav;m4a;ogg;wma", filename="__key__")
        ),
        merge_function=partial(
            _seq_crop_audio,
            drop_clipped=drop_clipped,
            mono=True,
            crop_size=crop_size,
            remain_random_one_crop=remain_random_one_crop,
        ),
    )
    if balanced_sampler:
        assert isinstance(urls, dict)
        ds = {k: Audiowebdataset(expand_with_brace(train_data), **dataset_kwargs) for k, train_data in urls.items()}
        dataset = BalancedDatasetSampler(**ds)
    else:
        assert isinstance(urls, list)
        dataset = Audiowebdataset(expand_with_brace(urls), **dataset_kwargs)
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
    ).unbatched()
    if training:
        dataloader = dataloader.shuffle(512)
    dataloader = dataloader.batched(
        batch_size,
        collation_fn=collate_with_lengths_wds,
    )
    return dataloader


def create_embedding_webdataset(
    urls: Union[List[str], Dict[str, List[str]]],
    tar_shuffle: Optional[int] = None,
    batch_size: int = 16,
    balanced_sampler: Optional[bool] = False,
    num_workers: int = 4,
    training: bool = False,
    **kwargs,
):
    dataset_kwargs = dict(
        tar_shuffle=tar_shuffle,
        batch_size=batch_size,
        rename_keys=(
            dict(embedding="pth", json="json", filename="__key__")
        ),
        map_kwargs = dict(embedding=lambda x: x.transpose(-2,-1)) #Transpose (B,T,D) -> (B,D,T)
    )
    if balanced_sampler:
        assert isinstance(urls, dict)
        ds = {k: Audiowebdataset(expand_with_brace(train_data), **dataset_kwargs) for k, train_data in urls.items()}
        dataset = BalancedDatasetSampler(**ds)
    else:
        assert isinstance(urls, list)
        dataset = Audiowebdataset(expand_with_brace(urls), **dataset_kwargs)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=num_workers > 1,
        pin_memory=True,
    ).unbatched()
    if training:
        dataloader = dataloader.shuffle(512)
    dataloader = dataloader.batched(
        batch_size,
        collation_fn=partial(collate_with_lengths_wds, flatten=False),
    )
    return dataloader


def write_audio_tar(
    audio_paths: List[str], labels: List, tar_path: str, suffix: str = "wav", num_shards: int = 20, force: bool = False
):
    assert len(audio_paths) == len(labels), "Number of audio files and labels must match."

    assert len(audio_paths) >= num_shards, "Number of shards must be less than number of audio files."
    shard_size = (len(audio_paths) + num_shards - 1) // num_shards

    def make_sample(filename, label=None):
        with open(filename, "rb") as buf:
            raw_data = buf.read()
        fpath = Path(filename)
        stem_name = str(fpath.stem).replace(".", "_")
        suffix = fpath.suffix.replace(".", "")
        ret_data = {
            suffix: raw_data,
            "__key__": f"{stem_name}",  # Just cast to str
        }
        # If we have some labels, also dump a .json file
        if label is not None:
            ret_data["json"] = json.dumps({"label": label}).encode("utf-8")
        return ret_data

    for shard in range(num_shards):
        start_index = shard * shard_size
        end_index = start_index + shard_size

        shard_audio_paths = audio_paths[start_index:end_index]
        shard_labels = labels[start_index:end_index]

        sharded_tar_path = tar_path.replace("*", f"0{shard:05d}")
        if not force and Path(sharded_tar_path).exists():
            logger.info(f"Tar file {sharded_tar_path} already exists.")
            continue

        with wds.TarWriter(sharded_tar_path) as ostream:
            for audio_path, label in zip(shard_audio_paths, shard_labels):
                sample = make_sample(audio_path, label)
                if len(sample[suffix]) < 100:
                    logger.warning(f"Skipping {audio_path} due to short length.")
                    continue
                ostream.write(sample)


def batch_to_sample(example, label, key):
    sample = {
        "pth": example,
        "json": json.dumps({"target": label["label"]}).encode("utf-8"),
        "__key__": key,
    }
    return sample


def write_encoded_batches_to_wds(encoded_batches: List, tar_path: str, num_workers: int = 0):
    with wds.TarWriter(tar_path) as ostream:
        if num_workers == 0:
            for batch, label, keys in encoded_batches:
                for return_values in map(batch_to_sample, batch, label, keys):
                    ostream.write(return_values)
        else:
            with multiprocessing.Pool(processes=num_workers) as pool:
                for batch, label, keys in encoded_batches:
                    for return_values in pool.starmap(batch_to_sample, zip(batch, label, keys)):
                        ostream.write(return_values)
