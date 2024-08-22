# Copied from `xiaomitts/common/audiowebdataset.py`

import json
import random
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union  # type: ignore

import numpy as np
import torch
import torchaudio
import webdataset as wds
from loguru import logger


def db_to_linear(scalar: float) -> float:
    return 10 ** (scalar / 20)


def crop_or_pad_latent_with_audio(
    latent: torch.Tensor,
    wav: torch.Tensor,
    crop_size: int,
    conditional_crop_size: Optional[int] = None,
    pad_last: bool = False,
):
    dim, n_latents = latent.shape
    available_crops = n_latents // crop_size
    # Round to nearest power of 2
    crop_size_for_wav = (2 ** int(np.round(np.log(wav.shape[0] / n_latents) / np.log(2)))) * crop_size
    for i in range(available_crops):
        latent_crop = latent[..., i * crop_size : (i + 1) * crop_size]
        wav_crop = wav[i * crop_size_for_wav : (i + 1) * crop_size_for_wav, ...]
        if conditional_crop_size is not None:
            cond = torch.cat([wav[: i * crop_size_for_wav, ...], wav[(i + 1) * crop_size_for_wav :, ...]])
            cond, *_ = _crop_audio_for_condition(cond, conditional_crop_size)
            yield latent_crop, wav_crop, cond
        else:
            yield latent_crop, wav_crop

    if (available_crops == 0) or (pad_last):
        last_crop_latent = latent[..., available_crops * crop_size :]
        padded_latent = torch.zeros((dim, crop_size))
        padded_latent[..., : last_crop_latent.shape[-1]] = last_crop_latent

        last_crop_wav = wav[available_crops * crop_size_for_wav :, ...]
        padded_wav = torch.zeros((crop_size_for_wav, *last_crop_wav.shape[1:]))
        padded_wav[: last_crop_wav.shape[0]] = last_crop_wav
        if conditional_crop_size is not None:
            cond = padded_wav[: available_crops * crop_size_for_wav, ...]
            cond = _crop_audio_for_condition(cond, conditional_crop_size)
            yield padded_latent, padded_wav, cond
        else:
            yield padded_latent, padded_wav


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


def _crop_audio_for_condition(audio: torch.Tensor, crop_size: int):
    audio_length = audio.shape[-1]
    # Audio too short, pad zeros and return length
    if audio_length < crop_size:
        conditional_audio = torch.nn.functional.pad(audio, (0, crop_size - audio_length))
        start = 0
        end = audio_length
    # Randomly crop from the entire clip
    else:
        start = np.random.randint(0, audio.shape[-1] - crop_size + 1)
        end = start + crop_size
        conditional_audio = audio[start:end]
    return conditional_audio, start, end


def _seq_crop_audio(
    data,
    crop_size: Optional[int],
    mono: bool = True,
    drop_clipped: bool = True,
    drop_below_db: Optional[float] = None,
    conditional_crop_size: Optional[Union[float, Tuple[float, float]]] = None,
    random_gain: Optional[Tuple[int, int]] = None,
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
        if conditional_crop_size is not None:
            if isinstance(conditional_crop_size, float):
                conditional_audio, start, end = _crop_audio_for_condition(
                    audio, crop_size=int(conditional_crop_size * sr)
                )
            else:
                conditional_crop_size_sampled = np.random.uniform(*conditional_crop_size)
                conditional_audio, start, end = _crop_audio_for_condition(
                    audio, crop_size=int(conditional_crop_size_sampled * sr)
                )
            extra = (conditional_audio, torch.as_tensor([start, end]), *extra)

        for crop in crops:
            if drop_below_db is not None:
                energy = torch.sum(crop**2)
                energy_db = 10 * torch.log10(energy)
                if energy_db < drop_below_db:
                    continue
            if random_gain is not None:
                factor = db_to_linear(np.random.uniform(*random_gain))
                crop *= factor
            yield (crop, *extra)


def _seq_crop_latent(
    data,
    crop_size: int,
    conditional_crop_size: Optional[float] = None,
    random_gain: Optional[Tuple[int, int]] = None,
    drop_clipped: bool = False,
    drop_below_db: Optional[float] = None,
    handler=None,
):
    """WebDataset crop filter, yields random crops"""
    for sample in data:
        audio, latent, *extra = sample
        audio, sample_rate = audio
        if audio.ndim == 2:
            audio = audio.mean(0)
        if audio.abs().max() >= 0.99 and drop_clipped:
            continue
        if random_gain is not None:
            factor = db_to_linear(np.random.uniform(*random_gain))
            audio *= factor
        crops = crop_or_pad_latent_with_audio(
            latent.float(),
            audio.float(),
            crop_size=crop_size,
            conditional_crop_size=(
                int(sample_rate * conditional_crop_size) if conditional_crop_size is not None else None
            ),
            pad_last=False,
        )

        for crop in crops:
            if drop_below_db is not None:
                energy = torch.sum(crop[1] ** 2)
                energy_db = 10 * torch.log10(energy)
                if energy_db < drop_below_db:
                    continue
            yield (*crop, *extra)


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
        handler=wds.warn_and_continue,
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


class EmbeddingWebdataset(wds.DataPipeline):
    def __init__(
        self,
        urls,
        shuffle: Optional[int] = None,
        resample: bool = False,
        batch_size: Optional[int] = None,
    ):
        assert isinstance(urls, list)
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
            pipeline.extend([wds.split_by_node, wds.split_by_worker, wds.tarfile_to_samples()])
        pipeline.extend([wds.decode(), wds.to_tuple("pth", "json", "__key__")])
        if batch_size is not None:
            pipeline.append(wds.batched(batch_size))
        super().__init__(pipeline)


# Can also replace with wds.Randomix
class BalancedDatasetSampler(wds.DataPipeline, wds.compat.FluidInterface):

    def __init__(self, **datasets):

        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        sources = {k: iter(ds) for k, ds in self.datasets.items()}
        while True:
            ret = {}
            try:
                for k, source in sources.items():
                    ret[k] = next(source)
                yield ret
            except StopIteration:
                return


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


def collate_with_lengths_wds(samples, combine_scalars=True, flatten: bool = True, combine_tensors=True):
    batched = list(zip(*samples))
    result = []
    for b in batched:
        if isinstance(b[0], (int, float)):
            if combine_scalars:
                b = np.array(list(b))
        elif isinstance(b[0], torch.Tensor):
            if combine_tensors:
                # Added lengths
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
    balanced_samper: Optional[bool] = False,
    num_workers: int = 4,
    training: bool = False,
    random_gain: Optional[Tuple[int, int]] = None,  # Adds random gain
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
            random_gain=random_gain,
            mono=True,
            crop_size=crop_size,
            remain_random_one_crop=remain_random_one_crop,
        ),
    )
    if balanced_samper:
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


def create_latent_webdataset(
    urls: Union[List[str], Dict[str, List[str]]],
    tar_shuffle: Optional[int] = None,
    resample: Optional[bool] = None,
    batch_size: int = 16,
    drop_clipped: bool = False,
    target_sample_rate: Optional[int] = None,
    crop_size: int = 20,
    balanced_samper: Optional[bool] = False,
    num_workers: int = 4,
    training: bool = False,
    random_gain: Optional[Tuple[int, int]] = None,  # Adds random gain
    conditional_crop_size: Optional[float] = None,
    **kwargs,
):
    dataset_kwargs = dict(
        tar_shuffle=tar_shuffle,
        target_sample_rate=target_sample_rate,
        resample=resample,
        batch_size=batch_size,
        rename_keys=dict(audio="flac;mp3;sox;wav;m4a;ogg;wma", latent="pth"),
        map_kwargs=dict(latent=lambda sample: sample.transpose(-2, -1)),
        merge_function=partial(
            _seq_crop_latent,
            drop_clipped=drop_clipped,
            conditional_crop_size=conditional_crop_size,
            random_gain=random_gain,
            crop_size=crop_size,
        ),
    )
    if balanced_samper:
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


def create_gpt_webdataset(
    urls: Union[List[str], Dict[str, List[str]]],
    tar_shuffle: Optional[int] = None,
    resample: Optional[bool] = None,
    batch_size: int = 16,
    drop_clipped: Optional[bool] = False,
    target_sample_rate: Optional[int] = None,
    balanced_sampler: Optional[bool] = False,
    num_workers: int = 4,
    training: bool = False,
    max_audio_length: float = 15,
    min_audio_length: float = 3,
    max_text_length: Optional[int] = None,
    conditional_crop_size: Union[float, Tuple[float, float]] = 1.5,
    text_tokenizer: Optional[Callable] = None,
    **kwargs,
):

    def filter_lengths(sample):
        audio, sr = sample["audio"]
        audio_length = audio.shape[-1] / sr
        if audio_length > max_audio_length:
            return False
        if audio_length < min_audio_length:
            return False
        if max_text_length is not None:
            if len(sample["text"]) > max_text_length:
                return False
        return True

    def codes_transform(x):
        codes = torch.tensor(x["codes"])
        # Need to transpose for num_codebooks > 1 quantizers
        if codes.ndim == 2:
            codes = codes.transpose(-2, -1)
        return codes

    dataset_kwargs = dict(
        target_sample_rate=target_sample_rate,
        resample=resample,
        tar_shuffle=tar_shuffle,
        batch_size=batch_size,
        rename_keys=dict(
            audio="flac;mp3;sox;wav;m4a;ogg;wma",
            codes="json",
            text="json",
            filename="__key__",
        ),
        filter_function=filter_lengths,
        map_kwargs=dict(
            # audio=lambda x: x[0],
            codes=codes_transform,  # Transposing since during padding #263, we need to assert that codes are of shape (B,...,T)
            # text=text_t
            text=lambda sample: torch.as_tensor(text_tokenizer(sample["text"])).long(),
        ),
        merge_function=partial(
            _seq_crop_audio, crop_size=None, conditional_crop_size=conditional_crop_size
        ),  # Returns (audio, )
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
    ).unbatched()  # Unbatch to shuffle across workers
    if training:
        dataloader = dataloader.shuffle(512)
    dataloader = dataloader.batched(batch_size, collation_fn=collate_with_lengths_wds)
    return dataloader


def write_audio_tar(audio_paths: List[str], labels: List, tar_path: str, num_shards: int = 20, force=False):
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
                ostream.write(sample)
