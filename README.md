# X-ARES: eXtensive Audio Representation and Evaluation Suite

## Introduction

X-ARES is a toolkit for training, evaluating, and exporting audio encoders for various audio tasks. It is heavily inspired by the [HEAR benchmark](https://hearbenchmark.com/).

## Supported tasks

### Speech

- [x] Speech Commands V2
- [x] LibriCount
- [x] VoxLingua107
- [x] VoxCeleb1
- [x] LibriSpeech-Male-Female
- [x] Fluent Speech Commands
- [x] VocalSound
- [x] CREMA-D
- [x] RAVDESS
- [ ] LibriSpeech-Phoneme
- [x] speechocean762
- [x] ASV2015

### Environment

- [x] ESC-50
- [ ] FSD50k
- [x] UrbanSound 8k
- [ ] DESED
- [x] FSD18-Kaggle
- [x] Clotho

### Music

- [ ] MAESTRO
- [x] GTZAN Genre
- [x] NSynth
- [x] FMA

## Installation

X-ARES is available on [PyPI](https://pypi.org/project/xares/). You can install it via pip.

```bash
pip install xares
```

For development, you can clone the repository and install the package in editable mode.

```bash
git clone <this-repo>
cd xares
pip install -e .[examples]
```

## Run with the baseline pretrained audio encoder (Dasheng)

You can run the benchmark with the baseline pretrained audio encoder (Dasheng) with 8 parallel jobs using the following command:

```bash
python -m xares.run --max-jobs 8 example/dasheng/dasheng_encoder.py src/tasks/*.py
```

It will download the datasets from [Zenodo](https://zenodo.org/communities/mispeech/records), and then evaluate the encoder on all the tasks.
If the automatic download fails, you can also manually download the datasets using `tools/download_manually.sh`.

Alternatively, you can run tasks from within Python. Here is an example of running the ASVspoof2015 task in a single process:

```python
>>> from example.dasheng.dasheng_encoder import DashengEncoder
>>> from tasks.asvspoof_task import asvspoof2015_config
>>> from xares.task import XaresTask

>>> task = XaresTask(encoder=DashengEncoder(), config=asvspoof2015_config())
>>> task.run()
```

## Run with your own pretrained audio encoder

Two examples of audio encoder wrapper could be found at `example/dasheng/dasheng_encoder.py` and `example/wav2vec2/wav2vec2.py`.

We provide a check function to verify if the encoder is correctly implemented:

```python
>>> from xares.audio_encoder_checker import check_audio_encoder

>>> encoder = YourEncoder()
>>> check_audio_encoder(encoder)
True
```

And then you can run the benchmark with your own encoder:

```bash
python -m xares.run --max-jobs 8 your_encoder.py src/tasks/*.py
```

## Add new tasks

Adding a new task is easy. Refer to the existing task implementations for guidance.
You need to create a `TaskConfig` tailored to your chosen dataset.
