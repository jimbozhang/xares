# xares

X-ARES: eXtensive Audio Representation and Evaluation Suite

**This project is still in the early stage of development and welcomes contributions, especially in expanding supported tasks and improving robustness.**

## Introduction

X-ARES is a toolkit for training, evaluating, and exporting audio encoders for various audio tasks. It is heavily inspired by the [HEAR benchmark](https://hearbenchmark.com/).

## Planned supported tasks

Currently, only four tasks (ESC-50, VocalSound, CREMA-D and LibriSpeech-Male-Female) have been implemented. However, adding new tasks is straightforward: it typically only need to implement the audio data download and packaging functions, as manual audio decoding and training implementations are not necessary. We encourage anyone to submit pull requests to expand the set of supported tasks.

### Speech

- Speech Commands V1
- Speech Commands V2
- LibriCount
- VoxLingua107
- VoxCeleb1
- LibriSpeech-Male-Female
- LibriSpeech-Phoneme
- Fluent Speech Commands
- VocalSound
- CREMA-D
- RAVDESS
- ASV2015
- DiCOVA
- speechocean762

### Environment

- ESC-50
- FSD50k
- UrbanSound 8k
- DESED
- (A task designed for car)
- (Another task designed for car)
- (A task designed for soundbox)
- (A task designed for headphone)
- LITIS Rouen
- FSD18-Kaggle
- AudioCaps

### Music

- MAESTRO
- GTZAN Genre
- NSynth
- MTG-Jamendo
- FMA

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

The ESC-50 task is used as an example.

```python
from example.dasheng.dasheng_encoder import DashengEncoder
from tasks.esc50.esc50_task import ESC50Task

task = ESC50Task(encoder=DashengEncoder())

score = task.run()
```

## Run with your own pretrained audio encoder

An example of audio encoder wrapper could be found at `example/dasheng/dasheng_encoder.py` and `example/wav2vec2/wav2vec2.py`.

We provide a check function to verify if the encoder is correctly implemented:

```python
>>> from xares.audio_encoder_checker import check_audio_encoder
>>> encoder = DashengEncoder()
>>> check_audio_encoder(encoder)
True
```

## Add your own task

To add a new task, refer to existing task implementations for guidance. Essentially, create a task class that inherits from `TaskBase` and implements the `make_encoded_tar()` and the `run()` tailored to your chosen dataset.
