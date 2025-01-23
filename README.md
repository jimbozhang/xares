# xares

X-ARES: eXtensive Audio Representation and Evaluation Suite

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
- [ ] speechocean762
- [x] ASV2015

### Environment

- [x] ESC-50
- [ ] FSD50k
- [x] UrbanSound 8k
- [ ] DESED
- [ ] FSD18-Kaggle
- [x] Clotho

### Music

- [ ] MAESTRO
- [x] GTZAN Genre
- [ ] NSynth
- [ ] FMA

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

```bash
python -m xares.run --max-jobs 8 example/dasheng/dasheng_encoder.py "src/tasks/*.py"
```

Or from inside python:

```python
from xares.task import XaresTask
from example.dasheng.dasheng_encoder import DashengEncoder
from tasks.asvspoof_task import asvspoof2015_config
task = XaresTask(encoder=DashengEncoder(), config=asvspoof2015_config())
task.run()
```



## Run with your own pretrained audio encoder

An example of audio encoder wrapper could be found at `example/dasheng/dasheng_encoder.py` and `example/wav2vec2/wav2vec2.py`.

We provide a check function to verify if the encoder is correctly implemented:

```python
>>> from xares.audio_encoder_checker import check_audio_encoder

>>> encoder = YourEncoder()
>>> check_audio_encoder(encoder)
True
```

And then you can run the benchmark with your own encoder:

```bash
python -m xares.run --max-jobs 8 your_encoder.py "src/tasks/*.py"
```

## Add your own task

To add a new task, refer to the existing task implementations for guidance. You need to create a TaskConfig tailored to your chosen dataset.
