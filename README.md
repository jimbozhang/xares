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

The ESC-50 task is used as an example.

```python
from example.dasheng.dasheng_encoder import DashengEncoder
from tasks.esc50.esc50_task import ESC50Task

task = ESC50Task(encoder=DashengEncoder())

score = task.run()
```

## Run all tasks parallelly

You can run all tasks parallelly with the following command:

```bash
python run.py [-h] [--max-jobs MAX_JOBS] [--task-list TASK_LIST] encoder_module encoder_class
```

The command line arguments are as follows:

```plaintext
Run tasks with a maximum concurrency limit.

positional arguments:
  encoder_module        Encoder module. eg: example.dasheng.dasheng_encoder
  encoder_class         Encoder classname. eg: DashengEncoder

options:
  -h, --help            show this help message and exit
  --max-jobs MAX_JOBS   Maximum number of concurrent tasks.
  --task-list TASK_LIST File containing a list of task modules to execute.
```

There is an example task list file at `tasklist`.

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
