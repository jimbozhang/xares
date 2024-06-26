# xares

X-ARES: eXtensive Audio Representation and Evaluation Suite

## Introduction

X-ARES is a toolkit for training, evaluating, and exporting audio encoders for various audio tasks. It is heavily inspired by the [HEAR benchmark](https://hearbenchmark.com/), but offers faster performance, greater flexibility, and enhanced extensibility.

## Roadmap

### Features

- [ ] Basic pipeline for training, evaluating, and exporting
- [ ] Slurm support

### Planned supported tasks

#### Speech

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

#### Environment

- ESC-50
- FSD50k
- UrbanSound 8k
- DCASE Task 4
- (A task designed for the Mi car)
- (A task designed for the Xiaoai soundbox)
- LITIS Rouen
- FSD18-Kaggle
- AudioCaps

#### Music

- MAESTRO
- GTZAN Genre
- NSynth
- MTG-Jamendo
- MusDB18
- FMA

## Installation

```bash
pip install xares
```

## Configure your machine/cluster for training

```plain
to be done
```

## Run with the baseline pretrained audio encoder (Dasheng)

The ESC-50 task is used as an example.

```python
from example.dasheng.dasheng_encoder import DashengEncoder
from tasks.esc50 import esc50_task

task = esc50_task.ESC50Task(env_root="./env", encoder=DashengEncoder(), force_retrain_mlp=True)

task.run_all()
```

## Run with your own pretrained audio encoder

An example of audio encoder wrapper could be found at `example/dasheng/dasheng_encoder.py`. It is very simple because the "dasheng" model is already in the required in/out format.

```python
from dataclasses import dataclass
from dasheng import dasheng_base
from xares.audio_encoder_base import AudioEncoderBase


@dataclass
class DashengEncoder(AudioEncoderBase):
    model = dasheng_base()
    sampling_rate = 16_000
    output_dim = 768

    def __call__(self, audio, sampling_rate):
        # Since the "dasheng" model is already in the required in/out format, we directly use the super class method
        return super().__call__(audio, sampling_rate)
```

Another example could be found at `example/wav2vec2/wav2vec2.py`. It is more complex, you need to covert the input audio and the encoded embedding to the required format.

```plain
to be done
```

## Add your own task

```plain
to be done
```
