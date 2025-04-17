# X-ARES: eXtensive Audio Representation and Evaluation Suite

## Introduction

X-ARES is a benchmark for evaluating audio encoders on various audio tasks. It is heavily inspired by the [HEAR benchmark](https://hearbenchmark.com/).

## Supported tasks

### Speech

- [x] ASV2015
- [x] CREMA-D
- [x] Fluent Speech Commands
- [x] LibriCount
- [x] LibriSpeech-ASR
- [x] LibriSpeech-Male-Female
- [x] RAVDESS
- [x] Speech Commands V2
- [ ] speechocean762
- [x] VocalSound
- [x] VoxCeleb1
- [x] VoxLingua107

### Environment

- [x] Clotho
- [x] DESED
- [x] ESC-50
- [x] FSD18-Kaggle
- [x] FSD50k
- [x] UrbanSound 8k

### Music

- [x] FMA
- [x] GTZAN Genre
- [ ] MAESTRO
- [x] NSynth

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

>>> task = XaresTask(config=asvspoof2015_config(encoder=DashengEncoder()))
>>> task.run()
```

## Baseline Results

X-ARES provides two evaluation methods to assess the quality of audio representations: MLP (Linear Fine-Tuning) and kNN (Unparameterized Evaluation).

**MLP: Linear Fine-Tuning on Task-Specific Data.**
A linear layer will be trained using the provided user embeddings, optimized with predefined hyperparameters for each task. This approach assesses how effectively the fixed representations can be adapted to specific tasks by training an additional linear layer, using predefined hyperparameters tailored for each task. This method evaluates the adaptability and effectiveness of the pre-trained models when applied to new, task-specific contexts without altering the original model parameters.

**kNN: Unparameterized Evaluation.**
Pre-trained model embeddings will be used directly for K-nearest neighbor (KNN) classification without training. This method aims to evaluate the inherent quality of the audio representations without any fine-tuning. While this approach may not always yield the highest performance in real-world applications, it serves as a rigorous test of the fundamental representational power of the embeddings. By avoiding parameterized layers, this method provides a clear view of how well the model captures essential features of the audio data.

Here are the evaluation results for several baseline models using MLP and kNN methods. The weighted average is calculated using the test set size for each dataset.

### MLP Result

| Task                           | dasheng   | wav2vec2 | whisper   | data2vec  |
|:------------------------------:|:---------:|:--------:|:---------:|:---------:|
| ASV2015                        | **0.964** | 0.914    | 0.885     | 0.892     |
| Clotho                         | **0.029** | 0.018    | 0.029     | 0.006     |
| CREMA-D                        | **0.767** | 0.568    | 0.600     | 0.566     |
| DESED                          | **0.537** | 0.081    | 0.125     | 0.137     |
| ESC-50                         | **0.857** | 0.579    | 0.614     | 0.249     |
| Fluent Speech Commands         | 0.946     | 0.417    | 0.878     | **0.962** |
| Free Music Archive Small       | **0.643** | 0.518    | 0.595     | 0.360     |
| FSD50k                         | **0.409** | 0.166    | 0.262     | 0.084     |
| FSD18-Kaggle                   | **0.534** | 0.352    | 0.478     | 0.196     |
| GTZAN Genre                    | **0.851** | 0.681    | 0.751     | 0.495     |
| LibriCount                     | **0.681** | 0.605    | 0.549     | 0.507     |
| LibriSpeech-100h               | 0.713     | 0.001    | 0.802     | **0.860** |
| LibriSpeech-MF                 | **0.986** | 0.952    | 0.877     | 0.754     |
| NSynth-Instruments             | **0.688** | 0.464    | 0.491     | 0.340     |
| RAVDESS                        | **0.749** | 0.440    | 0.460     | 0.469     |
| Speech Commands V1             | **0.969** | 0.805    | 0.955     | 0.930     |
| UrbanSound 8k                  | **0.833** | 0.676    | 0.719     | 0.443     |
| Vocal Imitation                | **0.253** | 0.108    | 0.197     | 0.112     |
| VocalSound                     | **0.910** | 0.791    | 0.871     | 0.807     |
| VoxCeleb1                      | **0.780** | 0.340    | 0.205     | 0.103     |
| VoxLingua33                    | 0.814     | 0.557    | **0.855** | 0.617     |
| **Weighted Average**           | **0.721** | 0.384    | 0.646     | 0.565     |

### kNN Result

| Task                           | dasheng   | wav2vec2 | whisper   | data2vec  |
:------------------------------:|:---------:|:--------:|:---------:|:---------:|
| ASV2015                        | 0.869     | 0.858    | 0.843     | **0.942** |
| CREMA-D                        | 0.380     | 0.175    | **0.382** | 0.325     |
| ESC-50                         | **0.618** | 0.091    | 0.191     | 0.037     |
| Fluent Speech Commands         | **0.260** | 0.008    | 0.032     | 0.156     |
| Free Music Archive Small       | **0.592** | 0.135    | 0.396     | 0.126     |
| GTZAN Genre                    | **0.758** | 0.347    | 0.504     | 0.119     |
| LibriCount                     | **0.311** | 0.241    | 0.253     | 0.186     |
| LibriSpeech-MF                 | 0.791     | 0.552    | 0.586     | **0.632** |
| NSynth-Instruments             | **0.499** | 0.219    | 0.211     | 0.104     |
| RAVDESS                        | **0.408** | 0.171    | 0.287     | 0.289     |
| Speech Commands V1             | **0.903** | 0.208    | 0.096     | 0.850     |
| UrbanSound 8k                  | **0.662** | 0.334    | 0.214     | 0.153     |
| Vocal Imitation                | **0.107** | 0.006    | 0.017     | 0.008     |
| VocalSound                     | 0.382 | 0.265    | **0.417**     | 0.295     |
| VoxCeleb1                      | **0.262** | 0.003    | 0.010     | 0.033     |
| VoxLingua33                    | **0.376** | 0.034    | 0.058     | 0.050     |
| **Weighted Average**           | **0.504** | 0.254    | 0.301     | 0.379     |

## Run with your own pretrained audio encoder

Examples of audio encoder wrapper could be found at `examples`, where the baseline encoders are implemented.

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

### Notes on Encoder implementation

By sure that your encoder supports variable length inference up to 10 minutes of audio.
We recommend to simply chunk the input audio in your encoder to mitigate any out-of-memory issues, like:

```python
class MyCustomEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sampling_rate = 16000
        self.output_dim = 512
        self.hop_size_in_ms = 10
        self.model = my_model_implementation()
        # This code is only for cases where the model itself does not implement chunking
        self.custom_max_audio_length = int(self.sampling_rate * 10)

    def forward(self, audio: torch.Tensor):
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        self.model.eval()
        with torch.inference_mode():
            if audio.shape[-1] > self.custom_max_audio_length:
                embeds = []
                for chunk in audio.split(self.custom_max_audio_length, dim=-1):
                    if chunk.shape[-1] < self.sampling_rate:
                        chunk = torch.nn.functional.pad(
                            chunk, (0, self.sampling_rate - chunk.shape[-1]))
                    
                    embed = self.model(chunk)
                    embeds.append(embed)
                encoded_audio = torch.cat(embeds, dim=1)
            else:
                encoded_audio = self.model(audio)

        return encoded_audio
```

## Add new tasks

Adding a new task is easy. Refer to the existing task implementations for guidance.
You need to create a `TaskConfig` tailored to your chosen dataset.
