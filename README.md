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
- [x] MAESTRO
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

| Dataset                      | dasheng   | wav2vec2 | whisper   | data2vec  |
|------------------------------|-----------|----------|-----------|-----------|
| asvspoof                     | **0.956** | 0.914    | 0.885     | 0.892     |
| clotho                       | **0.033** | 0.018    | 0.029     | 0.006     |
| crema_d                      | **0.772** | 0.568    | 0.600     | 0.566     |
| desde                        | **0.532** | 0.081    | 0.125     | 0.137     |
| esc50                        | **0.869** | 0.579    | 0.614     | 0.249     |
| fluentspeechcommands_kws     | 0.916     | 0.417    | 0.878     | **0.962** |
| freemusicarchive_genre       | **0.640** | 0.518    | 0.595     | 0.360     |
| fsd50k                       | **0.408** | 0.165    | 0.225     | 0.074     |
| fsdkaggle2018                | **0.557** | 0.352    | 0.478     | 0.196     |
| gtzan                        | **0.869** | 0.681    | 0.751     | 0.495     |
| libricount                   | **0.688** | 0.605    | 0.549     | 0.507     |
| librispeech_asr              | 0.061     | 0.---    | **0.---** | 0.---     |
| librispeech_male_female      | 0.859     | 0.703    | **0.877** | 0.692     |
| nsynth_instument             | **0.261** | 0.251    | 0.259     | 0.223     |
| ravdess                      | **0.725** | 0.440    | 0.460     | 0.469     |
| speechcommandsv1             | **0.967** | 0.805    | 0.955     | 0.930     |
| urbansound8k                 | **0.835** | 0.676    | 0.719     | 0.443     |
| vocalimiations               | **0.238** | 0.108    | 0.197     | 0.112     |
| vocalsound                   | **0.910** | 0.791    | 0.871     | 0.807     |
| voxceleb1                    | **0.512** | 0.069    | 0.215     | 0.043     |
| voxlingua33                  | 0.782     | 0.492    | **0.862** | 0.577     |
| **Weighted Average**         | **0.728** | 0.500    | 0.629     | 0.541     |

### kNN Result

| Dataset                       | dasheng   | wav2vec2 | whisper   | data2vec  |
|-------------------------------|-----------|----------|-----------|-----------|
| asvspoof                      | 0.833     | 0.611    | 0.600     | **0.919** |
| crema_d                       | 0.381     | 0.175    | **0.382** | 0.325     |
| esc50                         | **0.621** | 0.091    | 0.191     | 0.037     |
| fluentspeechcommands_kws      | **0.025** | 0.008    | 0.032     | 0.156     |
| freemusicarchive_genre        | **0.589** | 0.135    | 0.396     | 0.126     |
| gtzan                         | **0.753** | 0.347    | 0.504     | 0.119     |
| libricount                    | **0.310** | 0.241    | 0.253     | 0.186     |
| librispeech_male_female       | 0.493     | 0.552    | 0.586     | **0.632** |
| nsynth_instument              | **0.253** | 0.235    | 0.233     | 0.209     |
| ravdess                       | **0.369** | 0.171    | 0.287     | 0.289     |
| speechcommandsv1              | **0.903** | 0.208    | 0.096     | 0.850     |
| urbansound8k                  | **0.662** | 0.334    | 0.214     | 0.153     |
| vocalimiations                | **0.031** | 0.006    | 0.017     | 0.008     |
| vocalsound                    | **0.336** | 0.265    | 0.417     | 0.295     |
| voxceleb1                     | **0.035** | 0.002    | 0.007     | 0.001     |
| voxlingua33                   | **0.340** | 0.014    | 0.207     | 0.050     |
| **Weighted Average**          | **0.384** | 0.271    | 0.251     | 0.350     |

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

## Add new tasks

Adding a new task is easy. Refer to the existing task implementations for guidance.
You need to create a `TaskConfig` tailored to your chosen dataset.
