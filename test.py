from example.dasheng.dasheng_encoder import DashengEncoder
from example.wav2vec2.wav2vec2_encoder import Wav2vec2Encoder
from tasks.esc50 import esc50_task

task = esc50_task.ESC50Task(env_root="./env", encoder=Wav2vec2Encoder(), force_retrain_mlp=True) #encoder=Wav2vec2Encoder() DashengEncoder()

task.run_all()