from example.dasheng.dasheng_encoder import DashengEncoder
from example.wav2vec2.wav2vec2_encoder import Wav2vec2Encoder
from example.ced.ced_encoder import CedEncoder
from tasks.esc50 import esc50_task

task = esc50_task.ESC50Task(env_root="./env", encoder=DashengEncoder(), force_retrain_mlp=True) #, force_generate_encoded_tar=True) #换encoder须加force_generate_encoded_tar=True
task.run_all()

# task = esc50_task.ESC50Task(env_root="./env", encoder=Wav2vec2Encoder(), force_retrain_mlp=True, force_generate_encoded_tar=True) 
# task.run_all()

# task = esc50_task.ESC50Task(env_root="./env", encoder=CedEncoder(), force_retrain_mlp=True, force_generate_encoded_tar=True) 
# task.run_all()