import torch
from dataclasses import dataclass
from xares.audio_encoder_base import AudioEncoderBase
from dasheng.train.models import AudioTransformerMAE_Encoder
from dasheng.pretrained.pretrained import *

class Dasheng(Dasheng):
    @classmethod
    def from_pretrained(
            cls, pretrained_url: str,
            **additional_model_kwargs) -> AudioTransformerMAE_Encoder:
        """
        Class method to create a new Vocos model instance from a pre-trained model stored in the Hugging Face model hub.
        """
        if 'http' in pretrained_url:
            dump = torch.hub.load_state_dict_from_url(pretrained_url,
                                                      map_location='cpu')
        else:
            dump = torch.load(pretrained_url, map_location='cpu')
    
        model_parmeters, model_config = dump['model'], dump['config']
        model_config.update({
            'depth':12, 'num_heads':12, 'embed_dim':768,
        })
        instance = cls(**{**model_config, **additional_model_kwargs})
        instance.load_state_dict(model_parmeters, strict=True)
        return instance


@dataclass
class DashengEncoder(AudioEncoderBase):
    model = Dasheng.from_pretrained(PRETRAINED_CHECKPOINTS['dasheng_base'])
    sampling_rate = 16_000 # model sr
    output_dim = 768

    def __call__(self, audio, sampling_rate = 44_100): # dataset sr
        # Since the "dasheng" model is already in the required in/out format, we directly use the super class method
        return super().__call__(audio, sampling_rate)