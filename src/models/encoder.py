import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import Wav2Vec2Model
from torch import nn


class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_config):
        
        def extract_variable_length_features(self, x: torch.Tensor):
            """
            x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
                the mel spectrogram of the audio
            """
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1)

            # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
            # x = (x + self.positional_embedding).to(x.dtype)
            x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

            for block in self.blocks:
                x = block(x)

            x = self.ln_post(x)
            return x

        if model_config.encoder_path_hf is not None:
            from transformers import WhisperModel
            encoder = WhisperModel.from_pretrained(model_config.encoder_path_hf,torch_dtype=torch.bfloat16).encoder
        else:
            import whisper
            encoder = whisper.load_model(name=model_config.encoder_path, device='cpu').encoder
            encoder.extract_variable_length_features = types.MethodType(extract_variable_length_features, encoder)
        return encoder



class Wav2Vec2WrappedEncoder(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_path)

    def forward(self, input_values, attention_mask=None):
        outputs = self.model(input_values=input_values, attention_mask=attention_mask)
        return outputs.last_hidden_state  # hoặc pooled output nếu cần