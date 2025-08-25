import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from jiwer import wer
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class EncoderAdapter(nn.Module):
    def __init__(self, dim, intermediate_size, n_embd) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(dim, intermediate_size, bias=True)
        self.fc_2 = nn.Linear(dim, intermediate_size, bias=True)
        self.proj = nn.Linear(intermediate_size, n_embd, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class AsrModelForGeneration(nn.Module):
    def __init__(self, in_features, **_):
        super().__init__()

        model_name = "Qwen/Qwen2.5-0.5B"
        self.transformer = AutoModelForCausalLM.from_pretrained(model_name,attn_implementation="sdpa")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.encoder_adapter = EncoderAdapter(
            dim=in_features,
            intermediate_size=self.transformer.config.intermediate_size,
            n_embd=self.transformer.config.hidden_size,
        )
        self.sep_token = "<|vision_end|>"
        self.eos_token = self.tokenizer.eos_token
        self.embed_tokens = self.transformer.get_input_embeddings()

        for param in self.transformer.parameters():
            param.requires_grad = False
        self.criterion = lambda pred, ref: torch.tensor(wer(ref, pred))


    @property
    def device(self):
        return list(self.parameters())[0].device

    def _length_to_mask(self, lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        batch_size = len(lengths)
        idx = torch.arange(max_length, device=self.device)
        idx = idx.repeat(batch_size).view(batch_size, max_length)
        mask = (idx < lengths.unsqueeze(-1)).long()
        return mask

    def _to_tokens(self, text):
        tokens = self.tokenizer(text, padding=True, return_tensors="pt")
        return tokens.input_ids.to(self.device), tokens.attention_mask.to(self.device)

    def _prepare_text(self, text :List[str]) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_text = [self.sep_token + t + self.eos_token for t in text]
        text_tokens, text_masks = self._to_tokens(new_text)
        text_embeddings = self.embed_tokens(text_tokens)
        targets = text_tokens.masked_fill(text_tokens == self.tokenizer.pad_token_id, -100)
        sequence_lengths = text_masks.sum(dim=1)
        batch_indices = torch.arange(text_tokens.size(0), device=text_tokens.device)
        last_token_indices = sequence_lengths - 1
        targets[batch_indices, last_token_indices] = self.tokenizer.pad_token_id
        return text_embeddings, text_masks, targets

    def _prepare_audio_text_inputs(
        self, audio_features, audio_length: Optional[torch.Tensor], text: Optional[List[str]] = None
    ):
        if audio_length is None:
            audio_length = torch.tensor(
                [audio_features.shape[1] for _ in range(len(audio_features))], device=self.device, dtype=torch.long
            )

        audio_mask = self._length_to_mask(audio_length, max_length=audio_features.shape[1])
        input_embeds = audio_features
        input_mask = audio_mask
        labels = None

        if text is not None:
            text_embeds, text_mask, labels = self._prepare_text(text)
            input_embeds = torch.cat([input_embeds, text_embeds], dim=1)
            input_mask = torch.cat([input_mask, text_mask], dim=1)
        return input_embeds, input_mask, labels

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_length: Optional[torch.Tensor] = None,
        text: Optional[List[str]] = None,
        return_loss: bool = True,
    ) -> Union[torch.Tensor, Tuple[List[str], List[str]]]:
        audio_features = self.encoder_adapter(audio_features.to(self.device))
        input_embeds, input_mask, labels = self._prepare_audio_text_inputs(
            audio_features=audio_features, audio_length=audio_length, text=text
        )

        if return_loss:
            with torch.autocast(device_type="cuda"):
                result = self.transformer(
                    input_ids=None,
                    inputs_embeds=input_embeds,
                    attention_mask=input_mask,
                    labels=labels,
                    logits_to_keep=labels.shape[1] if labels is not None else 0,
                )
            return result.loss
        else:  # Inference generate
            with torch.inference_mode(), torch.autocast(device_type="cuda"):
                generated_output = self.transformer.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=input_mask,
                    temperature=1.0,
                    max_new_tokens=200,
                    top_k=None,
                    top_p=1.0,
                )
                generated_texts = self.tokenizer.batch_decode(
                    generated_output, add_special_tokens=False, skip_special_tokens=True
                )
                return generated_texts, text
