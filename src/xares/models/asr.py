import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from jiwer import wer
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


class Decoder(nn.Module):
    def __init__(self, audio_features_dim: int):
        super().__init__()

        model_name = "Qwen/Qwen2.5-0.5B"
        self.transformer = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.encoder_adapter = EncoderAdapter(
            dim=audio_features_dim,
            intermediate_size=self.transformer.config.intermediate_size,
            n_embd=self.transformer.config.hidden_size,
        )

        self.max_seq_length = 4096
        self.past_key_values = None

    def freeze_lm(self) -> None:
        for param in self.transformer.parameters():
            param.requires_grad = False

    def clear_kvcache(self) -> None:
        self.past_key_values = None

    def forward(
        self,
        encoded_audio: Optional[torch.Tensor] = None,
        attention_mask_a: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask_t: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        enable_kvcache: bool = False,
    ) -> Tuple[List[Tensor], Tensor]:
        assert encoded_audio is not None or input_ids is not None, "Either encoded_audio or input_ids must be provided"
        assert self.transformer.device == self.encoder_adapter.fc_1.weight.device
        device = self.transformer.device

        if encoded_audio is not None:
            encoded_audio = encoded_audio.to(device)
            x_a = self.encoder_adapter(encoded_audio)
            if attention_mask_a is None:
                attention_mask_a = torch.ones(x_a.shape[0], x_a.shape[1], dtype=torch.long)
            attention_mask_a = attention_mask_a.to(device)
        else:
            x_a = torch.tensor([], device=device)
            attention_mask_a = torch.tensor([], device=device)

        if input_ids is not None:
            input_ids = input_ids.to(device)
            if input_ids.dim() == 1:
                input_ids = torch.unsqueeze(input_ids, 0)
            x_t = self.transformer.get_input_embeddings()(input_ids)
            if attention_mask_t is None:
                attention_mask_t = torch.ones(x_t.shape[0], x_t.shape[1], dtype=torch.long)
            attention_mask_t = attention_mask_t.to(device)
        else:
            x_t = torch.tensor([], device=device)
            attention_mask_t = torch.tensor([], device=device)

        x = torch.cat([x_a, x_t], dim=1)
        attention_mask = torch.cat([attention_mask_a, attention_mask_t], dim=1)

        output = self.transformer(
            inputs_embeds=x,
            attention_mask=attention_mask,
            past_key_values=self.past_key_values if enable_kvcache else None,
            labels=labels,
            num_logits_to_keep=labels.shape[1] if labels is not None else 0,
        )

        if enable_kvcache:
            self.past_key_values = output.past_key_values

        return output


# Copied from https://github.com/Lightning-AI/litgpt/blob/f6031e3a88e272ec86ad8f412573699589f4d41b/litgpt/generate/base.py#L30
def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Example:
    # sorted_probs=[0.1, 0.15, 0.2, 0.25, 0.3] -> sorted_cumprobs=[0.1, 0.25, 0.45, 0.7, 1.0]
    # sorted_indices_to_remove = [1, 1, 0, 0, 0] if top_p=0.7
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least 1 token always to prevent the case where no token is selected
    # In this case the most probable one is always kept
    sorted_indices_to_remove[-1:] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def sample(
    logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None, top_p: float = 1.0
) -> torch.Tensor:
    if top_p < 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    logits = logits[0, -1]
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        # do not use `torch.where` as in nanogpt because it will repeat top-k collisions
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    # optionally scale the logits and sample from a probability distribution
    if temperature > 0.0 or top_p > 0.0:
        if temperature > 0.0:
            logits = logits / temperature
        # optionally crop the logits to smallest set of logits with a cumulative probability above top_p
        if top_p < 1.0:
            logits = sample_top_p(logits, top_p)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return multinomial_num_samples_1(probs)
    return torch.argmax(logits, dim=-1, keepdim=True)


## End of copied code


class AsrModelForGeneration(nn.Module):
    def __init__(self, in_features, **_):
        super().__init__()

        self.decoder = Decoder(audio_features_dim=in_features)
        self.decoder.freeze_lm()
        self.tokenizer = self.decoder.tokenizer
        self.sep_token = "<|vision_end|>"

        self.criterion = lambda pred, ref: torch.tensor(wer(ref, pred))

    def forward(self, batch_a: torch.Tensor, batch_t: torch.Tensor = None, return_loss: bool = True) -> torch.Tensor:
        result = self.decoder(
            encoded_audio=batch_a,
            attention_mask_a=None,
            input_ids=batch_t,
            attention_mask_t=None,
            labels=batch_t if return_loss else None,
        )
        if return_loss:
            return result.loss
        else:
            # Generate text for each audio input in the batch
            generated_texts = []
            for i in range(batch_a.shape[0]):
                single_audio = batch_a[i : i + 1]  # Keep batch dimension
                output_text = self.tokenizer.decode(self.generate(single_audio))
                generated_texts.append(output_text)

            label_texts = []
            assert batch_a.shape[0] == batch_t.shape[0]
            for i in range(batch_t.shape[0]):
                label_text = self.tokenizer.decode(batch_t[i])
                label_text = label_text.replace(self.sep_token, "").strip(self.tokenizer.pad_token)
                label_texts.append(label_text)

            return generated_texts, label_texts

    @torch.inference_mode()
    def generate(
        self,
        audio_features: torch.Tensor,
        max_returned_tokens: int = 200,
        *,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        input_ids = self.decoder.tokenizer(self.sep_token, return_tensors="pt")
        self.decoder.clear_kvcache()
        logits = self.decoder(
            encoded_audio=audio_features,
            attention_mask_a=None,
            input_ids=input_ids["input_ids"],
            enable_kvcache=True,
        ).logits

        outputs = []
        next_t = sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)

        while next_t != self.decoder.tokenizer.eos_token_id and len(outputs) < max_returned_tokens:
            outputs.append(next_t.item())
            logits = self.decoder(input_ids=next_t, enable_kvcache=True).logits
            next_t = sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        return outputs
