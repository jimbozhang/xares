import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def download_model_to_local(model_names: str | List[str], output_root: str = "."):
    # Download the model to the local directory to avoid issues under multi-processes
    if isinstance(model_names, str):
        model_names = [model_names]

    if "bert-base-uncased:tokenizer" in model_names:
        model_name = "google-bert/bert-base-uncased"
        output_path = Path(output_root) / model_name
        if not output_path.exists():
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(output_path)
        model_names.remove("bert-base-uncased:tokenizer")

    if len(model_names) > 0:
        logger.warning(f"Models {model_names} are not supported for download")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Mlp(nn.Module):
    def __init__(self, in_features, out_features: int | None = None, criterion="CrossEntropyLoss"):
        super().__init__()
        out_features = out_features or in_features
        self.ln = nn.LayerNorm(in_features)
        self.fc = nn.Linear(in_features, out_features)
        self.criterion = getattr(nn, criterion)()

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None, return_loss: bool = False):
        x = self.ln(x)
        x = self.fc(x)
        if y is not None and return_loss:
            return self.criterion(x, y)
        return x, y


class AudioTextContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, embed_regularization: bool = True):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.tensor(1.0 / temperature).log(), requires_grad=True)
        self.embed_regularization = embed_regularization

    def similarity(self, emb_x, emb_y):
        with torch.autocast(device_type="cuda", enabled=False):
            return self.temperature.exp() * emb_x @ emb_y.t()

    def forward(self, audio_embed, text_embed):
        sim_a2t = self.similarity(audio_embed, text_embed)
        sim_t2a = self.similarity(text_embed, audio_embed)

        sim_targets = torch.zeros(sim_a2t.size()).to(sim_a2t.device)
        sim_targets.fill_diagonal_(1)

        loss_a2t = -torch.sum(F.log_softmax(sim_a2t, dim=1) * sim_targets, dim=1).mean()

        loss_t2a = -torch.sum(F.log_softmax(sim_t2a, dim=1) * sim_targets, dim=1).mean()

        loss_atc = (loss_a2t + loss_t2a) / 2
        if self.embed_regularization:
            loss_atc = (
                loss_atc
                + torch.mean(torch.abs(audio_embed)) / torch.sqrt(torch.sum(audio_embed**2))
                + torch.mean(torch.abs(text_embed)) / torch.sqrt(torch.sum(text_embed**2))
            )
        return loss_atc


class RetrivalMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features: int = 1024,
        out_features: int | None = None,
        criterion="AudioTextContrastiveLoss",
    ):
        super().__init__()
        out_features = out_features or hidden_features
        from transformers import AutoTokenizer

        self.ln = nn.LayerNorm(in_features)
        self.audio_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features), nn.GELU(), nn.Linear(hidden_features, out_features)
        )
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.text_model = torch.nn.Embedding(self.tokenizer.vocab_size, hidden_features)
        self.pad_token_id = self.tokenizer.pad_token_id

        self.text_mlp = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
        )

        self.criterion = globals()[criterion]()

    def forward(self, x: torch.Tensor, text: torch.Tensor, return_loss: bool = False):
        text_inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        model_inputs = text_inputs.to(x.device)
        text_embeddings = self.text_model(model_inputs.input_ids)
        text_embeddings = self.text_mlp(text_embeddings)
        outputmask = (model_inputs.input_ids != self.pad_token_id).unsqueeze(-1).float()
        text_embeddings = (text_embeddings * outputmask).sum(1) / outputmask.sum(1)
        x = self.ln(x)
        x = self.audio_mlp(x)
        if return_loss:
            return self.criterion(x, text_embeddings)
        return x, text_embeddings
