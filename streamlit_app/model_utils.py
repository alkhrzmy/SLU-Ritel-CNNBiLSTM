from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


@dataclass(slots=True)
class LabelMaps:
    product: Dict[str, int]
    quantity: Dict[str, int]
    intent: Dict[str, int]
    inverse_product: Dict[str, str]
    inverse_quantity: Dict[str, str]
    inverse_intent: Dict[str, str]

    @classmethod
    def from_json(cls, path: Path) -> "LabelMaps":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            product=data["product"],
            quantity=data["quantity"],
            intent=data["intent"],
            inverse_product=data["inverse"]["product"],
            inverse_quantity=data["inverse"]["quantity"],
            inverse_intent=data["inverse"]["intent"],
        )

    def decode(self, product_idx: int, quantity_idx: int, intent_idx: int) -> Dict[str, str]:
        return {
            "product": self.inverse_product[str(product_idx)],
            "quantity": self.inverse_quantity[str(quantity_idx)],
            "intent": self.inverse_intent[str(intent_idx)],
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: tuple[int, int]):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.block(x)


class CNNBiLSTMMultiTask(nn.Module):
    def __init__(
        self,
        num_products: int,
        num_quantities: int,
        num_intents: int,
        hidden_size: int = 256,
        lstm_layers: int = 2,
    ):
        super().__init__()
        self.time_downsample = 8

        self.cnn = nn.Sequential(
            ConvBlock(1, 64, pool=(2, 2)),
            ConvBlock(64, 128, pool=(1, 2)),
            ConvBlock(128, 256, pool=(1, 2)),
            nn.AdaptiveAvgPool2d((1, None)),
        )

        self.proj = nn.Sequential(
            nn.Conv1d(256, hidden_size, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0.0,
        )

        self.product_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, num_products),
        )
        self.quantity_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, num_quantities),
        )
        self.intent_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, num_intents),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):  # type: ignore[override]
        feats = self.cnn(x)
        feats = feats.squeeze(2)
        feats = self.proj(feats)
        feats = feats.transpose(1, 2)

        lengths_down = torch.div(lengths + self.time_downsample - 1, self.time_downsample, rounding_mode="floor")
        lengths_down = torch.clamp(lengths_down, min=1, max=feats.size(1))

        packed = nn.utils.rnn.pack_padded_sequence(feats, lengths_down.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        mask = (torch.arange(out.size(1), device=lengths_down.device)[None, :] < lengths_down[:, None]).float().unsqueeze(-1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        return {
            "product": self.product_head(pooled),
            "quantity": self.quantity_head(pooled),
            "intent": self.intent_head(pooled),
        }


def load_model(model_path: Path, label_maps: LabelMaps, device: torch.device | str = "cpu") -> CNNBiLSTMMultiTask:
    model = CNNBiLSTMMultiTask(
        num_products=len(label_maps.product),
        num_quantities=len(label_maps.quantity),
        num_intents=len(label_maps.intent),
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
