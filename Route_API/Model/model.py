"""
Architecture du modèle LSTM — opérations atomiques uniquement.
"""

from pathlib import Path

import torch
import torch.nn as nn

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "caption_model_best.pth"

EMBED_DIM  = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
MAX_LEN    = 40
MIN_FREQ   = 3

PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<start>", "<end>"


class LSTMCaptioner(nn.Module):
    """Décodeur LSTM : vecteur image (512-dim) → séquence de mots."""

    def __init__(self, feature_dim: int, embed_dim: int,
                 hidden_dim: int, vocab_size: int, num_layers: int):
        super().__init__()
        self.image_proj = nn.Linear(feature_dim, hidden_dim)
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm       = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc         = nn.Linear(hidden_dim, vocab_size)
        self.dropout    = nn.Dropout(0.3)

    def forward(self, features, captions):
        """Passe forward complète (entraînement — teacher forcing)."""
        h0     = self.image_proj(features).unsqueeze(0)
        c0     = torch.zeros_like(h0)
        embeds = self.dropout(self.embedding(captions[:, :-1]))
        out, _ = self.lstm(embeds, (h0, c0))
        return self.fc(out)

    def init_hidden(self, features):
        """Initialise l'état caché LSTM depuis le vecteur image."""
        h = self.image_proj(features).unsqueeze(0)
        c = torch.zeros_like(h)
        return h, c

    def step(self, token, h, c):
        """Un pas de décodage : token courant → logits + nouvel état caché."""
        embed         = self.embedding(token)
        out, (h, c)   = self.lstm(embed, (h, c))
        logits        = self.fc(out.squeeze(1))
        return logits, h, c
