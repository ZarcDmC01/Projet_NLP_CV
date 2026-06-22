"""
Logique métier du modèle LSTM.
Construit le vocab, charge les poids, orchestre l'inférence.
"""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Route_API.Model.model import (
    LSTMCaptioner,
    MODEL_PATH,
    EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, MAX_LEN, MIN_FREQ,
    PAD, UNK, SOS, EOS,
)

TEXT_PATH = Path(__file__).parent.parent.parent / "vocab"


def _tokenize(caption: str) -> list[str]:
    caption = caption.lower().strip()
    caption = re.sub(r"[^\w\s]", "", caption)
    return caption.split()


def build_vocab() -> tuple[dict, dict]:
    """Lit Flickr8k.token.txt et retourne (word2idx, idx2word)."""
    token_file = TEXT_PATH / "Flickr8k.token.txt"
    df = pd.read_csv(token_file, names=["id", "caption"], delimiter="\t")
    df["id"] = df["id"].apply(lambda x: x.split("#")[0])

    all_tokens  = [tok for cap in df["caption"] for tok in _tokenize(cap)]
    word_counts = Counter(all_tokens)
    vocab_words = [PAD, UNK, SOS, EOS] + [
        w for w, c in word_counts.most_common() if c >= MIN_FREQ
    ]
    word2idx = {w: i for i, w in enumerate(vocab_words)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word


def _load_model(word2idx: dict, device) -> tuple[LSTMCaptioner, bool]:
    """Instancie LSTMCaptioner et charge les poids si disponibles."""
    model = LSTMCaptioner(
        feature_dim=512,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        vocab_size=len(word2idx),
        num_layers=NUM_LAYERS,
    ).to(device)

    if not MODEL_PATH.exists():
        return model, False

    state = torch.load(MODEL_PATH, weights_only=True, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, True


class ModelService:

    _model    = None
    _word2idx = None
    _idx2word = None
    _device   = None
    _ready    = False

    # ================================================================
    #  Chargement (appelé une fois au démarrage via lifespan)
    # ================================================================

    @classmethod
    def load(cls, device):
        cls._device   = device
        cls._word2idx, cls._idx2word = build_vocab()
        cls._model, cls._ready = _load_model(cls._word2idx, device)
        print(f"[ModelService] LSTM {'OK' if cls._ready else 'non trouvé'} | "
              f"vocab={len(cls._word2idx)} mots | device={device}")

    # ================================================================
    #  Boucle de génération — orchestration sur les ops atomiques
    # ================================================================

    @classmethod
    def _generate_sequence(cls, features: torch.Tensor) -> str:
        """Décode une séquence de mots à partir d'un vecteur image (1, 512)."""
        h, c  = cls._model.init_hidden(features)
        token = torch.tensor([[cls._word2idx[SOS]]], device=cls._device)
        words: list[str] = []
        for _ in range(MAX_LEN):
            logits, h, c = cls._model.step(token, h, c)
            pred         = logits.argmax(-1)
            word         = cls._idx2word[pred.item()]
            if word == EOS:
                break
            if word not in (PAD, UNK):
                words.append(word)
            token = pred.unsqueeze(0)
        return " ".join(words)

    # ================================================================
    #  API publique
    # ================================================================

    @classmethod
    def generate_caption(cls, image_id: str) -> dict:
        """
        Récupère les features persistées pour image_id (via ImageService),
        lance la génération et retourne {"status": ..., "caption": ...}.
        """
        # Import local pour éviter la dépendance circulaire au chargement du module
        from Route_API.Image.image_service import ImageService

        if not cls._ready:
            return {
                "status":  "error",
                "caption": "Modèle non chargé — lancez CV_Image_Description_Local.ipynb.",
            }

        raw = ImageService.get_features(image_id)
        if raw is None:
            return None  # le controller lèvera une 404

        if isinstance(raw, np.ndarray):
            features = torch.from_numpy(raw).float().unsqueeze(0).to(cls._device)
        else:
            features = raw.unsqueeze(0).to(cls._device)

        caption = cls._generate_sequence(features)
        return {"status": "ok", "caption": caption}

    # ================================================================
    #  Statut
    # ================================================================

    @classmethod
    def get_status(cls) -> dict:
        return {
            "model_ready": cls._ready,
            "model_path":  str(MODEL_PATH),
            "vocab_size":  len(cls._word2idx) if cls._word2idx else 0,
            "device":      str(cls._device),
        }
