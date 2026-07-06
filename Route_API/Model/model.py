import os
import pickle
from pathlib import Path

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

from Route_API.NLP.NLP import NLP

MODEL_PATH     = Path(__file__).parent.parent.parent / "flickr8k_caption_generator_resnet4.keras"
TOKENIZER_PATH = Path(__file__).parent.parent.parent / "tokenizer.pkl"
META_PATH      = Path(__file__).parent.parent.parent / "text_meta.pkl"

FEAT_DIM   = 2048
BEAM_WIDTH = 3

_nlp = NLP()


def _load_vocab() -> tuple[dict, dict, int]:
    """Charge le tokenizer entraîné (word_index) et construit le mapping inverse.
    vocab_size = tokenizer.num_words + 1 : c'est la taille réelle de la couche de
    sortie du modèle (le tokenizer garde tous les mots vus, mais num_words limite
    ceux effectivement utilisés à l'entraînement)."""
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    w2i = tokenizer.word_index
    i2w = {v: k for k, v in w2i.items()}
    vocab_size = (tokenizer.num_words or len(w2i)) + 1
    return w2i, i2w, vocab_size


def _load_meta() -> dict:
    """Charge max_length et les tokens start/end sauvegardés à l'entraînement."""
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


class CaptionModel:

    def __init__(self):
        self._keras_model = None
        self._ready       = False
        self._w2i, self._i2w, self._vocab_size = _load_vocab()
        meta          = _load_meta()
        self._max_len = meta["max_length"]
        self._sos     = meta["start_token"]
        self._eos     = meta["end_token"]

    # ---- chargement ------------------------------------------------

    def load(self) -> "CaptionModel":
        import keras
        import torch
        torch.set_num_threads(1)
        self._keras_model = keras.models.load_model(str(MODEL_PATH), compile=False)
        self._ready = True
        return self

    # ---- propriétés ------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    # ---- inférence -------------------------------------------------

    def generate(self, features: np.ndarray) -> str:
        """Génère une légende par beam search (meilleure qualité qu'un simple argmax)."""
        image_feat = features.reshape(1, FEAT_DIM).astype("float32")
        sos_idx    = self._w2i.get(self._sos, 1)
        eos_idx    = self._w2i.get(self._eos)

        candidates = [(0.0, [sos_idx])]

        for _ in range(self._max_len):
            next_candidates = []

            for score, seq in candidates:
                if seq[-1] == eos_idx:
                    next_candidates.append((score, seq))
                    continue

                padded = np.array([_nlp.pad_sequence(seq, self._max_len)])
                preds  = self._keras_model.predict([image_feat, padded], verbose=0)[0]

                for idx in np.argsort(preds)[-BEAM_WIDTH:]:
                    idx = int(idx)
                    next_candidates.append((score + float(np.log(preds[idx] + 1e-10)), seq + [idx]))

            candidates = sorted(next_candidates, key=lambda c: c[0], reverse=True)[:BEAM_WIDTH]

            if all(seq[-1] == eos_idx for _, seq in candidates):
                break

        best_seq = candidates[0][1]
        return " ".join(
            w for i in best_seq[1:]
            if i != eos_idx and (w := self._i2w.get(i, ""))
        )
