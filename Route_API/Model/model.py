import os
import pickle
from pathlib import Path

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

from Route_API.NLP.NLP import NLP

MODEL_PATH     = Path(__file__).parent.parent.parent / "flickr8k_caption_generator_resnet2.keras"
TOKENIZER_PATH = Path(__file__).parent.parent.parent / "tokenizer.pkl"

MAX_LEN  = 34
FEAT_DIM = 2048
_SOS     = "startseq"
_EOS     = "endseq"

_nlp = NLP()


def _load_vocab() -> tuple[dict, dict]:
    """Charge le tokenizer entraîné (word_index) et construit le mapping inverse."""
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    w2i = tokenizer.word_index
    return w2i, {v: k for k, v in w2i.items()}


class CaptionModel:

    def __init__(self):
        self._keras_model     = None
        self._ready           = False
        self._w2i, self._i2w  = _load_vocab()

    # ---- chargement ------------------------------------------------

    def load(self) -> "CaptionModel":
        import keras
        self._keras_model = keras.models.load_model(str(MODEL_PATH), compile=False)
        self._ready = True
        return self

    # ---- propriétés ------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def vocab_size(self) -> int:
        return len(self._w2i) + 1

    # ---- inférence -------------------------------------------------

    def generate(self, features: np.ndarray) -> str:
        seq        = [self._w2i.get(_SOS, 1)]
        image_feat = features.reshape(1, FEAT_DIM).astype("float32")

        for _ in range(MAX_LEN):
            padded   = np.array([_nlp.pad_sequence(seq, MAX_LEN)])
            preds    = self._keras_model.predict([image_feat, padded], verbose=0)
            next_idx = int(np.argmax(preds[0]))
            word     = self._i2w.get(next_idx, "")
            if not word or word == _EOS:
                break
            seq.append(next_idx)

        return " ".join(
            w for i in seq[1:]
            if (w := self._i2w.get(i, "")) and w != _EOS
        )
