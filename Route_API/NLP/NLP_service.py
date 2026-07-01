from pathlib import Path

import pandas as pd
from fastapi import APIRouter

from Route_API.NLP.NLP import NLP

nlp    = NLP()
router = APIRouter(tags=["NLP_service"])

_VOCAB_PATH = Path(__file__).parent.parent.parent / "vocab" / "Flickr8k.token.txt"


def build_vocab() -> tuple[dict, dict]:
    """
    Construit word2idx / idx2word depuis Flickr8k.token.txt
    avec le même nettoyage que l'entraînement du modèle Keras.
    Réplique le tri fréquence-décroissante de Keras Tokenizer (index à partir de 1).
    """
    df = pd.read_csv(_VOCAB_PATH, names=["id", "caption"], delimiter="\t")

    counts: dict[str, int] = {}
    for cap in df["caption"]:
        for word in nlp.clean_for_model(cap).split():
            counts[word] = counts.get(word, 0) + 1

    ordered = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    w2i = {w: i + 1 for i, (w, _) in enumerate(ordered)}
    return w2i, {v: k for k, v in w2i.items()}
