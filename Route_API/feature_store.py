"""
Store partagé pour les vecteurs de features image.
Point de découplage entre ImageService (écriture) et ModelService (lecture).
"""

import pickle
from pathlib import Path

import numpy as np

_STORE_PATH = (
    Path("/tmp/features_api.pkl")
    if Path("/tmp").exists()
    else Path(__file__).parent.parent / "features_api.pkl"
)


def _load() -> dict:
    if _STORE_PATH.exists():
        with open(_STORE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def _save(store: dict) -> None:
    with open(_STORE_PATH, "wb") as f:
        pickle.dump(store, f)


def save(image_id: str, features: np.ndarray) -> None:
    store = _load()
    store[image_id] = features
    _save(store)


def get(image_id: str) -> np.ndarray | None:
    return _load().get(image_id)
