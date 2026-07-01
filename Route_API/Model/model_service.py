import numpy as np

from Route_API.Model.model import CaptionModel, MODEL_PATH, FEAT_DIM
from Route_API import feature_store


class ModelService:

    _caption_model: CaptionModel | None = None

    @classmethod
    def load(cls, _device=None) -> None:
        if not MODEL_PATH.exists():
            print(f"[ModelService] Modèle introuvable : {MODEL_PATH}")
            return
        cls._caption_model = CaptionModel().load()
        print(f"[ModelService] Keras OK | vocab={cls._caption_model.vocab_size} mots")

    @classmethod
    def generate_caption(cls, image_id: str) -> dict | None:
        if cls._caption_model is None or not cls._caption_model.ready:
            return {"status": "error", "caption": "Modèle non chargé."}

        features = feature_store.get(image_id)
        if features is None:
            return None

        features = np.array(features, dtype="float32")
        if features.shape[0] != FEAT_DIM:
            return {
                "status":  "error",
                "caption": f"Features obsolètes ({features.shape[0]}-dim) — re-uploadez l'image.",
            }

        return {"status": "ok", "caption": cls._caption_model.generate(features)}

    @classmethod
    def get_status(cls) -> dict:
        return {
            "model_ready": cls._caption_model is not None and cls._caption_model.ready,
            "model_path":  str(MODEL_PATH),
            "vocab_size":  cls._caption_model.vocab_size if cls._caption_model else 0,
        }
