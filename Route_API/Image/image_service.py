import uuid

import numpy as np

from Route_API.Image.image import ImageProcessingPipeline
from Route_API import feature_store


class ImageService:

    _pipeline       = ImageProcessingPipeline(target_size=(299, 299))
    _model_backbone = None

    # ================================================================
    #  CHARGEMENT DU BACKBONE
    # ================================================================

    @classmethod
    def set_model(cls, model_backbone):
        cls._model_backbone = model_backbone

    # ================================================================
    #  PREPROCESSING — bytes → (1, 299, 299, 3) normalisé InceptionV3
    # ================================================================

    @classmethod
    def _preprocess(cls, image_bytes: bytes) -> np.ndarray:
        img = cls._pipeline.load_from_bytes(image_bytes)
        img = cls._pipeline.resize(img)
        arr = cls._pipeline.to_array(img)   # float32, [0, 255]
        arr = (arr / 127.5) - 1.0           # InceptionV3 : [-1, 1]
        return cls._pipeline.add_batch_dim(arr)

    # ================================================================
    #  API PUBLIQUE
    # ================================================================

    @classmethod
    def extract_features(cls, image_bytes: bytes) -> np.ndarray | None:
        """Preprocessing + extraction → vecteur numpy (2048,), ou None si backbone absent."""
        if cls._model_backbone is None:
            return None
        tensor   = cls._preprocess(image_bytes)
        features = cls._pipeline.extract_features(tensor, cls._model_backbone)
        return cls._pipeline.flatten_features(features)

    # ================================================================
    #  PERSISTANCE via FeatureStore partagé
    # ================================================================

    @classmethod
    def process_image(cls, image_bytes: bytes, filename: str = "") -> dict:
        features = cls.extract_features(image_bytes)
        if features is None:
            return {
                "status":  "preprocessing_ok",
                "message": "Modèle non chargé — extraction de features en attente",
            }
        image_id = filename or str(uuid.uuid4())
        feature_store.save(image_id, features)
        return {"status": "features_extracted", "image_id": image_id}
