import gc
import uuid

import numpy as np
import torch
import torchvision.transforms as transforms

from Route_API.Image.image import ImageProcessingPipeline
from Route_API import feature_store

# Normalisation ImageNet standard torchvision, utilisée à l'entraînement (scripts/main.py)
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageService:

    _pipeline       = ImageProcessingPipeline()
    _model_backbone = None  # torchvision.models.resnet50 (fc remplacé par Identity), en mode eval

    # ================================================================
    #  CHARGEMENT DU BACKBONE
    # ================================================================

    @classmethod
    def set_model(cls, model_backbone):
        cls._model_backbone = model_backbone

    # ================================================================
    #  PREPROCESSING — bytes → tenseur (1, 3, 224, 224) normalisé ResNet50 (ImageNet)
    # ================================================================

    @classmethod
    def _preprocess(cls, image_bytes: bytes) -> torch.Tensor:
        img = cls._pipeline.load_from_bytes(image_bytes)
        return _TRANSFORM(img).unsqueeze(0)

    # ================================================================
    #  API PUBLIQUE
    # ================================================================

    @classmethod
    def extract_features(cls, image_bytes: bytes) -> np.ndarray | None:
        """Preprocessing + extraction → vecteur numpy (2048,), ou None si backbone absent."""
        if cls._model_backbone is None:
            return None
        tensor = cls._preprocess(image_bytes)
        with torch.no_grad():
            output = cls._model_backbone(tensor)
        features = output.cpu().numpy().flatten()
        del tensor, output
        gc.collect()
        return features

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
