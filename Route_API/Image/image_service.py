import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from Route_API.Image.image import ImageProcessingPipeline

FEATURES_STORE = Path("/tmp/features_api.pkl") if Path("/tmp").exists() else Path(__file__).parent.parent.parent / "features_api.pkl"

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class _TorchBackboneAdapter:
    """
    Adapte un feature extractor PyTorch (ResNet34 sans FC)
    à l'interface .predict(numpy) attendue par ImageProcessingPipeline.

    Entrée  : numpy (1, 224, 224, 3)  float32 [0, 1]  — channels last
    Sortie  : numpy (1, 512)          — vecteur de features
    """

    def __init__(self, feature_extractor: nn.Module, device):
        self._extractor = feature_extractor
        self._device    = device

    def predict(self, img_array: np.ndarray, verbose=0) -> np.ndarray:  # noqa: ARG002
        # (1, H, W, C) → (1, C, H, W) puis normalisation ImageNet
        tensor = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = tensor.to(self._device)
        with torch.no_grad():
            features = self._extractor(tensor)          # (1, 512, 1, 1)
            features = features.squeeze(-1).squeeze(-1) # (1, 512)
        return features.cpu().numpy()


class ImageService:

    _pipeline       = ImageProcessingPipeline(target_size=(224, 224))
    _model_backbone = None  # chargé quand le modèle sera prêt

    # ================================================================
    #  CHARGEMENT DU BACKBONE
    # ================================================================

    @classmethod
    def set_model(cls, model_backbone):
        """Charge le backbone CNN une fois le modèle disponible."""
        cls._model_backbone = model_backbone

    @classmethod
    def load_torch_backbone(cls, feature_extractor: nn.Module, device):
        """Branche un extracteur PyTorch (ResNet34 sans FC) comme backbone."""
        cls._model_backbone = _TorchBackboneAdapter(feature_extractor, device)

    # ================================================================
    #  PREPROCESSING COMMUN — bytes → tensor batch numpy (1, H, W, C)
    # ================================================================

    @classmethod
    def _preprocess(cls, image_bytes: bytes) -> np.ndarray:
        img = cls._pipeline.load_from_bytes(image_bytes)
        img = cls._pipeline.resize(img)
        arr = cls._pipeline.to_array(img)
        arr = cls._pipeline.normalize(arr)
        return cls._pipeline.add_batch_dim(arr)

    # ================================================================
    #  API PUBLIQUE
    # ================================================================

    @classmethod
    def extract_features(cls, image_bytes: bytes) -> np.ndarray | None:
        """Preprocessing + extraction → vecteur numpy (512,), ou None si backbone absent."""
        if cls._model_backbone is None:
            return None
        tensor   = cls._preprocess(image_bytes)
        features = cls._pipeline.extract_features(tensor, cls._model_backbone)
        return cls._pipeline.flatten_features(features)

    # ================================================================
    #  PERSISTANCE — save_features / load_features de image.py
    # ================================================================

    @classmethod
    def _load_store(cls) -> dict:
        """Charge le dictionnaire {image_id: features} depuis le pkl."""
        if FEATURES_STORE.exists():
            return cls._pipeline.load_features(str(FEATURES_STORE))
        return {}

    @classmethod
    def store_features(cls, image_id: str, features: np.ndarray):
        """Ajoute les features d'une image dans le store persistant."""
        store = cls._load_store()
        store[image_id] = features
        cls._pipeline.save_features(store, str(FEATURES_STORE))

    @classmethod
    def get_features(cls, image_id: str) -> np.ndarray | None:
        """Récupère les features d'une image depuis le store."""
        return cls._load_store().get(image_id)

    # ================================================================
    #  API PUBLIQUE
    # ================================================================

    @classmethod
    def process_image(cls, image_bytes: bytes, filename: str = "") -> dict:
        """
        Preprocessing + extraction + persistance.
        Retourne l'image_id pour que le model puisse récupérer le vecteur.
        """
        features = cls.extract_features(image_bytes)
        if features is None:
            return {
                "status":  "preprocessing_ok",
                "message": "Modèle non chargé — extraction de features en attente"
            }

        image_id = filename or str(uuid.uuid4())
        cls.store_features(image_id, features)
        return {
            "status":   "features_extracted",
            "image_id": image_id
        }
