import io
import pickle
import numpy as np
from PIL import Image


class ImageProcessingPipeline:

    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    # ================================================================
    #  1. CHARGEMENT
    # ================================================================

    def load_from_path(self, image_path: str) -> Image.Image:
        """Charge une image depuis un chemin fichier."""
        return Image.open(image_path).convert("RGB")

    def load_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """Charge une image depuis des bytes (upload API)."""
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ================================================================
    #  2. TRAITEMENT GÉOMÉTRIQUE
    # ================================================================

    def resize(self, img: Image.Image) -> Image.Image:
        """Redimensionne l'image à la taille cible."""
        return img.resize((self.target_size[1], self.target_size[0]))

    # ================================================================
    #  3. CONVERSION NUMÉRIQUE
    # ================================================================

    def to_array(self, img: Image.Image) -> np.ndarray:
        """Convertit une image PIL en tableau numpy float32."""
        return np.array(img, dtype="float32")

    def normalize(self, img_array: np.ndarray) -> np.ndarray:
        """Normalise les pixels dans l'intervalle [0, 1]."""
        return img_array / 255.0

    def add_batch_dim(self, img_array: np.ndarray) -> np.ndarray:
        """Ajoute la dimension batch → (1, H, W, C)."""
        return img_array.reshape((1,) + img_array.shape)

    # ================================================================
    #  4. EXTRACTION DES FEATURES
    # ================================================================

    def extract_features(self, img_tensor: np.ndarray, model_backbone) -> np.ndarray:
        """Extrait le vecteur de features via un backbone CNN."""
        return model_backbone.predict(img_tensor, verbose=0)

    def flatten_features(self, feature_vector: np.ndarray) -> np.ndarray:
        """Aplatit le vecteur de features → tableau 1D."""
        return feature_vector.reshape(-1)

    # ================================================================
    #  5. PERSISTANCE
    # ================================================================

    @staticmethod
    def save_features(features_dict: dict, filename: str = "features.pkl"):
        """Sauvegarde les features extraites dans un fichier sérialisé."""
        with open(filename, 'wb') as f:
            pickle.dump(features_dict, f)

    @staticmethod
    def load_features(filename: str = "features.pkl") -> dict:
        """Charge les features depuis un fichier sérialisé."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
