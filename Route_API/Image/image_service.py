from Route_API.Image.image import ImageProcessingPipeline


class ImageService:

    _pipeline = ImageProcessingPipeline(target_size=(224, 224))
    _model_backbone = None  # chargé quand le modèle sera prêt

    # ================================================================
    #  PIPELINE COMPLÈTE — une image en bytes → vecteur de features
    # ================================================================

    @classmethod
    def process_image(cls, image_bytes: bytes) -> dict:
        """
        Orchestre toutes les étapes du pipeline :
        1. load_from_bytes  →  charge l'image depuis l'upload
        2. resize           →  redimensionne à (224, 224)
        3. to_array         →  PIL → numpy float32
        4. normalize        →  pixels / 255 → [0, 1]
        5. add_batch_dim    →  ajoute la dimension batch (1, H, W, C)
        6. extract_features →  vecteur via backbone CNN  (si modèle chargé)
        7. flatten_features →  vecteur 1D
        """
        img    = cls._pipeline.load_from_bytes(image_bytes)
        img    = cls._pipeline.resize(img)
        arr    = cls._pipeline.to_array(img)
        arr    = cls._pipeline.normalize(arr)
        tensor = cls._pipeline.add_batch_dim(arr)

        if cls._model_backbone is None:
            return {
                "status": "preprocessing_ok",
                "tensor_shape": list(tensor.shape),
                "message": "Modèle non chargé — extraction de features en attente"
            }

        features = cls._pipeline.extract_features(tensor, cls._model_backbone)
        features = cls._pipeline.flatten_features(features)

        return {
            "status": "features_extracted",
            "features_shape": list(features.shape),
            "features": features.tolist()
        }

    @classmethod
    def set_model(cls, model_backbone):
        """Charge le backbone CNN une fois le modèle disponible."""
        cls._model_backbone = model_backbone
