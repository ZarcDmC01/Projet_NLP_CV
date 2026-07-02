import os
import pickle
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from processing import TextProcessingPipeline, ImageProcessingPipeline
from training import CaptionTrainingPipeline


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRE : Chargement des splits officiels Flickr8k
# ─────────────────────────────────────────────────────────────────────────────

def load_image_ids(filename):
    """
    Charge la liste des identifiants d'images depuis un fichier de split officiel
    (Flickr_8k.trainImages.txt, devImages.txt, testImages.txt).
    Retourne un set d'identifiants sans extension (.jpg).
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.read().strip().splitlines()
    return set(line.split('.')[0] for line in lines if line.strip())


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION CENTRALE
# ─────────────────────────────────────────────────────────────────────────────

TOKEN_FILE_PATH      = "data/Flickr8k.token.txt"
IMAGES_DIR_PATH      = "data/Flicker8k_Dataset"
TRAIN_SPLIT_PATH     = "data/Flickr_8k.trainImages.txt"
DEV_SPLIT_PATH       = "data/Flickr_8k.devImages.txt"

# ─────────────────────────────────────────────────────────────────────────────
# CHOIX DU BACKBONE DE VISION (Transfer Learning)
# ─────────────────────────────────────────────────────────────────────────────
# Changez simplement cette valeur pour basculer entre les backbones déjà
# configurés ci-dessous : "resnet50" ou "inceptionv3".
BACKBONE = "inceptionv3"

# Chaque backbone a sa propre taille d'image cible, sa dimension de sortie,
# son fichier de cache de features et son nom de modèle final, afin de ne
# jamais écraser les résultats déjà obtenus avec ResNet50.
BACKBONE_CONFIGS = {
    "resnet50": {
        "target_size": (224, 224),
        "feature_dim": 2048,
        "features_pickle": "data/features_resnet50.pkl",
        "model_output": "data/flickr8k_caption_generator_resnet4.keras",
        "checkpoint": "data/checkpoint_best_resnet50.keras",
    },
    "inceptionv3": {
        "target_size": (299, 299),
        "feature_dim": 2048,
        "features_pickle": "data/features_inceptionv3.pkl",
        "model_output": "data/flickr8k_caption_generator_inceptionv3.keras",
        "checkpoint": "data/checkpoint_best_inceptionv3.keras",
    },
}

_active_config       = BACKBONE_CONFIGS[BACKBONE]
FEATURES_PICKLE      = _active_config["features_pickle"]
TARGET_IMAGE_SIZE    = _active_config["target_size"]
FEATURE_DIMENSION    = _active_config["feature_dim"]
MODEL_OUTPUT_PATH    = _active_config["model_output"]
CHECKPOINT_PATH      = _active_config["checkpoint"]

BATCH_SIZE = 64
EPOCHS = 50


def build_vision_backbone(name, device):
    """
    Instancie le backbone CNN pré-entraîné demandé (tête de classification
    retirée) ainsi que le pipeline de pré-traitement torchvision associé.
    Retourne (model, preprocess_transform).
    """
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()

    elif name == "inceptionv3":
        weights = models.Inception_V3_Weights.DEFAULT
        # aux_logits=True est requis pour charger les poids pré-entraînés ;
        # on désactive ensuite la branche auxiliaire (utile seulement pour
        # entraîner le classifieur d'origine, inutile en extraction de
        # features).
        model = models.inception_v3(weights=weights, aux_logits=True)
        model.aux_logits = False
        model.AuxLogits = None
        model.fc = torch.nn.Identity()

    else:
        raise ValueError(f"Backbone inconnu : '{name}'. Choix possibles : {list(BACKBONE_CONFIGS)}")

    model.eval()
    model = model.to(device)

    preprocess_transform = transforms.Compose([
        transforms.Resize(TARGET_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return model, preprocess_transform


def main():
    print("=== ÉTAPE 1 : TRAITEMENT TEXTUEL ===")
    text_pipe = TextProcessingPipeline()

    print("Chargement des descriptions brutes...")
    descriptions = text_pipe.load_raw_descriptions(TOKEN_FILE_PATH)
    print(f"Nombre de photos avec descriptions chargées : {len(descriptions)}")

    print("Chargement des IDs officiels du split d'entraînement...")
    if not os.path.exists(TRAIN_SPLIT_PATH):
        raise FileNotFoundError(f"Le fichier de split d'entraînement '{TRAIN_SPLIT_PATH}' est introuvable.")
    train_ids = load_image_ids(TRAIN_SPLIT_PATH)
    print(f"Nombre d'images dans le split d'entraînement : {len(train_ids)}")
    print("Chargement des IDs officiels du split de validation (dev)...")
    if not os.path.exists(DEV_SPLIT_PATH):
        raise FileNotFoundError(f"Le fichier de split de validation '{DEV_SPLIT_PATH}' est introuvable.")
    dev_ids = load_image_ids(DEV_SPLIT_PATH)
    print(f"Nombre d'images dans le split de validation : {len(dev_ids)}")

    print("Nettoyage syntaxique de base de toutes les légendes...")
    descriptions = text_pipe.clean_descriptions_base(descriptions)

    print("Filtrage par fréquence globale et finalisation (uniquement sur le train set)...")
    train_descriptions = text_pipe.filter_and_finalize_descriptions(
        cleaned_descriptions=descriptions,
        valid_ids=train_ids,
        min_frequency=2
    )
    print("Préparation des descriptions de validation (mêmes règles, sans re-fit du tokenizer)...")
    dev_descriptions = text_pipe.filter_and_finalize_descriptions(
        cleaned_descriptions=descriptions,
        valid_ids=dev_ids,
        min_frequency=2
    )

    print("Création et ajustement du Tokenizer Keras...")
    tokenizer = text_pipe.create_tokenizer(train_descriptions)
    vocab_size = tokenizer.num_words + 1
    max_len = text_pipe.calculate_max_length(train_descriptions)

    print(f"→ Taille finale du vocabulaire retenu : {vocab_size}")
    print(f"→ Longueur maximale d'une phrase (tokens inclus) : {max_len}")

    print("Sauvegarde de la pipeline textuelle pour l'inférence future...")
    text_pipe.save_pipeline(tokenizer_path="data/tokenizer.pkl", metadata_path="data/text_meta.pkl")


    print("\n=== ÉTAPE 2 : EXTRACTION DES CARACTÉRISTIQUES GRAPHIQUES ===")
    img_pipe = ImageProcessingPipeline(target_size=TARGET_IMAGE_SIZE)

    if os.path.exists(FEATURES_PICKLE):
        print(f"Fichier de cache trouvé ('{FEATURES_PICKLE}'). Chargement des features...")
        image_features = img_pipe.load_features(FEATURES_PICKLE)
    else:
        print(f"Aucun cache trouvé. Initialisation du modèle '{BACKBONE}' pré-entraîné...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vision_model, preprocess_transform = build_vision_backbone(BACKBONE, device)
        print(f"Modèle de vision envoyé sur le périphérique : {device}")

        image_features = dict()
        all_files = os.listdir(IMAGES_DIR_PATH)

        print("Démarrage de l'extraction des caractéristiques d'images...")
        for name in tqdm(all_files):
            image_id = name.split('.')[0]
            path = os.path.join(IMAGES_DIR_PATH, name)

            if os.path.isfile(path) and name.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_pil = Image.open(path).convert('RGB')
                    
                    tensor_img = preprocess_transform(img_pil).unsqueeze(0).to(device)

                    with torch.no_grad():
                        feature_vector = vision_model(tensor_img)

                    image_features[image_id] = feature_vector.cpu().numpy().flatten()

                except Exception as e:
                    print(f"Erreur lors du traitement de '{name}' : {e}")

        img_pipe.save_features(image_features, filename=FEATURES_PICKLE)

    print(f"Vecteurs disponibles : {len(image_features)}")

    train_features = {
        img_id: feat
        for img_id, feat in image_features.items()
        if img_id in train_ids
    }
    print(f"Features d'entraînement après filtrage : {len(train_features)}")

    dev_features = {
        img_id: feat
        for img_id, feat in image_features.items()
        if img_id in dev_ids
    }
    print(f"Features de validation après filtrage : {len(dev_features)}")

    print("\n=== ÉTAPE 3 : ENTRAÎNEMENT DU MODÈLE FUSION ===")

    train_pipe = CaptionTrainingPipeline(
        vocab_size=vocab_size,
        max_length=max_len,
        feature_dim=FEATURE_DIMENSION
    )

    print("Démarrage de l'entraînement...")
    trained_model = train_pipe.fit(
        train_descriptions=train_descriptions,
        train_features=train_features,
        tokenizer=tokenizer,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        val_descriptions=dev_descriptions,
        val_features=dev_features,
        patience=3,
        checkpoint_path=CHECKPOINT_PATH
    )

    print("Sauvegarde du modèle fusion entraîné...")
    train_pipe.save_model(MODEL_OUTPUT_PATH)
    print("=== PIPELINE DISQUE PRÊTE ET MODÈLE ENTRAÎNÉ AVEC SUCCÈS ===")


if __name__ == "__main__":
    main()