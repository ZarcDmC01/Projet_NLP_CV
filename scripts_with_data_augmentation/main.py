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
IMAGES_DIR_PATH       = "data/Flicker8k_Dataset"
TRAIN_SPLIT_PATH     = "data/Flickr_8k.trainImages.txt"
DEV_SPLIT_PATH       = "data/Flickr_8k.devImages.txt"

# Nouveau nom de cache : la structure change (liste de vecteurs par image
# au lieu d'un seul vecteur), et les paramètres d'augmentation ont changé
# (moins de versions, flip désactivé) — on veut donc repartir d'un cache
# neuf plutôt que de recharger silencieusement l'ancien.
FEATURES_PICKLE      = "data/features_resnet50_augmented_v2.pkl"
TARGET_IMAGE_SIZE    = (224, 224)
FEATURE_DIMENSION    = 2048

BATCH_SIZE = 64
EPOCHS = 50

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DE LA DATA AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
# Nombre de versions augmentées générées EN PLUS de l'image originale,
# pour chaque image du split d'entraînement uniquement (le split de
# validation reste "propre", sans augmentation, pour une évaluation fiable).
# Réduit à 2 : un ratio trop élevé de versions augmentées noie le signal
# propre (voir data_generator / original_feature_prob qui pondère aussi
# le tirage en faveur de l'original).
AUGMENTATIONS_PER_IMAGE = 2

# Patience un peu plus large que la version sans augmentation : la loss de
# validation peut fluctuer davantage le temps que le modèle absorbe la
# diversité supplémentaire.
EARLY_STOPPING_PATIENCE = 5


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


    print("\n=== ÉTAPE 2 : EXTRACTION DES CARACTÉRISTIQUES GRAPHIQUES (+ DATA AUGMENTATION) ===")
    img_pipe = ImageProcessingPipeline(target_size=TARGET_IMAGE_SIZE)

    if os.path.exists(FEATURES_PICKLE):
        print(f"Fichier de cache trouvé ('{FEATURES_PICKLE}'). Chargement des features...")
        image_features = img_pipe.load_features(FEATURES_PICKLE)
    else:
        print("Aucun cache trouvé. Initialisation du modèle ResNet50 pré-entraîné...")

        weights = models.ResNet50_Weights.DEFAULT
        resnet50 = models.resnet50(weights=weights)
        resnet50.fc = torch.nn.Identity()
        resnet50.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet50 = resnet50.to(device)
        print(f"Modèle de vision envoyé sur le périphérique : {device}")

        # Deux pipelines de prétraitement : un "propre" (toujours utilisé,
        # y compris pour produire la version de base de chaque image), et un
        # "augmenté" (utilisé uniquement pour générer des copies additionnelles
        # des images d'entraînement).
        base_transform = img_pipe.get_transform(augment=False)
        augmentation_transform = img_pipe.get_transform(augment=True)  # flip désactivé par défaut, voir processing.py

        def extract_one(img_pil, transform):
            tensor_img = transform(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                feature_vector = resnet50(tensor_img)
            return feature_vector.cpu().numpy().flatten()

        image_features = dict()
        all_files = os.listdir(IMAGES_DIR_PATH)

        print("Démarrage de l'extraction des caractéristiques d'images...")
        print(f"→ {AUGMENTATIONS_PER_IMAGE} version(s) augmentée(s) supplémentaire(s) par image "
              f"d'entraînement (aucune augmentation sur les images de validation/test).")

        for name in tqdm(all_files):
            image_id = name.split('.')[0]
            path = os.path.join(IMAGES_DIR_PATH, name)

            if not (os.path.isfile(path) and name.lower().endswith(('.png', '.jpg', '.jpeg'))):
                continue

            try:
                img_pil = Image.open(path).convert('RGB')

                # On stocke systématiquement une LISTE de vecteurs par image
                # (même si elle ne contient qu'un seul élément pour les images
                # hors train set), afin que la boucle d'entraînement puisse
                # piocher aléatoirement une version à chaque epoch.
                feature_list = [extract_one(img_pil, base_transform)]

                if image_id in train_ids:
                    for _ in range(AUGMENTATIONS_PER_IMAGE):
                        feature_list.append(extract_one(img_pil, augmentation_transform))

                image_features[image_id] = feature_list

            except Exception as e:
                print(f"Erreur lors du traitement de '{name}' : {e}")

        img_pipe.save_features(image_features, filename=FEATURES_PICKLE)

    # Filet de sécurité : si jamais un cache "ancien format" (un seul vecteur
    # par image, pas de liste) est chargé, on le normalise en liste à un
    # élément pour rester compatible avec le data_generator.
    image_features = {
        img_id: (feat if isinstance(feat, list) else [feat])
        for img_id, feat in image_features.items()
    }

    print(f"Vecteurs disponibles : {len(image_features)}")

    train_features = {
        img_id: feat
        for img_id, feat in image_features.items()
        if img_id in train_ids
    }
    print(f"Features d'entraînement après filtrage : {len(train_features)} images "
          f"({sum(len(v) for v in train_features.values())} vecteurs au total avec augmentation)")

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
        patience=EARLY_STOPPING_PATIENCE,
        checkpoint_path="data/checkpoint_best.keras"
    )

    print("Sauvegarde du modèle fusion entraîné...")
    train_pipe.save_model("data/flickr8k_caption_generator_resnet_data_aug.keras")
    print("=== PIPELINE DISQUE PRÊTE ET MODÈLE ENTRAÎNÉ AVEC SUCCÈS ===")


if __name__ == "__main__":
    main()