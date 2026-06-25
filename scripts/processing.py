import os
import string
import pickle
from collections import Counter
from tqdm import tqdm

os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.src.legacy.preprocessing.text import Tokenizer


class TextProcessingPipeline:
    """
    Pipeline dédiée au chargement, nettoyage, filtrage par fréquence,
    balisage et tokenisation des légendes textuelles du dataset Flickr8k.
    Inclut la sauvegarde et le chargement de l'état complet de la pipeline
    pour une réutilisation en inférence.
    """

    def __init__(self, start_token="startseq", end_token="endseq"):
        self.start_token = start_token
        self.end_token = end_token
        self.tokenizer = None
        self.max_length = None

    # ─────────────────────────────────────────────
    # ÉTAPE 1 : Chargement brut
    # ─────────────────────────────────────────────

    def load_raw_descriptions(self, filename):
        """
        Charge le fichier de tokens brut et associe à chaque identifiant d'image
        une liste contenant ses 5 descriptions associées.
        """
        with open(filename, 'r', encoding='utf-8') as file:
            doc = file.read()

        mapping = dict()
        for line in doc.split('\n'):
            tokens = line.split()
            if len(tokens) < 2:
                continue
            image_id, image_desc = tokens[0], tokens[1:]
            image_id = image_id.split('.')[0]
            desc_phrase = ' '.join(image_desc)

            if image_id not in mapping:
                mapping[image_id] = list()
            mapping[image_id].append(desc_phrase)
        return mapping

    # ─────────────────────────────────────────────
    # ÉTAPE 2 : Nettoyage syntaxique de base
    # ─────────────────────────────────────────────

    def clean_descriptions_base(self, descriptions):
        """
        Nettoyage syntaxique de base : minuscules, ponctuation, bruits numériques.
        Renvoie un dictionnaire de phrases nettoyées SANS les balises start/end.
        """
        cleaned_mapping = dict()
        table = str.maketrans('', '', string.punctuation)

        for key, desc_list in descriptions.items():
            cleaned_mapping[key] = list()
            for desc in desc_list:
                words = desc.split()
                words = [word.lower() for word in words]
                words = [word.translate(table) for word in words]
                words = [
                    word for word in words
                    if word.isalpha() and (len(word) > 1 or word in ['a', 'i'])
                ]
                cleaned_mapping[key].append(' '.join(words))

        return cleaned_mapping

    # ─────────────────────────────────────────────
    # ÉTAPE 3 : Filtrage des mots rares + balisage
    # ─────────────────────────────────────────────

    def filter_and_finalize_descriptions(self, cleaned_descriptions, min_frequency=3, valid_ids=None):
        word_counts = Counter()
        for key, desc_list in cleaned_descriptions.items():
            if valid_ids is not None and key not in valid_ids:
                continue
            for desc in desc_list:
                word_counts.update(desc.split())

        final_mapping = dict()
        for key, desc_list in cleaned_descriptions.items():
            if valid_ids is not None and key not in valid_ids:
                continue
            final_mapping[key] = list()
            for desc in desc_list:
                words = desc.split()
                processed_words = [
                    word if word_counts[word] >= min_frequency else 'unk'
                    for word in words
                ]
                processed_desc = ' '.join(processed_words)
                caption_with_tokens = f"{self.start_token} {processed_desc} {self.end_token}"
                final_mapping[key].append(caption_with_tokens)

        return final_mapping

    # ─────────────────────────────────────────────
    # Wrapper pratique : brut → nettoyé → balisé
    # ─────────────────────────────────────────────

    def clean_descriptions(self, raw_descriptions, min_frequency=3):
        """
        Wrapper complet : applique clean_descriptions_base puis
        filter_and_finalize_descriptions en un seul appel.
        Utilisé notamment dans le notebook d'évaluation.
        """
        base_cleaned = self.clean_descriptions_base(raw_descriptions)
        return self.filter_and_finalize_descriptions(base_cleaned, min_frequency=min_frequency)

    # ─────────────────────────────────────────────
    # Tokenisation
    # ─────────────────────────────────────────────

    def create_tokenizer(self, descriptions):
        """
        Entraîne le Tokenizer Keras sur l'ensemble des textes nettoyés et balisés.
        """
        lines = [
            desc
            for desc_list in descriptions.values()
            for desc in desc_list
        ]
        self.tokenizer = Tokenizer(oov_token='unk')
        self.tokenizer.fit_on_texts(lines)
        return self.tokenizer

    def calculate_max_length(self, descriptions):
        """
        Calcule la longueur maximale (en nombre de mots) présente dans le dataset.
        """
        max_len = max(
            len(desc.split())
            for desc_list in descriptions.values()
            for desc in desc_list
        )
        self.max_length = max_len
        return max_len

    # ─────────────────────────────────────────────
    # Persistance de la pipeline
    # ─────────────────────────────────────────────

    def save_pipeline(self,
                      tokenizer_path="tokenizer.pkl",
                      metadata_path="text_meta.pkl"):
        """
        Sauvegarde le tokenizer et les métadonnées (max_length, tokens de contrôle)
        dans deux fichiers pickle distincts, nécessaires pour l'inférence.
        """
        if self.tokenizer is None:
            raise ValueError("Le tokenizer n'a pas encore été créé. Appelez create_tokenizer() d'abord.")
        if self.max_length is None:
            raise ValueError("max_length n'est pas défini. Appelez calculate_max_length() d'abord.")

        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

        metadata = {
            'max_length': self.max_length,
            'start_token': self.start_token,
            'end_token': self.end_token,
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Pipeline sauvegardée → tokenizer : '{tokenizer_path}' | métadonnées : '{metadata_path}'")

    def load_pipeline(self,
                      tokenizer_path="tokenizer.pkl",
                      metadata_path="text_meta.pkl"):
        """
        Recharge le tokenizer et les métadonnées depuis les fichiers pickle.
        Indispensable en inférence pour reproduire exactement le même encodage.
        """
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        self.max_length = metadata['max_length']
        self.start_token = metadata['start_token']
        self.end_token = metadata['end_token']

        print(f"Pipeline chargée → vocab : {len(self.tokenizer.word_index)} mots | max_length : {self.max_length}")


class ImageProcessingPipeline:
    """
    Pipeline dédiée au chargement et au traitement géométrique/numérique des images.
    Permet de définir les dimensions cibles pour s'adapter à n'importe quel backbone CNN.
    """

    def __init__(self, target_size=(224, 224)):
        """
        :param target_size: Tuple (hauteur, largeur).
                            (224, 224) pour VGG16 / ResNet50,
                            (299, 299) pour InceptionV3.
        """
        self.target_size = target_size


    def preprocess_single_image(self, image_path):
        """
        Charge une image, l'ajuste, et applique la normalisation standard 
        ImageNet requise par ResNet50 (PyTorch).
        """
        import torch
        import torchvision.transforms as transforms
        from PIL import Image

        img = Image.open(image_path).convert('RGB')

        preprocess = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        tensor_img = preprocess(img).unsqueeze(0)
        return tensor_img


    def extract_features_batch(self, directory_path, model_backbone, valid_ids=None):
        """
        Parcourt un répertoire d'images, applique le traitement numérique,
        et extrait les features en masse via un modèle de Transfer Learning fourni.

        :param directory_path: Chemin vers le dossier contenant les images.
        :param model_backbone: Modèle Keras sans tête de classification.
        :param valid_ids: Optionnel. Set d'IDs à filtrer.
        """
        features = dict()
        all_files = os.listdir(directory_path)

        print(f"Extraction des caractéristiques graphiques (Taille cible : {self.target_size})...")
        for name in tqdm(all_files):
            image_id = name.split('.')[0]

            if valid_ids is not None and image_id not in valid_ids:
                continue

            path = os.path.join(directory_path, name)
            if os.path.isfile(path) and name.lower().endswith(('.png', '.jpg', '.jpeg')):
                prepared_img = self.preprocess_single_image(path)
                feature_vector = model_backbone.predict(prepared_img, verbose=0)
                features[image_id] = feature_vector.reshape(-1)
        return features

    @staticmethod
    def save_features(features_dict, filename="features.pkl"):
        """Sauvegarde les descripteurs extraits dans un fichier binaire sérialisé."""
        with open(filename, 'wb') as f:
            pickle.dump(features_dict, f)
        print(f"Features sauvegardées → '{filename}' ({len(features_dict)} images)")

    @staticmethod
    def load_features(filename="features.pkl"):
        """Charge les descripteurs stockés en mémoire locale."""
        with open(filename, 'rb') as f:
            return pickle.load(f)