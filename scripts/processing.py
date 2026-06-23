import os
import string
import pickle
from collections import Counter
from tqdm import tqdm

# Configuration du backend conformément à votre environnement de calcul
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.src.legacy.preprocessing.text import Tokenizer

class TextProcessingPipeline:
    """
    Pipeline améliorée dédiée au chargement, nettoyage, filtrage par fréquence,
    balisage et tokenisation des légendes textuelles du dataset Flickr8k.
    """
    def __init__(self, start_token="startseq", end_token="endseq"):
        self.start_token = start_token
        self.end_token = end_token
        self.tokenizer = None
        self.max_length = None

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
            if len(line) < 2:
                continue
            # L'identifiant de l'image et l'index de la description
            image_id, image_desc = tokens[0], tokens[1:]
            # Nettoyage de l'extension pour garder uniquement la clé unique
            image_id = image_id.split('.')[0]
            # Reconstitution de la phrase brute
            desc_phrase = ' '.join(image_desc)
            
            if image_id not in mapping:
                mapping[image_id] = list()
            mapping[image_id].append(desc_phrase)
        return mapping

    def clean_descriptions_base(self, descriptions):
        """
        Étape 1 : Nettoyage syntaxique de base (Minuscules, ponctuation, bruits numériques).
        Renvoie un dictionnaire de phrases nettoyées MAIS sans les balises start/end.
        """
        cleaned_mapping = dict()
        table = str.maketrans('', '', string.punctuation)
        
        for key, desc_list in descriptions.items():
            if key not in cleaned_mapping:
                cleaned_mapping[key] = list()
            
            for desc in desc_list:
                # Tokenisation par mot pour le nettoyage individuel
                words = desc.split()
                # Passage en minuscules
                words = [word.lower() for word in words]
                # Suppression de la ponctuation
                words = [word.translate(table) for word in words]
                # Suppression des tokens isolés de moins de 2 lettres (ex: 'a') ou contenant des chiffres
                words = [word for word in words if len(word) > 1 and word.isalpha()]
                
                # Reconstitution de la phrase intermédiaire nettoyée
                cleaned_desc = ' '.join(words)
                cleaned_mapping[key].append(cleaned_desc)
                
        return cleaned_mapping

    def filter_and_finalize_descriptions(self, cleaned_descriptions, min_frequency=3):
        """
        Étape 2 & 3 : Analyse globale du corpus pour filtrer les mots rares sous 
        le seuil 'min_frequency' en les remplaçant par 'unk', puis injection finale 
        des balises de contrôle de manière sécurisée.
        """
        # 1. Comptage global de la fréquence de chaque mot dans tout le dataset
        word_counts = Counter()
        for desc_list in cleaned_descriptions.values():
            for desc in desc_list:
                word_counts.update(desc.split())

        # 2. Identification des mots rares et reconstruction finale
        final_mapping = dict()
        
        for key, desc_list in cleaned_descriptions.items():
            final_mapping[key] = list()
            for desc in desc_list:
                words = desc.split()
                # Remplacement des mots sous le seuil par le jeton générique 'unk'
                processed_words = [word if word_counts[word] >= min_frequency else 'unk' for word in words]
                
                # Ré-assemblage de la phrase finale purifiée
                processed_desc = ' '.join(processed_words)
                
                # Étape Finale : Encadrement strict par vos balises de contrôle
                caption_with_tokens = f"{self.start_token} {processed_desc} {self.end_token}"
                final_mapping[key].append(caption_with_tokens)
                
        return final_mapping

    def create_tokenizer(self, descriptions):
        """
        Entraîne l'outil de vectorisation Keras sur l'ensemble des textes 
        parfaitement nettoyés et balisés.
        """
        lines = []
        for key in descriptions.keys():
            for desc in descriptions[key]:
                lines.append(desc)
                
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(lines)
        return self.tokenizer

    def calculate_max_length(self, descriptions):
        """
        Calcule la longueur maximale (en nombre de mots) présente dans le dataset.
        """
        max_len = 0
        for key in descriptions.keys():
            for desc in descriptions[key]:
                max_len = max(max_len, len(desc.split()))
        self.max_length = max_len
        return max_len


class ImageProcessingPipeline:
    """
    Pipeline dédiée au chargement et au traitement géométrique/numérique des images.
    Permet de définir les dimensions cibles pour s'adapter à n'importe quel backbone CNN.
    """
    def __init__(self, target_size=(224, 224)):
        """
        Initialise la pipeline visuelle.
        :param target_size: Tuple (hauteur, largeur). Par exemple (224, 224) pour VGG16/ResNet50 
                            ou (299, 299) pour InceptionV3.
        """
        self.target_size = target_size

    def preprocess_single_image(self, image_path):
        """
        Charge une image isolée, l'ajuste aux dimensions spécifiées, 
        et la convertit en tenseur normalisé prêt pour l'inférence.
        """
        img = keras.utils.load_img(image_path, target_size=self.target_size)
        img_array = keras.utils.img_to_array(img)
        
        # Ajout de la dimension de batch (1, H, W, C)
        img_array = img_array.reshape((1,) + img_array.shape)
        
        # Normalisation standard [0, 1]
        img_array = img_array / 255.0
        return img_array

    def extract_features_batch(self, directory_path, model_backbone, valid_ids=None):
        """
        Parcourt un répertoire d'images, applique le traitement numérique, 
        et extrait les features en masse via un modèle de Transfer Learning fourni.
        
        :param directory_path: Chemin vers le dossier contenant les images Flickr8k.
        :param model_backbone: Modèle Keras sans sa tête de classification.
        :param valid_ids: Optionnel. Liste ou set d'IDs à filtrer (ex: uniquement le train set).
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
                
                # Extraction via le réseau convolutif
                feature_vector = model_backbone.predict(prepared_img, verbose=0)
                
                # Sauvegarde avec aplatissement du vecteur (ex: de (1, 4096) à (4096,))
                features[image_id] = feature_vector.reshape(-1)
                
        return features

    @staticmethod
    def save_features(features_dict, filename="features.pkl"):
        """Sauvegarde les descripteurs extraits dans un fichier binaire sérialisé."""
        with open(filename, 'wb') as f:
            pickle.dump(features_dict, f)

    @staticmethod
    def load_features(filename="features.pkl"):
        """Charge les descripteurs stockés en mémoire locale."""
        with open(filename, 'rb') as f:
            return pickle.load(f)