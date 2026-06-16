import os
import string
import pickle
from tqdm import tqdm

# Configuration du backend conformément à votre environnement de calcul
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.src.legacy.preprocessing.text import Tokenizer

class TextProcessingPipeline:
    """
    Pipeline dédiée au chargement, nettoyage, tokenisation et vectorisation 
    des légendes textuelles (captions) du dataset Flickr8k.
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
            # Reconstitution de la chaîne de texte
            image_desc = ' '.join(image_desc)
            
            if image_id not in mapping:
                mapping[image_id] = list()
            mapping[image_id].append(image_desc)
        return mapping

    def clean_descriptions(self, descriptions_dict):
        """
        Prépare et nettoie l'intégralité du texte : passage en minuscules, 
        retrait de la ponctuation, élimination des tokens isolés ou numériques.
        """
        table = str.maketrans('', '', string.punctuation)
        cleaned_dict = dict()
        
        for key, desc_list in descriptions_dict.items():
            cleaned_list = list()
            for desc in desc_list:
                words = desc.split()
                words = [word.lower() for word in words]
                words = [word.translate(table) for word in words]
                words = [word for word in words if len(word) > 1 and word.isalpha()]
                
                # Encapsulation par les jetons de contrôle de séquence
                cleaned_desc = f"{self.start_token} {' '.join(words)} {self.end_token}"
                cleaned_list.append(cleaned_desc)
            cleaned_dict[key] = cleaned_list
        return cleaned_dict

    def _to_list(self, descriptions_dict):
        """Méthode interne pour aplatir le dictionnaire en une liste de chaînes."""
        all_desc = list()
        for key in descriptions_dict.keys():
            [all_desc.append(d) for d in descriptions_dict[key]]
        return all_desc

    def fit_tokenizer(self, train_descriptions):
        """
        Entraîne le Tokenizer Keras sur le sous-ensemble de descriptions 
        destiné à l'entraînement du modèle (Train Set).
        """
        lines = self._to_list(train_descriptions)
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(lines)
        
        # Calcul de la longueur maximale d'une séquence pour le padding futur
        self.max_length = max(len(d.split()) for d in lines)
        return self.tokenizer

    def save_pipeline(self, tokenizer_path="tokenizer.pkl", metadata_path="text_meta.pkl"):
        """Sérialise l'état de la pipeline pour une réutilisation ultérieure."""
        if self.tokenizer is None:
            raise ValueError("Le tokenizer doit d'abord être entraîné avec 'fit_tokenizer'.")
        
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
            
        meta = {"max_length": self.max_length, "start_token": self.start_token, "end_token": self.end_token}
        with open(metadata_path, 'wb') as f:
            pickle.dump(meta, f)

    def load_pipeline(self, tokenizer_path="tokenizer.pkl", metadata_path="text_meta.pkl"):
        """Charge un état pré-enregistré de la pipeline textuelle."""
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open(metadata_path, 'rb') as f:
            meta = pickle.load(f)
            self.max_length = meta["max_length"]
            self.start_token = meta["start_token"]
            self.end_token = meta["end_token"]


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