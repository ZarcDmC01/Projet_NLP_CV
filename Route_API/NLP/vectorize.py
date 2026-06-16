import pickle
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
            image_id, image_desc = tokens[0], tokens[1:]
            image_id = image_id.split('.')[0]
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
        import string
        table = str.maketrans('', '', string.punctuation)
        cleaned_dict = dict()

        for key, desc_list in descriptions_dict.items():
            cleaned_list = list()
            for desc in desc_list:
                words = desc.split()
                words = [word.lower() for word in words]
                words = [word.translate(table) for word in words]
                words = [word for word in words if len(word) > 1 and word.isalpha()]
                cleaned_desc = f"{self.start_token} {' '.join(words)} {self.end_token}"
                cleaned_list.append(cleaned_desc)
            cleaned_dict[key] = cleaned_list
        return cleaned_dict

    def _to_list(self, descriptions_dict):
        """Aplatit le dictionnaire en une liste de chaînes."""
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
