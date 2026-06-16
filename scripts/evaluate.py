import os
import pickle
import numpy as np
from PIL import Image

# Forcer le backend Torch pour Keras
os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Importation pour la traduction automatique (Hugging Face)
from transformers import Pipeline, pipeline

class CaptionEvaluator:
    """
    Classe chargée de l'inférence, de la génération par Beam Search
    et de la traduction des légendes générées.
    """
    def __init__(self, model_path, tokenizer_path, metadata_path):
        # 1. Chargement du modèle de fusion entraîné
        print("Chargement du modèle de génération de légendes...")
        self.model = keras.models.load_model(model_path)
        
        # 2. Chargement du Tokenizer et des métadonnées textuelles
        print("Chargement du tokenizer...")
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
            
        with open(metadata_path, 'rb') as f:
            meta = pickle.load(f)
            self.max_length = meta["max_length"]
            self.start_token = meta["start_token"]
            self.end_token = meta["end_token"]
            
        # Création du dictionnaire inverse (index -> mot) pour décoder les prédictions
        self.index_to_word = {idx: word for word, idx in self.tokenizer.word_index.items()}
        
        # 3. Initialisation du pipeline de traduction anglais -> français
        print("Chargement du modèle de traduction NLP (En -> Fr)...")
        self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

    def generate_caption_beam_search(self, image_feature, beam_width=3):
        """
        Génère une légende en utilisant l'algorithme Beam Search (Recherche par faisceau).
        """
        # Encode le mot de départ 'startseq'
        start_seq = self.tokenizer.texts_to_sequences([self.start_token])[0]
        
        # Structure d'un faisceau (beam) : [séquence_de_mots, score_probabilité_cumulé]
        sequences = [[start_seq, 0.0]]
        
        # Redimensionnement de la feature pour le modèle (1, 2048)
        image_feature = np.array([image_feature])
        
        for _ in range(self.max_length):
            all_candidates = list()
            
            # Explorer chaque faisceau actif
            for seq, score in sequences:
                # Si le faisceau a déjà atteint le token de fin, on le garde tel quel
                if seq[-1] == self.tokenizer.word_index.get(self.end_token):
                    all_candidates.append([seq, score])
                    continue
                
                # Padding de la séquence actuelle
                padded_seq = keras.utils.pad_sequences([seq], maxlen=self.max_length)
                
                # Prédiction des probabilités du mot suivant
                predictions = self.model.predict([image_feature, padded_seq], verbose=0)[0]
                
                # Sélection des meilleures probabilités (top B mots)
                best_word_indices = np.argsort(predictions)[-beam_width:]
                
                # Créer un nouveau candidat pour chaque mot sélectionné
                for idx in best_word_indices:
                    next_seq = list(seq) + [idx]
                    # Log-probabilité pour éviter le sous-tirage numérique (underflow)
                    next_score = score - np.log(predictions[idx] + 1e-10)
                    all_candidates.append([next_seq, next_score])
            
            # Tri de tous les candidats par le score le plus bas (probabilité la plus haute)
            ordered = sorted(all_candidates, key=lambda x: x[1])
            
            # On ne garde que les 'beam_width' meilleurs faisceaux pour l'itération suivante
            sequences = ordered[:beam_width]
            
            # Si tous les faisceaux actifs ont généré 'endseq', on peut s'arrêter
            if all(seq[-1] == self.tokenizer.word_index.get(self.end_token) for seq, _ in sequences):
                break
                
        # Extraction de la meilleure séquence globale
        best_seq = sequences[0][0]
        
        # Décodage des indices en mots textuels
        generated_words = [self.index_to_word.get(idx) for idx in best_seq if idx in self.index_to_word]
        
        # Nettoyage des balises de contrôle pour la phrase finale
        final_caption = [word for word in generated_words if word not in [self.start_token, self.end_token]]
        return ' '.join(final_caption)

    def translate_to_french(self, text_en):
        """Traduit la légende anglaise en français."""
        translation = self.translator(text_en, max_length=40)
        return translation[0]['translation_text']


def main():
    # --- CONFIGURATION DYNAMIQUE DES CHEMINS ---
    SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
    
    MODEL_PATH = os.path.join(ROOT_DIR, "flickr8k_caption_generator_resnet.keras")
    TOKENIZER_PATH = os.path.join(ROOT_DIR, "tokenizer.pkl")
    METADATA_PATH = os.path.join(ROOT_DIR, "text_meta.pkl")
    FEATURES_PATH = os.path.join(ROOT_DIR, "data", "features_resnet50.pkl")
    IMAGES_DIR = os.path.join(ROOT_DIR, "data", "Flickr8k_Dataset")

    # --- INITIALISATION ---
    evaluator = CaptionEvaluator(MODEL_PATH, TOKENIZER_PATH, METADATA_PATH)
    
    # Chargement du dictionnaire des caractéristiques d'images (features cache)
    print("Chargement des features d'images pré-extraites...")
    with open(FEATURES_PATH, 'rb') as f:
        image_features = pickle.load(f)

    # --- TEST SUR QUELQUES IMAGES DU JEU DE DONNÉES ---
    # Nous sélectionnons 3 images au hasard dans le dictionnaire pour valider le modèle
    import random
    sample_image_ids = random.sample(list(image_features.keys()), 3)
    
    print("\n" + "="*50)
    print("DÉMARRAGE DES TESTS QUALITATIFS")
    print("="*50)
    
    for img_id in sample_image_ids:
        print(f"\nID de l'image : {img_id}.jpg")
        
        # Récupération de la feature vectorielle (2048,)
        feature = image_features[img_id]
        
        # 1. Génération de la légende en Anglais via Beam Search (B=3)
        caption_en = evaluator.generate_caption_beam_search(feature, beam_width=3)
        print(f"-> Légende Générée (EN) : {caption_en}")
        
        # 2. Traduction de la légende en Français
        caption_fr = evaluator.translate_to_french(caption_en)
        print(f"-> Traduction (FR)      : {caption_fr}")

if __name__ == "__main__":
    main()