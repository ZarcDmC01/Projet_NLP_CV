import os
import string
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model

def extract_image_features(directory_path):
    """
    Étape 2 : Préparer les données photo.
    Parcourt récursivement le dossier pour trouver toutes les images valides,
    en ignorant les fichiers système indésirables (comme __MACOSX ou ._ files).
    """
    print(f" Analyse et extraction des caractéristiques depuis : '{os.path.abspath(directory_path)}'...")
    
    if not os.path.exists(directory_path):
        print(f" Erreur : Le dossier '{directory_path}' n'existe pas.")
        return {}

    # Charger le modèle VGG16 (Transfer Learning)
    base_model = VGG16()
    model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    
    features = {}
    
    # os.walk permet de chercher dans TOUS les sous-dossiers automatiquement
    for root, dirs, files in os.walk(directory_path):
        # Sécurité : ignorer les dossiers Mac cachés pour gagner du temps
        if "__MACOSX" in root:
            continue
            
        for img_name in files:
            # On vérifie l'extension et on élimine les fichiers fantômes cachés (commençant par ._)
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')) and not img_name.startswith('._'):
                img_path = os.path.join(root, img_name)
                
                try:
                    # Traitement de l'image pour VGG16 (224x224)
                    image = load_img(img_path, target_size=(224, 224))
                    image = img_to_array(image)
                    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                    image = preprocess_input(image)
                    
                    # Extraction du vecteur (4096,)
                    feature = model.predict(image, verbose=0)
                    
                    # L'identifiant de l'image correspond au nom du fichier sans extension
                    image_id = img_name.split('.')[0]
                    features[image_id] = feature
                    
                    # Petit indicateur visuel de progression tous les 500 fichiers
                    if len(features) % 500 == 0:
                        print(f"📸 Progression : {len(features)} images analysées...")
                        
                except Exception as e:
                    # Permet de ne pas bloquer le script si une image est corrompue
                    continue
            
    print(f"Extraction terminée ! {len(features)} images réelles ont été traitées.")
    return features


def load_doc(filename):
    """Ouvre un fichier texte et retourne son contenu."""
    if not os.path.exists(filename):
        print(f" Erreur : Le fichier '{os.path.abspath(filename)}' est introuvable.")
        return None
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def load_descriptions(doc):
    """Associe chaque identifiant d'image à sa liste de descriptions."""
    mapping = {}
    for line in doc.split('\n'):
        tokens = line.split('\t')
        if len(line) < 2 or len(tokens) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1]
        image_id = image_id.split('.')[0]
        
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(image_desc)
    return mapping


def clean_descriptions(descriptions_dict):
    """Nettoie le texte (minuscules, ponctuation, balises start/end)."""
    table = str.maketrans('', '', string.punctuation)
    cleaned_desc = {}
    
    for img_id, desc_list in descriptions_dict.items():
        if img_id not in cleaned_desc:
            cleaned_desc[img_id] = []
        for desc in desc_list:
            desc = desc.lower()
            desc = desc.translate(table)
            words = desc.split()
            words = [word for word in words if word.isalpha()]
            
            cleaned_sentence = 'startseq ' + ' '.join(words) + ' endseq'
            cleaned_desc[img_id].append(cleaned_sentence)
            
    return cleaned_desc


def create_and_save_tokenizer(descriptions_dict, filepath="tokenzier.pkl"):
    """Crée le tokenizer et le sauvegarde au format pickle."""
    all_desc = []
    for img_id in descriptions_dict:
        for desc in descriptions_dict[img_id]:
            all_desc.append(desc)
            
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_desc)
    
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f" Tokenizer sauvegardé avec succès dans '{os.path.abspath(filepath)}'")
    
    return tokenizer


# --- BLOC D'EXÉCUTION ---
if __name__ == "__main__":
    print("=== DÉMARRAGE DU PRÉTRAITEMENT DES DONNÉES ===")
    
    # Chemins relatifs dynamiques (recherche automatique depuis C:\Mes_projets\Projet CNN&NLP\Projet CNN&NLP)
    ROOT_DATA_DIR = os.path.join("Data", "Flickr8k_Dataset")
    TEXT_FILE = os.path.join("Data", "Flickr8k_text", "Flickr8k.token.txt")
    
    # 1. Pipeline Texte
    doc = load_doc(TEXT_FILE)
    if doc:
        raw_descriptions = load_descriptions(doc)
        cleaned_descriptions = clean_descriptions(raw_descriptions)
        print(f" {len(cleaned_descriptions)} descriptions textuelles nettoyées.")
        
        tokenizer = create_and_save_tokenizer(cleaned_descriptions, filepath="tokenzier.pkl")
        vocab_size = len(tokenizer.word_index) + 1
        print(f" Taille du vocabulaire : {vocab_size} mots uniques.")
        
    # 2. Pipeline Image (Scan récursif robuste dans Data/Flickr8k_Dataset)
    features = extract_image_features(ROOT_DATA_DIR)
    
    if len(features) > 0:
        with open("features.pkl", "wb") as f:
            pickle.dump(features, f)
        print(f"Fichier 'features.pkl' sauvegardé avec succès dans '{os.path.abspath('features.pkl')}' !")
    else:
        print(" Échec : Aucune image n'a pu être traitée. Vérifiez la présence de vos images dans 'Data/Flickr8k_Dataset'.")