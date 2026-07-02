import os
import pickle
from tqdm import tqdm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

def extract_features(directory):
    """
    Charge le modèle InceptionV3 pré-entraîné, adapte la taille des images à 299x299,
    et extrait les caractéristiques (vecteurs de taille 2048).
    """
    # 1. Charger InceptionV3 sans la couche de classification (Top)
    # Inclure 'global_average_pooling2d' permet d'obtenir directement le vecteur de taille 2048
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    print("Modèle InceptionV3 chargé avec succès.")
    
    features = dict()
    
    # Lister toutes les images du dossier
    image_files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Extraction commencée pour {len(image_files)} images...")
    
    for name in tqdm(image_files):
        filename = os.path.join(directory, name)
        
        # InceptionV3 requiert des images de taille 299x299 (VGG demandait 224x224)
        image = load_img(filename, target_size=(299, 299))
        image = img_to_array(image)
        
        # Redimensionner pour ajouter la dimension du batch (1, 299, 299, 3)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        
        # Pré-traitement spécifique à InceptionV3 (normalisation des pixels entre -1 et 1)
        image = preprocess_input(image)
        
        # Extraction du vecteur de caractéristiques (1, 2048)
        feature = model.predict(image, verbose=0)
        
        # Stocker l'identifiant de l'image (sans l'extension .jpg)
        image_id = name.split('.')[0]
        features[image_id] = feature
        
    return features

if __name__ == "__main__":
    # Ajustez le chemin vers votre dossier d'images Flickr8k si nécessaire
    IMAGES_DIR = os.path.join("Data", "Flickr8k_Dataset") 
    OUTPUT_PICKLE = "features_inception.pkl"
    
    if not os.path.exists(IMAGES_DIR):
        print(f"Erreur : Le dossier d'images '{IMAGES_DIR}' est introuvable.")
        exit()
        
    # Lancement de l'extraction
    features_extracted = extract_features(IMAGES_DIR)
    print(f"Extraction terminée. Nombre total d'images traitées : {len(features_extracted)}")
    
    # Sauvegarde dans le fichier attendu par train.py
    with open(OUTPUT_PICKLE, 'wb') as f:
        pickle.dump(features_extracted, f)
        
    print(f"Fichier sauvegardé avec succès sous le nom '{OUTPUT_PICKLE}'. Vous pouvez maintenant lancer train.py !")