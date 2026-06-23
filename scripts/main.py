import os
import pickle
os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Importation des pipelines depuis vos fichiers de modules respectifs
from processing import TextProcessingPipeline, ImageProcessingPipeline
from training import CaptionTrainingPipeline

def main():
    # --- CONFIGURATION DES CHEMINS ---
    TOKEN_FILE_PATH = "data/Flickr8k.token.txt"      # Fichier texte contenant les légendes
    IMAGES_DIR_PATH = "data/Flicker8k_Dataset"        # Dossier contenant toutes les images (.jpg)
    
    # AJUSTEMENT POUR RESNET50 :
    # Les features de ResNet50 font 2048 dimensions au lieu de 4096 pour VGG16
    FEATURES_PICKLE = "data/features_inceptionv3.pkl"   
    TARGET_IMAGE_SIZE = (299, 299) 
    FEATURE_DIMENSION = 2048                         # Changé de 4096 à 2048
    
    # Hyperparamètres d'entraînement
    BATCH_SIZE = 64
    EPOCHS = 10

    print("=== ÉTAPE 1 : TRAITEMENT TEXTUEL ===")
    text_pipe = TextProcessingPipeline()

    print("Chargement des descriptions brutes...")
    # Optionnel : Vous pouvez passer 'data/Flickr8k.lemma.token.txt' ici si vous voulez tester la version lemmatisée
    raw_desc = text_pipe.load_raw_descriptions("data/Flickr8k.token.txt") 

    print("Nettoyage syntaxique de base...")
    base_cleaned_desc = text_pipe.clean_descriptions_base(raw_desc)

    print("Filtrage global des mots rares (Seuil: 3) et injection des balises...")
    cleaned_desc = text_pipe.filter_and_finalize_descriptions(base_cleaned_desc, min_frequency=3)

    print("Création du dictionnaire et indexation (Tokenizer)...")
    tokenizer = text_pipe.create_tokenizer(cleaned_desc)
    vocab_size = len(tokenizer.word_index) + 1

    max_len = text_pipe.calculate_max_length(cleaned_desc)

    print(f"Taille finale du vocabulaire retenu : {vocab_size}")
    print(f"Longueur maximale d'une description : {max_len}")
    
    text_pipe.save_pipeline()

    print("\n=== ÉTAPE 2 : EXTRACTION DE FEATURES VISUELLES (TRANSFER LEARNING - PYTORCH RESNET50) ===")
    img_pipe = ImageProcessingPipeline(target_size=TARGET_IMAGE_SIZE)
    
    if os.path.exists(FEATURES_PICKLE):
        print(f"Fichier de caractéristiques trouvé ({FEATURES_PICKLE}). Chargement direct...")
        image_features = img_pipe.load_features(FEATURES_PICKLE)
    else:
        print("Chargement du modèle ResNet50 depuis PyTorch (Weights: IMAGENET1K_V2)...")
        
        # 1. Détection et configuration du GPU (RTX 4070 Ti)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. Chargement des poids pré-entraînés officiels et passage en mode évaluation (eval)
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        resnet50 = models.resnet50(weights=weights)
        
        # 3. On remplace la dernière couche de classification (fc) par un Identity() 
        # pour obtenir directement le vecteur de caractéristiques de 2048 dimensions
        resnet50.fc = torch.nn.Identity()
        resnet50 = resnet50.to(device)
        resnet50.eval() # Important : coupe le Dropout et la Batch Normalization
        
        # 4. Définition de la pipeline de transformation d'images standard de PyTorch
        preprocess = transforms.Compose([
            transforms.Resize(TARGET_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("Démarrage de l'extraction sur le dossier d'images...")
        image_features = dict()
        all_files = os.listdir(IMAGES_DIR_PATH)
        valid_ids = set(cleaned_desc.keys())
        
        from tqdm import tqdm
        for name in tqdm(all_files):
            image_id = name.split('.')[0]
            
            if image_id not in valid_ids:
                continue
                
            path = os.path.join(IMAGES_DIR_PATH, name)
            if os.path.isfile(path) and name.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # Chargement via PIL et conversion en RGB (sécurité si image en niveaux de gris)
                    img = Image.open(path).convert('RGB')
                    tensor_img = preprocess(img).unsqueeze(0).to(device) # Ajout de la dimension de batch (1, 3, 224, 224)
                    
                    # Extraction sans calculer les gradients (gain de mémoire et vitesse)
                    with torch.no_grad():
                        feature_vector = resnet50(tensor_img)
                    
                    # Conversion du tenseur PyTorch en tableau NumPy aplati (shape: 2048,)
                    image_features[image_id] = feature_vector.cpu().numpy().flatten()
                except Exception as e:
                    print(f"Erreur lors du traitement de l'image {name} : {e}")
        
        # Sauvegarde sur le disque dur pour ne plus avoir à le refaire au prochain lancement
        img_pipe.save_features(image_features, filename=FEATURES_PICKLE)
    
    print(f"Nombre de vecteurs d'images disponibles pour l'entraînement : {len(image_features)}")

    print("\n=== ÉTAPE 3 : INITIALISATION DE L'ENTRAÎNEMENT DU MODÈLE FUSION ===")
    train_pipe = CaptionTrainingPipeline(
        vocab_size=vocab_size, 
        max_length=max_len, 
        feature_dim=FEATURE_DIMENSION  # Transmet automatiquement les 2048 dimensions à training.py
    )
    
    print("Démarrage de l'ajustement des poids du réseau de neurones...")
    trained_model = train_pipe.fit(
        train_descriptions=cleaned_desc,
        train_features=image_features,
        tokenizer=tokenizer,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    train_pipe.save_model("flickr8k_caption_generator_inception.keras")
    print("\n=== PROCESSUS TERMINÉ AVEC SUCCÈS ===")

if __name__ == "__main__":
    main()