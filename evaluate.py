import os
import pickle
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing import load_doc, load_descriptions

def idx_to_word(integer, tokenizer):
    """Traduits un index entier en mot grâce au dictionnaire du Tokenizer."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, tokenizer, image_feature, max_length):
    """Génère pas à pas une légende pour une image donnée (Inférence par décodage glouton)."""
    # La phrase commence obligatoirement par la balise d'initialisation
    in_text = 'startseq'
    
    for i in range(max_length):
        # Encoder la phrase textuelle actuelle en séquence numérique
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Ajouter du padding pour atteindre la dimension d'entrée fixe (34)
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Prédire le mot suivant (Distribution de probabilités Softmax)
        prediction = model.predict([image_feature, sequence], verbose=0)
        # Extraire l'index du mot ayant la plus haute probabilité
        idx = np.argmax(prediction)
        
        # Traduire l'index en mot réel
        word = idx_to_word(idx, tokenizer)
        if word is None:
            break
            
        # Ajouter le mot prédit à la phrase en cours
        in_text += ' ' + word
        
        # Si le modèle génère la balise de fin, on arrête la génération
        if word == 'endseq':
            break
            
    return in_text

def load_split_ids(filename):
    """Charge les identifiants d'images du jeu de test."""
    doc = load_doc(filename)
    if not doc: 
        return set()
    return set([line.split('.')[0] for line in doc.split('\n') if len(line) > 0])

def evaluate_model(model, test_descriptions, test_features, tokenizer, max_length):
    """Calcule les scores BLEU sur l'ensemble du corpus de test."""
    actual_captions, predicted_captions = list(), list()
    count = 0
    
    print(" Génération des légendes pour le jeu de test (opération sur CPU)...")
    for img_id, desc_list in test_descriptions.items():
        if img_id not in test_features:
            continue
            
        # Prédire la légende
        y_pred = predict_caption(model, tokenizer, test_features[img_id], max_length)
        
        # Séparer les phrases en listes de mots pour NLTK
        references = [d.split() for d in desc_list]
        prediction = y_pred.split()
        
        actual_captions.append(references)
        predicted_captions.append(prediction)
        
        count += 1
        if count % 100 == 0:
            print(f" Évaluation : {count}/{len(test_descriptions)} images traitées...")

    # Calcul des métriques standards BLEU (NLTK)
    print("\nCalcul des scores BLEU...")
    print(f" BLEU-1: {corpus_bleu(actual_captions, predicted_captions, weights=(1.0, 0, 0, 0)):.4f}")
    print(f" BLEU-2: {corpus_bleu(actual_captions, predicted_captions, weights=(0.5, 0.5, 0, 0)):.4f}")
    print(f" BLEU-3: {corpus_bleu(actual_captions, predicted_captions, weights=(0.33, 0.33, 0.33, 0)):.4f}")
    print(f" BLEU-4: {corpus_bleu(actual_captions, predicted_captions, weights=(0.25, 0.25, 0.25, 0.25)):.4f}")


if __name__ == "__main__":
    print(" Initialisation du pipeline d'évaluation...")
    
    # 1. Chemins d'accès configurés pour votre arborescence
    TEXT_FILE = os.path.join("Data", "Flickr8k_text", "Flickr8k.token.txt")
    TEST_IDS_FILE = os.path.join("Data", "Flickr8k_text", "Flickr_8k.testImages.txt")
    FEATURES_PICKLE = "features.pkl"
    TOKENIZER_PATH = "tokenzier.pkl"
    MODEL_PATH = "model_captioning.keras"
    
    # Validation de la présence du modèle entraîné
    if not os.path.exists(MODEL_PATH):
        print(f" Erreur : Le modèle '{MODEL_PATH}' est introuvable. Vous devez d'abord lancer train.py.")
        exit()
        
    # 2. Chargement des artefacts de données
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(FEATURES_PICKLE, 'rb') as f:
        all_features = pickle.load(f)
        
    doc = load_doc(TEXT_FILE)
    raw_descriptions = load_descriptions(doc)
    
    # Chargement du split de test
    test_ids = load_split_ids(TEST_IDS_FILE)
    print(f"Nombre d'images de test : {len(test_ids)}")
    
    # Filtrage des descriptions de test (sans ajouter startseq/endseq aux références réelles)
    test_descriptions = {k: v for k, v in raw_descriptions.items() if k in test_ids}
    test_features = {k: v for k, v in all_features.items() if k in test_ids}
    
    # 3. Chargement du modèle de deep learning
    print(" Chargement du modèle entraîné...")
    model = load_model(MODEL_PATH)
    
    # 4. Lancement de l'évaluation globale
    MAX_LENGTH = 34
    evaluate_model(model, test_descriptions, test_features, tokenizer, MAX_LENGTH)