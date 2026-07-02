import os
import pickle
import numpy as np
import tensorflow as tf
from preprocessing import load_doc, load_descriptions, clean_descriptions, create_and_save_tokenizer
from models import define_captioning_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def data_generator(descriptions, image_features, tokenizer, max_length, vocab_size, batch_size):
    """
    Mon générateur de données personnalisé.
    Je l'ai conçu pour générer en continu des paires [Image, Séquence de mots] -> Mot suivant
    afin de ne pas surcharger la mémoire RAM de ma machine.
    """
    X1, X2, y = list(), list(), list()
    n = 0
    while True:
        for img_id, desc_list in descriptions.items():
            if img_id not in image_features:
                continue
            feature = image_features[img_id][0]
            for desc in desc_list:
                # Je transforme la description textuelle en une séquence d'entiers
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    # J'applique un padding pour que toutes mes séquences de texte aient la même longueur
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # Je convertis le mot cible en vecteur One-Hot Encoding selon la taille de mon vocabulaire
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
            n += 1
            if n == batch_size:
                # Je livre le lot de données structurellement prêt pour Keras
                yield ((np.array(X1), np.array(X2)), np.array(y))
                X1, X2, y = list(), list(), list()
                n = 0

def load_split_ids(filename):
    """Je charge ici les identifiants uniques des images dédiées à l'entraînement."""
    doc = load_doc(filename)
    if not doc: 
        return set()
    return set([line.split('.')[0] for line in doc.split('\n') if len(line) > 0])

def filter_dataset(all_descriptions, all_features, split_ids):
    """Je filtre mes dictionnaires globaux pour ne conserver que les données du split choisi."""
    descriptions = {k: v for k, v in all_descriptions.items() if k in split_ids}
    features = {k: v for k, v in all_features.items() if k in split_ids}
    return descriptions, features


# --- MON BLOC D'ENTRAÎNEMENT ---
if __name__ == "__main__":
    print(" J'initialise mon pipeline et je lance le chargement global des données...")
    
    # Je définis mes chemins relatifs dynamiques vers mes fichiers de données
    TEXT_FILE = os.path.join("Data", "Flickr8k_text", "Flickr8k.token.txt")
    TRAIN_IDS_FILE = os.path.join("Data", "Flickr8k_text", "Flickr_8k.trainImages.txt")
    FEATURES_PICKLE = "features.pkl"
    TOKENIZER_PATH = "tokenzier.pkl"
    
    # J'ingère et je nettoie les descriptions textuelles du dataset
    doc = load_doc(TEXT_FILE)
    if not doc:
        print(f" Erreur critique : Je n'ai pas pu charger le fichier texte : {TEXT_FILE}")
        exit()
        
    raw_descriptions = load_descriptions(doc)
    cleaned_descriptions = clean_descriptions(raw_descriptions)
    train_ids = load_split_ids(TRAIN_IDS_FILE)
    print(f"J'ai sélectionné {len(train_ids)} images pour la phase d'entraînement.")
    
    # Je m'assure que j'ai bien extrait les caractéristiques des images au préalable
    if not os.path.exists(FEATURES_PICKLE):
        print(f" Erreur : Le fichier '{FEATURES_PICKLE}' est introuvable. Je dois d'abord lancer preprocessing.py.")
        exit()
        
    with open(FEATURES_PICKLE, 'rb') as f:
        all_features = pickle.load(f)
        
    # Je sépare mes données pour ne garder que le sous-ensemble d'entraînement
    train_descriptions, train_features = filter_dataset(cleaned_descriptions, all_features, train_ids)
    
    # Je gère la persistance de mon Tokenizer (chargement ou création)
    if os.path.exists(TOKENIZER_PATH):
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        print(" J'ai chargé mon Tokenizer existant ('tokenzier.pkl') avec succès.")
    else:
        tokenizer = create_and_save_tokenizer(train_descriptions, TOKENIZER_PATH)
    
    # Je configure mes hyperparamètres de dimensionnement et de calcul
    VOCAB_SIZE = len(tokenizer.word_index) + 1
    MAX_LENGTH = 34
    BATCH_SIZE = 64
    EPOCHS = 20         # J'ai configuré 50 époques pour donner au modèle le temps de converger
    
    print(f" Taille de mon vocabulaire : {VOCAB_SIZE} mots")
    print(f" Longueur maximale fixée pour mes séquences : {MAX_LENGTH}")
    
    # J'instancie mon architecture de modèle hybride CNN-LSTM
    model = define_captioning_model(VOCAB_SIZE, MAX_LENGTH)
    
    # J'encapsule proprement mon générateur dans un Dataset natif TensorFlow
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(train_descriptions, train_features, tokenizer, MAX_LENGTH, VOCAB_SIZE, BATCH_SIZE),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 4096), dtype=tf.float32, name="image_features"),
                tf.TensorSpec(shape=(None, MAX_LENGTH), dtype=tf.int32, name="text_sequences")
            ),
            tf.TensorSpec(shape=(None, VOCAB_SIZE), dtype=tf.float32, name="output_words")
        )
    )
    
    steps = len(train_descriptions) // BATCH_SIZE
    
    # ─── MA STRATÉGIE MLOPS : LES CALLBACKS ───
    # J'implémente un Early Stopping pour couper l'entraînement si mon modèle n'apprend plus rien pendant 3 époques
    early_stopping = EarlyStopping(
        monitor='loss', 
        patience=3, 
        restore_best_weights=True, 
        verbose=1
    )
    
    # Je configure une sauvegarde automatique pour figer uniquement mes meilleurs poids synaptiques
    checkpoint = ModelCheckpoint(
        "model_captioning.keras", 
        monitor='loss', 
        save_best_only=True, 
        verbose=1
    )
    
    print("\nJe démarre l'entraînement de mon modèle (Sécurisé par mon Early Stopping)...")
    
    # Je lance l'apprentissage en y rattachant mes deux dispositifs de sécurité
    model.fit(
        dataset, 
        epochs=EPOCHS, 
        steps_per_epoch=steps, 
        verbose=1,
        callbacks=[early_stopping, checkpoint]
    )
    
    print("\n Entraînement terminé ! J'ai sauvegardé ma meilleure configuration sous 'model_captioning.keras'.")