import os
import numpy as np
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
from keras.utils import to_categorical

class CaptionTrainingPipeline:
    """
    Pipeline responsable de l'orchestration de l'entraînement du modèle,
    de la création de l'architecture et de la génération des batches de données.
    """
    def __init__(self, vocab_size, max_length, feature_dim=4096):
        """
        :param vocab_size: Taille totale du dictionnaire de mots (tokenizer.num_words ou len(tokenizer.word_index)+1)
        :param max_length: Longueur maximale d'une légende nettoyée
        :param feature_dim: Dimension de sortie du modèle de Transfer Learning (ex: 4096 pour VGG16, 2048 pour ResNet50)
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.feature_dim = feature_dim
        self.model = None

    def build_model(self):
        """
        Construit l'architecture 'Merge' unifiée associant la branche Vision et la branche NLP.
        """
        # --- BRANCHE VISION (Extraction de Features) ---
        inputs_image = keras.Input(shape=(self.feature_dim,), name="image_inputs")
        fe1 = layers.Dropout(0.5)(inputs_image)
        fe2 = layers.Dense(256, activation="relu")(fe1)

        # --- BRANCHE NLP (Traitement de Séquence) ---
        inputs_text = keras.Input(shape=(self.max_length,), name="text_inputs")
        se1 = layers.Embedding(input_dim=self.vocab_size, output_dim=256, mask_zero=True)(inputs_text)
        se2 = layers.Dropout(0.5)(se1)
        se3 = layers.LSTM(256)(se2)

        # --- FUSION ET SORTIE ---
        decoder1 = layers.add([fe2, se3])
        decoder2 = layers.Dense(256, activation="relu")(decoder1)
        outputs = layers.Dense(self.vocab_size, activation="softmax", name="output_layer")(decoder2)

        # Assemblage final
        self.model = keras.Model(inputs=[inputs_image, inputs_text], outputs=outputs, name="Caption_Generator_Model")
        
        # Compilation du modèle (Compatible avec le backend Torch)
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
        return self.model

    def data_generator(self, descriptions, image_features, tokenizer, batch_size):
        """
        Générateur de données thread-safe produisant des batches d'entraînement à la volée.
        Découpe chaque phrase mot à mot pour l'apprentissage supervisé de la séquence.
        """
        X1, X2, y = list(), list(), list()
        n = 0
        
        while True:
            for key, desc_list in descriptions.items():
                # Vérification que l'image possède bien des features associées
                if key not in image_features:
                    continue
                
                feature = image_features[key]
                for desc in desc_list:
                    # Encodage de la chaîne de caractères en séquence d'entiers
                    seq = tokenizer.texts_to_sequences([desc])[0]
                    
                    # Découpage progressif en paires d'entrées / sorties
                    for i in range(1, len(seq)):
                        in_seq, out_seq = seq[:i], seq[i]
                        
                        # Padding de la séquence d'entrée pour uniformiser sa taille
                        in_seq = keras.utils.pad_sequences([in_seq], maxlen=self.max_length)[0]
                        
                        # Encodage de la sortie en One-Hot Vector
                        out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                        
                        # Stockage
                        X1.append(feature)
                        X2.append(in_seq)
                        y.append(out_seq)
                
                n += 1
                if n >= batch_size:
                    yield [np.array(X1), np.array(X2)], np.array(y)
                    X1, X2, y = list(), list(), list()
                    n = 0

    def fit(self, train_descriptions, train_features, tokenizer, epochs=20, batch_size=32):
        """
        Lance la boucle d'entraînement principale du réseau de neurones.
        """
        if self.model is None:
            self.build_model()
            
        print(self.model.summary())
        
        # Détermination du nombre de pas par époque (ici basé sur le nombre de clés d'images)
        steps = len(train_descriptions) // batch_size
        
        # Initialisation du générateur
        generator = self.data_generator(train_descriptions, train_features, tokenizer, batch_size)
        
        # Entraînement
        self.model.fit(
            generator,
            epochs=epochs,
            steps_per_epoch=steps,
            verbose=1
        )
        return self.model

    def save_model(self, filepath="caption_model.keras"):
        """Sauvegarde le modèle complet entraîné."""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Modèle sauvegardé avec succès à l'emplacement : {filepath}")