from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model

def define_captioning_model(vocab_size, max_length):
    """
    Étape 4 : Développer un modèle de Deep Learning (Transfer Learning & Décodeur LSTM).
    Construit l'architecture hybride permettant d'associer les caractéristiques
    visuelles de l'image avec la séquence de texte.
    """
    # 1. Branche Image (Encodeur - Feature Extractor issu de VGG16)
    # L'entrée prend le vecteur de taille 4096 extrait par la couche dense de VGG16
    inputs1 = Input(shape=(4096,), name="Image_Input")
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # 2. Branche Texte (Décodeur - Séquence textuelle avec LSTM)
    # L'entrée prend les indices des mots de longueur maximale max_length
    inputs2 = Input(shape=(max_length,), name="Text_Input")
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # 3. Fusion des deux modalités (Fusion Réseau)
    # Combine les informations visuelles (fe2) et textuelles (se3)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    
    # Couche de sortie : prédiction probabiliste du mot suivant parmi tout le vocabulaire
    outputs = Dense(vocab_size, activation='softmax', name="Output_Layer")(decoder2)
    
    # Initialisation globale du modèle de bout en bout
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    
    # Compilation du modèle avec la fonction de perte pour classification multiclasse
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

if __name__ == "__main__":
    # Test rapide de l'architecture pour vérification
    # Exemple : Vocabulaire fictif de 5000 mots et séquences de 34 mots maximum
    test_model = define_captioning_model(vocab_size=5000, max_length=34)
    test_model.summary()
    print("\n Architecture du modèle validée avec succès dans models.py !")