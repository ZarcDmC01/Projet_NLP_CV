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

    def __init__(self, vocab_size, max_length, feature_dim=2048):
        """
        :param vocab_size:   Taille du vocabulaire réellement utilisée par le tokenizer
                             (tokenizer.num_words + 1, incluant l'index 0 réservé au padding)
        :param max_length:   Longueur maximale d'une légende nettoyée
        :param feature_dim:  Dimension de sortie du backbone CNN
                             (2048 pour ResNet50, 4096 pour VGG16, 2048 pour InceptionV3)
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.feature_dim = feature_dim
        self.model = None

    # ─────────────────────────────────────────────
    # Architecture du modèle
    # ─────────────────────────────────────────────

    def build_model(self):
        """
        Construit l'architecture 'Merge' unifiée associant
        la branche Vision et la branche NLP.
        """
        inputs_image = keras.Input(shape=(self.feature_dim,), name="image_inputs")
        fe1 = layers.Dropout(0.5)(inputs_image)
        fe2 = layers.Dense(256, activation="relu")(fe1)

        inputs_text = keras.Input(shape=(self.max_length,), name="text_inputs")
        se1 = layers.Embedding(input_dim=self.vocab_size, output_dim=256, mask_zero=True)(inputs_text)
        se2 = layers.Dropout(0.5)(se1)
        se3 = layers.LSTM(256)(se2)

        decoder1 = layers.add([fe2, se3])
        decoder2 = layers.Dense(256, activation="relu")(decoder1)
        outputs = layers.Dense(self.vocab_size, activation="softmax", name="output_layer")(decoder2)

        self.model = keras.Model(
            inputs=[inputs_image, inputs_text],
            outputs=outputs,
            name="Caption_Generator_Model"
        )
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
        return self.model

    # ─────────────────────────────────────────────
    # Générateur de données
    # ─────────────────────────────────────────────

    def data_generator(self, descriptions, image_features, tokenizer, batch_size):
        """
        Générateur produisant des batches d'entraînement à la volée.
        Découpe chaque phrase mot à mot (teacher forcing).
        """
        X1, X2, y = list(), list(), list()
        n = 0

        while True:
            for key, desc_list in descriptions.items():
                if key not in image_features:
                    continue

                feature = image_features[key]
                for desc in desc_list:
                    seq = tokenizer.texts_to_sequences([desc])[0]

                    for i in range(1, len(seq)):
                        in_seq, out_seq = seq[:i], seq[i]
                        in_seq = keras.utils.pad_sequences([in_seq], maxlen=self.max_length)[0]
                        out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]

                        X1.append(feature)
                        X2.append(in_seq)
                        y.append(out_seq)

                n += 1
                if n >= batch_size:
                    yield [np.array(X1), np.array(X2)], np.array(y)
                    X1, X2, y = list(), list(), list()
                    n = 0

    # ─────────────────────────────────────────────
    # Entraînement
    # ─────────────────────────────────────────────

    def fit(self, train_descriptions, train_features, tokenizer, epochs=50, batch_size=32,
        val_descriptions=None, val_features=None, patience=3,
        checkpoint_path=None):
        """
        Lance la boucle d'entraînement principale du réseau de neurones.
        Si val_descriptions / val_features sont fournis, active l'early stopping
        sur la loss de validation.
        """
        if self.model is None:
            self.build_model()

        print(self.model.summary())

        steps = len(train_descriptions) // batch_size
        generator = self.data_generator(train_descriptions, train_features, tokenizer, batch_size)

        callbacks = []
        fit_kwargs = dict(
            epochs=epochs,
            steps_per_epoch=steps,
            verbose=1,
            callbacks=callbacks
        )

        if val_descriptions is not None and val_features is not None:
            val_steps = max(1, len(val_descriptions) // batch_size)
            val_generator = self.data_generator(val_descriptions, val_features, tokenizer, batch_size)

            fit_kwargs["validation_data"] = val_generator
            fit_kwargs["validation_steps"] = val_steps

            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                )
            )
            print(f"Early stopping activé (patience={patience}) sur {len(val_descriptions)} images de validation.")
        else:
            print("Aucune donnée de validation fournie : early stopping désactivé.")

        if checkpoint_path:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor="val_loss" if (val_descriptions and val_features) else "loss",
                    save_best_only=True,
                    verbose=1
                )
            )

        self.model.fit(generator, **fit_kwargs)
        return self.model

    # ─────────────────────────────────────────────
    # Génération de légendes
    # ─────────────────────────────────────────────

    def generate_caption_greedy(self, image_feature, tokenizer, max_length,
                                start_token="startseq", end_token="endseq"):
        """
        Génère une légende par Greedy Search (argmax à chaque étape).
        Rapide mais sous-optimal.

        :param image_feature: Vecteur numpy de shape (feature_dim,)
        :param tokenizer:     Tokenizer Keras ajusté sur le corpus d'entraînement
        :param max_length:    Longueur maximale de séquence
        :return:              Légende générée sous forme de chaîne de caractères
        """
        index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

        in_text = start_token

        for _ in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = keras.utils.pad_sequences([sequence], maxlen=max_length)

            yhat = self.model.predict(
                [np.array([image_feature]), np.array(sequence)],
                verbose=0
            )
            yhat_idx = np.argmax(yhat)
            word = index_to_word.get(yhat_idx)

            if word is None:
                break

            in_text += ' ' + word

            if word == end_token:
                break

        final_caption = in_text.replace(start_token, '').replace(end_token, '').strip()
        return final_caption

    def generate_caption_beam(self, image_feature, tokenizer, max_length,
                              beam_width=3, start_token="startseq", end_token="endseq"):
        """
        Génère une légende par Beam Search : explore plusieurs séquences en parallèle
        et conserve les 'beam_width' meilleures à chaque étape.
        Produit des résultats nettement meilleurs qu'un simple argmax.

        :param image_feature: Vecteur numpy de shape (feature_dim,)
        :param tokenizer:     Tokenizer Keras ajusté sur le corpus d'entraînement
        :param max_length:    Longueur maximale de séquence
        :param beam_width:    Nombre de séquences candidates à conserver (3 ou 5 recommandé)
        :return:              Meilleure légende générée sous forme de chaîne de caractères
        """
        index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

        candidates = [[0.0, [start_token]]]

        for _ in range(max_length):
            next_candidates = []

            for score, seq in candidates:
                last_word = seq[-1]

                if last_word == end_token:
                    next_candidates.append([score, seq])
                    continue

                in_text = ' '.join(seq)
                sequence = tokenizer.texts_to_sequences([in_text])[0]
                sequence = keras.utils.pad_sequences([sequence], maxlen=max_length)

                yhat = self.model.predict(
                    [np.array([image_feature]), np.array(sequence)],
                    verbose=0
                )[0]

                top_indices = np.argsort(yhat)[-beam_width:]

                for idx in top_indices:
                    word = index_to_word.get(idx)
                    if word is None:
                        continue
                    new_score = score + np.log(yhat[idx] + 1e-10)
                    new_seq = seq + [word]
                    next_candidates.append([new_score, new_seq])

            candidates = sorted(next_candidates, key=lambda x: x[0], reverse=True)[:beam_width]

            if all(seq[-1] == end_token for _, seq in candidates):
                break

        best_seq = candidates[0][1]

        final_caption = ' '.join(
            word for word in best_seq
            if word not in (start_token, end_token)
        )
        return final_caption

    # ─────────────────────────────────────────────
    # Persistance du modèle
    # ─────────────────────────────────────────────

    def save_model(self, filepath="flickr8k_caption_generator_resnet4.keras"):
        """Sauvegarde le modèle complet entraîné."""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Modèle sauvegardé → '{filepath}'")