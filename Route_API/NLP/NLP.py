import re
import string

_punct_table = str.maketrans('', '', string.punctuation)

# Le pipeline d'inférence (CaptionModel.generate → NLP.pad_sequence) n'a besoin
# d'aucune ressource nltk : le chargement (téléchargement des corpus + objets
# stopwords/lemmatizer/stemmer, plusieurs dizaines de Mo en RAM) est donc différé
# et ne se déclenche que si preprocess_caption/preprocess_captions sont réellement
# appelés (utilitaires de préparation de données, pas utilisés en production API).
_nltk_ready = False
_lemmatizer = None
_stemmer = None
_stop_words = None


def _ensure_nltk():
    global _nltk_ready, _lemmatizer, _stemmer, _stop_words
    if _nltk_ready:
        return
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.stem.porter import PorterStemmer

    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

    _lemmatizer = WordNetLemmatizer()
    _stemmer = PorterStemmer()
    _stop_words = set(stopwords.words('english'))
    _nltk_ready = True


class NLP():
    def __init__(self):
        pass

    # ------------------------------------------------------------------ #
    #  Étapes individuelles                                                #
    # ------------------------------------------------------------------ #

    def to_lowercase(self, text: str) -> str:
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        return text.translate(_punct_table)

    def remove_numbers(self, text: str) -> str:
        return re.sub(r'\d+', '', text)

    def normalize_whitespace(self, text: str) -> str:
        return ' '.join(text.split())

    def tokenize(self, text: str) -> list[str]:
        _ensure_nltk()
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        _ensure_nltk()
        return [t for t in tokens if t not in _stop_words]

    def lemmatize(self, tokens: list[str]) -> list[str]:
        _ensure_nltk()
        return [_lemmatizer.lemmatize(t) for t in tokens]

    def stem(self, tokens: list[str]) -> list[str]:
        _ensure_nltk()
        return [_stemmer.stem(t) for t in tokens]

    # ------------------------------------------------------------------ #
    #  TODO: Tokens début / fin (à revoir)                                #
    # ------------------------------------------------------------------ #

    def add_caption_tokens(self, tokens: list[str]) -> str:
        return 'startseq ' + ' '.join(tokens) + ' endseq'

    def clean_for_model(self, caption: str) -> str:
        """Nettoyage minimal utilisé à l'entraînement du modèle Keras :
        lowercase + suppression ponctuation + mots alpha uniquement (len > 1)
        + tokens startseq / endseq. Pas de stopwords ni lemmatisation."""
        text   = self.to_lowercase(caption)
        text   = self.remove_punctuation(text)
        tokens = [w for w in text.split() if w.isalpha() and len(w) > 1]
        return self.add_caption_tokens(tokens)

    @staticmethod
    def pad_sequence(seq: list[int], maxlen: int) -> list[int]:
        """Post-padding à zéro jusqu'à maxlen."""
        return (seq + [0] * maxlen)[:maxlen]

    # ------------------------------------------------------------------ #
    #  Pipeline complet pour légendes d'images                            #
    # ------------------------------------------------------------------ #

    def preprocess_caption(self, caption: str, use_stemming: bool = False) -> str:
        text = self.to_lowercase(caption)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.normalize_whitespace(text)

        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)

        if use_stemming:
            tokens = self.stem(tokens)
        else:
            tokens = self.lemmatize(tokens)

        return self.add_caption_tokens(tokens)

    def preprocess_captions(self, captions: list[str], use_stemming: bool = False) -> list[str]:
        return [self.preprocess_caption(c, use_stemming) for c in captions]
