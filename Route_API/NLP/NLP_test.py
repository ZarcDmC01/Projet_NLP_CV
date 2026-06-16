from NLP import NLP

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

_lemmatizer = WordNetLemmatizer()
_stemmer = PorterStemmer()
_stop_words = set(stopwords.words('english'))
_punct_table = str.maketrans('', '', string.punctuation)

nlp = NLP()

TEXT = "A young girl is running quickly through 2 green fields near the old trees!"

print("=" * 50)
print(f"Texte original : {TEXT}")
print("=" * 50)

# ------------------------------------------------------------------ #
#  Test étapes individuelles                                           #
# ------------------------------------------------------------------ #

try:
    text = nlp.to_lowercase(TEXT)
    print(f"  {text}")
    print("✅ to_lowercase : OK")
except Exception as e:
    print(f"❌ to_lowercase : {e}")

try:
    text = nlp.remove_punctuation(text)
    print(f"  {text}")
    print("✅ remove_punctuation : OK")
except Exception as e:
    print(f"❌ remove_punctuation : {e}")

try:
    text = nlp.remove_numbers(text)
    print(f"  {text}")
    print("✅ remove_numbers : OK")
except Exception as e:
    print(f"❌ remove_numbers : {e}")

try:
    text = nlp.normalize_whitespace(text)
    print(f"  {text}")
    print("✅ normalize_whitespace : OK")
except Exception as e:
    print(f"❌ normalize_whitespace : {e}")

try:
    tokens = nlp.tokenize(text)
    print(f"  {tokens}")
    print("✅ tokenize : OK")
except Exception as e:
    print(f"❌ tokenize : {e}")

try:
    tokens = nlp.remove_stopwords(tokens)
    print(f"  {tokens}")
    print("✅ remove_stopwords : OK")
except Exception as e:
    print(f"❌ remove_stopwords : {e}")

try:
    lemmas = nlp.lemmatize(tokens)
    print(f"  {lemmas}")
    print("✅ lemmatize : OK")
except Exception as e:
    print(f"❌ lemmatize : {e}")

try:
    stems = nlp.stem(tokens)
    print(f"  {stems}")
    print("✅ stem : OK")
except Exception as e:
    print(f"❌ stem : {e}")

try:
    caption = nlp.add_caption_tokens(lemmas)
    print(f"  {caption}")
    print("✅ add_caption_tokens : OK")
except Exception as e:
    print(f"❌ add_caption_tokens : {e}")

# ------------------------------------------------------------------ #
#  Test pipeline complet                                               #
# ------------------------------------------------------------------ #

print("=" * 50)

try:
    result = nlp.preprocess_caption(TEXT)
    print(f"  {result}")
    print("✅ preprocess_caption : OK")
except Exception as e:
    print(f"❌ preprocess_caption : {e}")

try:
    captions = [TEXT, "A dog is running in the park near 3 trees."]
    results = nlp.preprocess_captions(captions)
    for r in results:
        print(f"  {r}")
    print("✅ preprocess_captions : OK")
except Exception as e:
    print(f"❌ preprocess_captions : {e}")
