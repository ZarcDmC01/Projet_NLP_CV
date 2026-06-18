import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch

from Route_API.Model.model import (
    LSTMCaptioner,
    MODEL_PATH,
    EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, MAX_LEN,
    PAD, UNK, SOS, EOS,
)
from Route_API.Model.model_service import ModelService, build_vocab, _load_model
from Route_API.Image.image_service import ImageService

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print(f"Device : {DEVICE}")
print("=" * 60)

# ------------------------------------------------------------------ #
#  Test 1 — build_vocab                                              #
# ------------------------------------------------------------------ #

word2idx = idx2word = None

try:
    word2idx, idx2word = build_vocab()
    for token in (PAD, UNK, SOS, EOS):
        assert token in word2idx, f"Token manquant : {token}"
    assert len(word2idx) == len(idx2word)
    # idx2word est bien l'inverse de word2idx
    assert idx2word[word2idx[SOS]] == SOS
    print(f"  vocab_size={len(word2idx)}")
    print("✅ build_vocab : OK")
except Exception as e:
    print(f"❌ build_vocab : {e}")

# ------------------------------------------------------------------ #
#  Test 2 — LSTMCaptioner.__init__                                   #
# ------------------------------------------------------------------ #

vocab_size = len(word2idx) if word2idx else 2000
model = None

try:
    model = LSTMCaptioner(
        feature_dim=512,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        vocab_size=vocab_size,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)
    print(f"  paramètres entraînables={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("✅ LSTMCaptioner.__init__ : OK")
except Exception as e:
    print(f"❌ LSTMCaptioner.__init__ : {e}")

# ------------------------------------------------------------------ #
#  Test 3 — forward() — passe entraînement (teacher forcing)         #
# ------------------------------------------------------------------ #

dummy_features = torch.randn(4, 512, device=DEVICE)  # batch de 4
dummy_captions = torch.zeros(4, MAX_LEN, dtype=torch.long, device=DEVICE)

try:
    logits = model.forward(dummy_features, dummy_captions)
    # shape attendue : (batch, MAX_LEN-1, vocab_size)
    assert logits.shape == (4, MAX_LEN - 1, vocab_size), f"Shape inattendue : {logits.shape}"
    print(f"  logits.shape={logits.shape}")
    print("✅ LSTMCaptioner.forward : OK")
except Exception as e:
    print(f"❌ LSTMCaptioner.forward : {e}")

# ------------------------------------------------------------------ #
#  Test 4 — init_hidden()                                            #
# ------------------------------------------------------------------ #

try:
    feat_1 = torch.randn(1, 512, device=DEVICE)
    h, c   = model.init_hidden(feat_1)
    assert h.shape == (1, 1, HIDDEN_DIM), f"Shape h inattendue : {h.shape}"
    assert c.shape == (1, 1, HIDDEN_DIM), f"Shape c inattendue : {c.shape}"
    assert torch.all(c == 0), "c0 devrait être zéro"
    print(f"  h.shape={h.shape}  c.shape={c.shape}")
    print("✅ LSTMCaptioner.init_hidden : OK")
except Exception as e:
    print(f"❌ LSTMCaptioner.init_hidden : {e}")

# ------------------------------------------------------------------ #
#  Test 5 — step()                                                   #
# ------------------------------------------------------------------ #

try:
    sos_idx = word2idx[SOS] if word2idx else 2
    token   = torch.tensor([[sos_idx]], device=DEVICE)
    logits_step, h_new, c_new = model.step(token, h, c)
    assert logits_step.shape == (1, vocab_size), f"Shape logits inattendue : {logits_step.shape}"
    assert h_new.shape == h.shape
    assert c_new.shape == c.shape
    print(f"  logits_step.shape={logits_step.shape}")
    print("✅ LSTMCaptioner.step : OK")
except Exception as e:
    print(f"❌ LSTMCaptioner.step : {e}")

# ------------------------------------------------------------------ #
#  Test 6 — _load_model()                                            #
# ------------------------------------------------------------------ #

print("=" * 60)

try:
    loaded_model, ready = _load_model(word2idx, DEVICE)
    assert isinstance(loaded_model, LSTMCaptioner)
    assert isinstance(ready, bool)
    status_str = "poids chargés" if ready else f"non trouvé ({MODEL_PATH.name})"
    print(f"  ready={ready}  ({status_str})")
    print("✅ _load_model : OK")
except Exception as e:
    print(f"❌ _load_model : {e}")

# ------------------------------------------------------------------ #
#  Test 7 — ModelService.load()                                      #
# ------------------------------------------------------------------ #

try:
    ModelService.load(DEVICE)
    assert ModelService._word2idx is not None
    assert ModelService._idx2word is not None
    assert ModelService._model    is not None
    print(f"  ready={ModelService._ready}  vocab_size={len(ModelService._word2idx)}")
    print("✅ ModelService.load : OK")
except Exception as e:
    print(f"❌ ModelService.load : {e}")

# ------------------------------------------------------------------ #
#  Test 8 — get_status()                                             #
# ------------------------------------------------------------------ #

try:
    status = ModelService.get_status()
    for key in ("model_ready", "model_path", "vocab_size", "device"):
        assert key in status, f"Clé manquante : {key}"
    print(f"  {status}")
    print("✅ ModelService.get_status : OK")
except Exception as e:
    print(f"❌ ModelService.get_status : {e}")

# ------------------------------------------------------------------ #
#  Test 9 — _generate_sequence() (nécessite poids OU model non prêt) #
# ------------------------------------------------------------------ #

print("=" * 60)

try:
    feat_tensor = torch.randn(1, 512, device=DEVICE)
    if ModelService._ready:
        caption = ModelService._generate_sequence(feat_tensor)
        assert isinstance(caption, str)
        print(f"  caption='{caption}'")
    else:
        # Model non entraîné : génère quand même une séquence (tokens aléatoires)
        caption = ModelService._generate_sequence(feat_tensor)
        assert isinstance(caption, str)
        print(f"  model non entraîné, caption='{caption}'")
    print("✅ ModelService._generate_sequence : OK")
except Exception as e:
    print(f"❌ ModelService._generate_sequence : {e}")

# ------------------------------------------------------------------ #
#  Test 10 — generate_caption(image_id) pipeline complet             #
#  On persiste un vecteur de test dans ImageService pour éviter       #
#  d'avoir besoin de ResNet34 dans ce test.                           #
# ------------------------------------------------------------------ #

TEST_IMAGE_ID = "__test_model__"

try:
    # Vecteur features fictif (512,)
    fake_features = np.random.randn(512).astype(np.float32)
    ImageService.store_features(TEST_IMAGE_ID, fake_features)

    result = ModelService.generate_caption(TEST_IMAGE_ID)
    assert result is not None, "generate_caption a retourné None pour un image_id connu"
    assert "caption" in result
    assert "status" in result
    print(f"  status={result['status']}  caption='{result['caption']}'")
    print("✅ ModelService.generate_caption : OK")
except Exception as e:
    print(f"❌ ModelService.generate_caption : {e}")

# ------------------------------------------------------------------ #
#  Test 11 — generate_caption avec image_id inconnu → None           #
# ------------------------------------------------------------------ #

try:
    result_none = ModelService.generate_caption("__image_id_inexistant__")
    assert result_none is None, f"Attendu None, reçu {result_none}"
    print("✅ generate_caption image_id inconnu → None : OK")
except Exception as e:
    print(f"❌ generate_caption image_id inconnu : {e}")

print("=" * 60)
print("Tests terminés.")
