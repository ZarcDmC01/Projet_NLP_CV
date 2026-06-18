import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

from Route_API.Image.image import ImageProcessingPipeline
from Route_API.Image.image_service import ImageService

# ------------------------------------------------------------------ #
#  Image de test (première image Flickr8k disponible)                 #
# ------------------------------------------------------------------ #

IMAGE_DIR  = Path('d:/Projet_NLP_CV/Data/Flickr8k_Dataset/Flicker8k_Dataset')
TEST_IMAGE = next(IMAGE_DIR.glob('*.jpg'))

pipeline = ImageProcessingPipeline(target_size=(224, 224))

print("=" * 55)
print(f"Image de test : {TEST_IMAGE.name}")
print("=" * 55)

# ------------------------------------------------------------------ #
#  Test étapes atomiques — ImageProcessingPipeline                    #
# ------------------------------------------------------------------ #

img = None

try:
    img = pipeline.load_from_path(str(TEST_IMAGE))
    print(f"  mode={img.mode}  size={img.size}")
    print("✅ load_from_path : OK")
except Exception as e:
    print(f"❌ load_from_path : {e}")

try:
    img_bytes = TEST_IMAGE.read_bytes()
    img_from_bytes = pipeline.load_from_bytes(img_bytes)
    assert img_from_bytes.size == img.size
    print(f"  mode={img_from_bytes.mode}  size={img_from_bytes.size}")
    print("✅ load_from_bytes : OK")
except Exception as e:
    print(f"❌ load_from_bytes : {e}")

try:
    resized = pipeline.resize(img)
    assert resized.size == (224, 224), f"Taille inattendue : {resized.size}"
    print(f"  size={resized.size}")
    print("✅ resize : OK")
except Exception as e:
    print(f"❌ resize : {e}")

try:
    arr = pipeline.to_array(resized)
    assert arr.dtype == np.float32
    assert arr.shape == (224, 224, 3)
    print(f"  shape={arr.shape}  dtype={arr.dtype}  min={arr.min():.0f}  max={arr.max():.0f}")
    print("✅ to_array : OK")
except Exception as e:
    print(f"❌ to_array : {e}")

try:
    normalized = pipeline.normalize(arr)
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0, \
        f"Valeurs hors [0,1] : min={normalized.min():.4f} max={normalized.max():.4f}"
    print(f"  min={normalized.min():.4f}  max={normalized.max():.4f}")
    print("✅ normalize : OK")
except Exception as e:
    print(f"❌ normalize : {e}")

try:
    batched = pipeline.add_batch_dim(normalized)
    assert batched.shape == (1, 224, 224, 3)
    print(f"  shape={batched.shape}")
    print("✅ add_batch_dim : OK")
except Exception as e:
    print(f"❌ add_batch_dim : {e}")

# ------------------------------------------------------------------ #
#  Test extract_features + flatten (nécessite un backbone)            #
# ------------------------------------------------------------------ #

print("=" * 55)
print("Chargement du backbone ResNet34...")
print("=" * 55)

try:
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet    = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    extractor = nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()
    ImageService.load_torch_backbone(extractor, device)
    print(f"✅ load_torch_backbone : OK  (device={device})")
except Exception as e:
    print(f"❌ load_torch_backbone : {e}")

try:
    raw_features = pipeline.extract_features(batched, ImageService._model_backbone)
    assert raw_features.shape == (1, 512), f"Shape inattendue : {raw_features.shape}"
    print(f"  shape={raw_features.shape}  dtype={raw_features.dtype}")
    print("✅ extract_features : OK")
except Exception as e:
    print(f"❌ extract_features : {e}")

try:
    flat = pipeline.flatten_features(raw_features)
    assert flat.shape == (512,)
    print(f"  shape={flat.shape}  norm={np.linalg.norm(flat):.4f}")
    print("✅ flatten_features : OK")
except Exception as e:
    print(f"❌ flatten_features : {e}")

# ------------------------------------------------------------------ #
#  Test pipeline complet — ImageService                               #
# ------------------------------------------------------------------ #

print("=" * 55)

try:
    img_bytes = TEST_IMAGE.read_bytes()
    features  = ImageService.extract_features(img_bytes)
    assert features is not None, "extract_features a retourné None"
    assert features.shape == (512,)
    assert features.dtype == np.float32
    print(f"  shape={features.shape}  dtype={features.dtype}  norm={np.linalg.norm(features):.4f}")
    print("✅ ImageService.extract_features : OK")
except Exception as e:
    print(f"❌ ImageService.extract_features : {e}")

try:
    result = ImageService.process_image(img_bytes)
    assert result['status'] == 'features_extracted'
    assert result['features_shape'] == [512]
    assert len(result['features']) == 512
    print(f"  status={result['status']}  features_shape={result['features_shape']}")
    print("✅ ImageService.process_image : OK")
except Exception as e:
    print(f"❌ ImageService.process_image : {e}")

# ------------------------------------------------------------------ #
#  Test sans backbone (comportement attendu)                          #
# ------------------------------------------------------------------ #

print("=" * 55)

try:
    ImageService._model_backbone = None
    result_no_model = ImageService.process_image(img_bytes)
    assert result_no_model['status'] == 'preprocessing_ok'
    print(f"  status={result_no_model['status']}")
    print(f"  message={result_no_model['message']}")
    print("✅ process_image sans backbone : OK")
except Exception as e:
    print(f"❌ process_image sans backbone : {e}")

try:
    features_none = ImageService.extract_features(img_bytes)
    assert features_none is None
    print("✅ extract_features sans backbone → None : OK")
except Exception as e:
    print(f"❌ extract_features sans backbone : {e}")

print("=" * 55)
print("Tests terminés.")
