from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import torchvision.models as models
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from Route_API.Image.image_service import ImageService
from Route_API.Model.model_service import ModelService

from Route_API.Image.image_ctrler import router as image_router
from Route_API.Model.model_ctrler import router as model_router
from Route_API.NLP.NLP_ctrler import router as NLP_router
from Route_API.Security.Security_ctrler import router as Security_router
from Route_API.Monitoring.Monitoring_ctrler import router as Monitor_ctrler


def _load_resnet34(device) -> nn.Module:
    """ResNet34 pré-entraîné sans la couche FC finale → extracteur (512,)."""
    resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    extractor = nn.Sequential(*list(resnet.children())[:-1])
    return extractor.to(device).eval()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Démarrage — chargement des modèles...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Image : backbone ResNet34 → ImageService
    feature_extractor = _load_resnet34(device)
    ImageService.load_torch_backbone(feature_extractor, device)

    # 2. Modèle : LSTM decoder → ModelService
    ModelService.load(device)

    print("[API] Prêt.")
    yield
    print("[API] Arrêt.")


app = FastAPI(title="Commentary_API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001", "https://projetnlpcv.vercel.app"],
    allow_origin_regex=r"https://projetnlpcv.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(image_router)
app.include_router(model_router)
app.include_router(NLP_router)
app.include_router(Security_router)
app.include_router(Monitor_ctrler)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
