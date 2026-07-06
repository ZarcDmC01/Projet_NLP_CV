import os
from dotenv import load_dotenv
load_dotenv()  # charge .env en local (Render utilise directement ses variables d'env dashboard)

os.environ.setdefault("KERAS_BACKEND", "torch")
# Limite les threads BLAS/torch pour réduire le pic de RAM (contrainte 512Mi sur Render free tier)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from Route_API.Image.image_service import ImageService
from Route_API.Model.model_service import ModelService

from Route_API.Image.image_ctrler import router as image_router
from Route_API.Model.model_ctrler import router as model_router
from Route_API.NLP.NLP_ctrler import router as NLP_router
from Route_API.Security.Security_ctrler import router as Security_router
from Route_API.Monitoring.Monitoring_ctrler import router as Monitor_ctrler


def _load_backbone():
    import torch
    import torchvision.models as models
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet50.fc = torch.nn.Identity()
    resnet50.eval()
    return resnet50


def _load_models():
    print("[API] Chargement des modèles en arrière-plan...")
    ImageService.set_model(_load_backbone())
    ModelService.load()
    print("[API] Modèles prêts.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Démarrage — port ouvert, modèles en cours de chargement...")
    asyncio.get_event_loop().run_in_executor(None, _load_models)
    yield
    print("[API] Arrêt.")


app = FastAPI(title="Commentary_API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001", "https://projetnlpcv.vercel.app"],
    allow_origin_regex=r"https://projet-nlp-cv.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


app.include_router(image_router)
app.include_router(model_router)
app.include_router(NLP_router)
app.include_router(Security_router)
app.include_router(Monitor_ctrler)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
