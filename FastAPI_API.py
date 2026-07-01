import os
os.environ.setdefault("KERAS_BACKEND", "torch")

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


def _load_inception():
    import keras
    return keras.applications.InceptionV3(
        weights="imagenet", include_top=False, pooling="avg"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Démarrage — chargement des modèles...")
    ImageService.set_model(_load_inception())
    ModelService.load()
    print("[API] Prêt.")
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
