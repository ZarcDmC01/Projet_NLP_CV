from fastapi import FastAPI

from Image.image_ctrler import router as image_router

app = FastAPI(title="API NLP¨")

app.include_router(Image)
app.include_router(NLP)

