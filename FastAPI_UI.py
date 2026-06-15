from fastapi import FastAPI
import uvicorn

from Image.image_ctrler import router as image_router

app = FastAPI(title="API NLP¨")

app.include_router(Image)
app.include_router(NLP)

if __name__ == "__main__":

    uvicorn.run(app, host="localhost", port=8000)