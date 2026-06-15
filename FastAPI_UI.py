from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import uvicorn

from Image.image_ctrler import router as image_router
from NLP.NLP_ctrler import router as NLP_router
from Security.Security_ctrler import router as Security_router
from Monitoring.Monitoring_ctrler import router as Monitor_ctrler

app = FastAPI(title="API NLP")
templates = Jinja2Templates(directory="templates")

app.include_router(image_router)
app.include_router(NLP_router)
app.include_router(Security_router)
app.include_router(Monitor_ctrler)

@app.get('/index')
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/load_image')
async def load_image(request: Request):
    return templates.TemplateResponse("load_image.html", {"request": request})

@app.get('/legend')
async def get_legend(request: Request):
    return templates.TemplateResponse("legend.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)