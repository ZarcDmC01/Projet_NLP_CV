from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI(title="Commentary_UI")
templates = Jinja2Templates(directory="templates")

@app.get('/')
async def root():
    return RedirectResponse(url='/login')

@app.get('/index')
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get('/login')
async def login(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")

@app.get('/signup')
async def signup(request: Request):
    return templates.TemplateResponse(request=request, name="signup.html")

@app.get('/load_image')
async def load_image(request: Request):
    return templates.TemplateResponse(request=request, name="load_image.html")

@app.post('/legend')
async def get_legend(request: Request, file: UploadFile = File(...)):
    return templates.TemplateResponse(request=request, name="legend.html", context={"legend": "Légende à venir...", "image_url": None})


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)


print('July ne sert à rien')