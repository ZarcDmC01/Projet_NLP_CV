import os
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI(title="Commentary_UI")

_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(_dir, "templates"))

API_URL = os.environ.get("API_URL", "http://localhost:8000")


@app.get('/')
async def root():
    return RedirectResponse(url='/login')

@app.get('/index')
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"api_url": API_URL})

@app.get('/login')
async def login(request: Request):
    return templates.TemplateResponse(request=request, name="login.html", context={"api_url": API_URL})

@app.get('/signup')
async def signup(request: Request):
    return templates.TemplateResponse(request=request, name="signup.html", context={"api_url": API_URL})

@app.get('/load_image')
async def load_image(request: Request):
    return templates.TemplateResponse(request=request, name="load_image.html", context={"api_url": API_URL})

@app.get('/legend')
async def legend(request: Request):
    return templates.TemplateResponse(request=request, name="legend.html", context={"api_url": API_URL})


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)
