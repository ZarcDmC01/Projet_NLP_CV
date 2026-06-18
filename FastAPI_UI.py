from fastapi import FastAPI, Request
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

@app.get('/legend')
async def legend(request: Request):
    return templates.TemplateResponse(request=request, name="legend.html")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)