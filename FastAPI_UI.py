from fastapi import FastAPI
import uvicorn

from Image.image_ctrler import router as image_router
from NLP.NLP_ctrler import router as NLP_router
from Security.Security_ctrler import router as Security_router
from Monitoring.Monitoring_ctrler import router as Monitor_ctrler

app = FastAPI(title="API NLP¨")

app.include_router(image_router)
app.include_router(NLP_router)
app.include_router(Security_router)
app.include_router(Monitor_ctrler)

# @wrapper Security

@app.get('/index')
async def index():
    pass
    return (render_template('index.html'))

@app.get('/load_image')
async def load_image():
    pass
    return (render_templates('load_image.html'))

@app.get('/legend')
async def get_legend():
    pass
    return (render_templates('legend.html'))


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)