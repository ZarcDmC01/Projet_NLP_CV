from fastapi import FastAPI, Request, UploadFile, File
import uvicorn

from Route_API.Image.image_ctrler import router as image_router
from Route_API.NLP.NLP_ctrler import router as NLP_router
from Route_API.Security.Security_ctrler import router as Security_router
from Route_API.Monitoring.Monitoring_ctrler import router as Monitor_ctrler

app = FastAPI(title="Commentary_API")

app.include_router(image_router)
app.include_router(NLP_router)
app.include_router(Security_router)
app.include_router(Monitor_ctrler)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)