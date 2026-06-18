from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from Route_API.Model.model_service import ModelService
from Route_API.Security.Security import Security

router = APIRouter(prefix="/model", tags=["Model"])


class CaptionRequest(BaseModel):
    image_id: str  # retourné par POST /image/


@router.post("/caption")
async def generate_caption(
    body: CaptionRequest,
    _=Depends(Security.get_current_user),
):
    result = ModelService.generate_caption(body.image_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"image_id '{body.image_id}' introuvable.")
    return result


@router.get("/status")
async def model_status():
    return ModelService.get_status()
