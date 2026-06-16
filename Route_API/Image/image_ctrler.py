from fastapi import APIRouter, Depends, File, UploadFile

from Route_API.Image.image_service import ImageService
from Route_API.Security.Security import Security

router = APIRouter(prefix="/image", tags=["Image"])


@router.post("/")
async def upload_image(
    file: UploadFile = File(...),
    current_user=Depends(Security.get_current_user)
):
    """
    Reçoit une image uploadée, exécute automatiquement tout le pipeline :
    chargement → redimensionnement → normalisation → extraction de features.
    Accessible uniquement aux utilisateurs connectés.
    """
    image_bytes = await file.read()
    result = ImageService.process_image(image_bytes)
    return result
