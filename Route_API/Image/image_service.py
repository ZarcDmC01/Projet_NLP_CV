from fastapi import APIRouter

from Route_API.Image.image import image

img = image()

router = APIRouter(tags=['image_service'])
