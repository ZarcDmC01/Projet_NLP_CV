from fastapi import APIRouter, FastAPI

router = APIRouter(tags=['Image'])

@router.post('/image')
async def post_image():
    pass