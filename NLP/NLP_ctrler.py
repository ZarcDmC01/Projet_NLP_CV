from fastapi import APIRouter

from NLP.NLP_service import router as NLP_service

router = APIRouter(tags = ['NLP_ctrler'])

@router.get('/prompt')
async def get_prompt():
    pass