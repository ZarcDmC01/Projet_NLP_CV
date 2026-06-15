from fastapi import APIRouter

from Route_API.NLP.NLP import NLP

nlp = NLP()

router = APIRouter(tags=['NLP_service'])

