from fastapi import APIRouter

from Monitoring import Monitoring

Monitor = Monitoring()

router = APIRouter(['Monitoring_service'])