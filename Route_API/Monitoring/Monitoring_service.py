from fastapi import APIRouter

from Route_API.Monitoring.Monitoring import Monitoring

Monitor = Monitoring()

router = APIRouter(tags=['Monitoring_service'])