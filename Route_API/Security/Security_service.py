from fastapi import APIRouter

from Route_API.Security.Security import Security

Secu = Security()

router = APIRouter(tags=['Security_service'])