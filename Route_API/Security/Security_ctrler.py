from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from Route_API.Security.Security import Security, SignupRequest, LoginRequest, TokenResponse
from Route_API.Security.Security_service import SecurityService

router = APIRouter(prefix="/auth", tags=["Security"])


@router.post("/register", response_model=TokenResponse)
def register(signup_data: SignupRequest, db: Session = Depends(Security.get_db)):
    return SecurityService.register(db, signup_data)


@router.post("/login", response_model=TokenResponse)
def login(login_data: LoginRequest, db: Session = Depends(Security.get_db)):
    return SecurityService.login(db, login_data)
