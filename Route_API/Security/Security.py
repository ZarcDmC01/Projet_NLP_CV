# class Security():
#     def __init__(self):
#         pass




from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
from typing import Optional
import uuid

from models import User, UserProfile, StorageQuota, UserRole, AccountStatus, SessionLocal

# Configuration
SECRET_KEY = "ta-clé-secrète-super-secure-change-moi-en-prod-🔐"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 10080  # 7 jours

# Setup password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()


# ====== UTILITIES ======

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        return user_id
    except JWTError:
        return None


# ====== DEPENDENCIES ======

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    token = credentials.credentials
    user_id = decode_token(token)

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Utilisateur non trouvé",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


# ====== SCHEMAS PYDANTIC ======

from pydantic import BaseModel


class SignupRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


# ====== FONCTIONS AUTH ======

def register_user(db: Session, signup_data: SignupRequest) -> TokenResponse:
    existing_user = db.query(User).filter(
        (User.username == signup_data.username) | (User.email == signup_data.email)
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cet utilisateur ou email existe déjà"
        )

    user_id = str(uuid.uuid4())

    user = User(
        id=user_id,
        username=signup_data.username,
        email=signup_data.email,
        hashed_password=get_password_hash(signup_data.password),
        role=UserRole.USER,
        status=AccountStatus.ACTIVE,
        email_verified=False
    )

    profile = UserProfile(
        id=str(uuid.uuid4()),
        user_id=user_id,
        language="fr",
        timezone="UTC"
    )

    quota = StorageQuota(
        id=str(uuid.uuid4()),
        user_id=user_id,
        limit_bytes=5 * 1024 * 1024 * 1024  # 5GB
    )

    db.add(user)
    db.add(profile)
    db.add(quota)
    db.commit()
    db.refresh(user)

    access_token = create_access_token(data={"sub": user.id})
    return TokenResponse(access_token=access_token, token_type="bearer")


def login_user(db: Session, login_data: LoginRequest) -> TokenResponse:
    user = db.query(User).filter(User.username == login_data.username).first()

    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nom d'utilisateur ou mot de passe incorrect"
        )

    access_token = create_access_token(data={"sub": user.id})
    return TokenResponse(access_token=access_token, token_type="bearer")
