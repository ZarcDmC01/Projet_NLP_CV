from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel
from typing import Optional
import uuid

from models import User, UserProfile, StorageQuota, UserRole, AccountStatus, SessionLocal

# Déclarés au niveau module car requis par Depends() à la définition de la classe
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer = HTTPBearer()


def _get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ================================================================
#  SCHEMAS PYDANTIC
# ================================================================

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


# ================================================================
#  CLASSE SECURITY — une méthode = une action
# ================================================================

class Security:

    SECRET_KEY = "ta-clé-secrète-super-secure-change-moi-en-prod"
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 10080  # 7 jours

    get_db = staticmethod(_get_db)

    # ================================================================
    #  1. MOT DE PASSE
    # ================================================================

    @staticmethod
    def get_password_hash(password: str) -> str:
        """Transforme le mot de passe en hash bcrypt."""
        return _pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Compare le mot de passe saisi avec le hash stocké en base."""
        return _pwd_context.verify(plain_password, hashed_password)

    # ================================================================
    #  2. TOKEN JWT
    # ================================================================

    @classmethod
    def create_access_token(cls, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Génère un token JWT signé avec expiration."""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=cls.ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, cls.SECRET_KEY, algorithm=cls.ALGORITHM)

    @classmethod
    def decode_token(cls, token: str) -> Optional[str]:
        """Décode le token JWT et retourne l'user_id (sub), None si invalide."""
        try:
            payload = jwt.decode(token, cls.SECRET_KEY, algorithms=[cls.ALGORITHM])
            return payload.get("sub")
        except JWTError:
            return None

    # ================================================================
    #  3. BASE DE DONNÉES
    # ================================================================

    @staticmethod
    def create_user(db: Session, signup_data: SignupRequest, hashed_password: str) -> User:
        """Crée un utilisateur, son profil et son quota en base."""
        existing = db.query(User).filter(
            (User.username == signup_data.username) | (User.email == signup_data.email)
        ).first()

        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cet utilisateur ou email existe déjà"
            )

        user_id = str(uuid.uuid4())
        user = User(
            id=user_id,
            username=signup_data.username,
            email=signup_data.email,
            hashed_password=hashed_password,
            role=UserRole.USER,
            status=AccountStatus.ACTIVE,
            email_verified=False
        )
        db.add(user)
        db.add(UserProfile(id=str(uuid.uuid4()), user_id=user_id, language="fr", timezone="UTC"))
        db.add(StorageQuota(id=str(uuid.uuid4()), user_id=user_id, limit_bytes=5 * 1024 * 1024 * 1024))
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """Recherche un utilisateur par son nom d'utilisateur."""
        return db.query(User).filter(User.username == username).first()

    @staticmethod
    def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
        """Recherche un utilisateur par son identifiant unique."""
        return db.query(User).filter(User.id == user_id).first()

    # ================================================================
    #  4. DÉPENDANCE FASTAPI — routes protégées
    # ================================================================

    @staticmethod
    async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(_bearer),
        db: Session = Depends(_get_db)
    ) -> User:
        """Vérifie le token Bearer et retourne l'utilisateur connecté."""
        user_id = Security.decode_token(credentials.credentials)

        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token invalide ou expiré",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user = Security.get_user_by_id(db, user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Utilisateur non trouvé",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
