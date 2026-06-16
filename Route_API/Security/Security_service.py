from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from Route_API.Security.Security import Security, SignupRequest, LoginRequest, TokenResponse


class SecurityService:

    # ====== INSCRIPTION ======
    # 1. get_password_hash  →  2. create_user  →  3. create_access_token

    @staticmethod
    def register(db: Session, signup_data: SignupRequest) -> TokenResponse:
        hashed = Security.get_password_hash(signup_data.password)
        user = Security.create_user(db, signup_data, hashed)
        token = Security.create_access_token({"sub": user.id})
        return TokenResponse(access_token=token, token_type="bearer")

    # ====== CONNEXION ======
    # 1. get_user_by_username  →  2. verify_password  →  3. create_access_token

    @staticmethod
    def login(db: Session, login_data: LoginRequest) -> TokenResponse:
        user = Security.get_user_by_username(db, login_data.username)

        if not user or not Security.verify_password(login_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Nom d'utilisateur ou mot de passe incorrect"
            )

        token = Security.create_access_token({"sub": user.id})
        return TokenResponse(access_token=token, token_type="bearer")

    # ====== APPEL API PROTÉGÉ ======
    # Géré directement via Depends(Security.get_current_user) sur les routes
    # 1. decode_token  →  2. get_user_by_id (vérifie en base)
