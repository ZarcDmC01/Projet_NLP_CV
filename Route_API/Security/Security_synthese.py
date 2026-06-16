# =============================================================================
#  SYNTHÈSE SÉCURITÉ — Comparaison TUTURE (Flask) vs Projet NLP/CV (FastAPI)
# =============================================================================


# =============================================================================
#  1. HASH DU MOT DE PASSE
#  Objectif : ne jamais stocker le mot de passe en clair dans la base
# =============================================================================

# --- TUTURE (Flask) ---
# from werkzeug.security import generate_password_hash, check_password_hash
#
# Au moment de l'inscription :
#   hashed = generate_password_hash("monMotDePasse")  → "$2b$12$K8Zx9m..."
#
# Au moment du login :
#   check_password_hash(hash_stocké, "monMotDePasse") → True ou False

# --- NLP/CV (FastAPI) --- même principe, librairie plus robuste
# from passlib.context import CryptContext
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
#
# get_password_hash("monMotDePasse")              → hash bcrypt
# verify_password("monMotDePasse", hash_stocké)  → True ou False

# Différence : passlib/bcrypt est plus configurable que werkzeug.
# Le résultat est identique : un hash à sens unique, non déchiffrable.


# =============================================================================
#  2. GÉNÉRATION DU TOKEN JWT
#  Objectif : prouver l'identité de l'utilisateur sans session serveur
# =============================================================================

# --- TUTURE ---
# import jwt, datetime
#
# def generate_token(username):
#     payload = {
#         "user": username,
#         "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
#     }
#     return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
#
# Durée : 2 heures
# Payload : { "user": "email@exemple.com" }

# --- NLP/CV ---
# from jose import jwt
#
# def create_access_token(data: dict, expires_delta=None):
#     to_encode = data.copy()
#     expire = datetime.utcnow() + (expires_delta or timedelta(minutes=10080))
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
#
# Durée : 7 jours (10080 minutes)
# Payload : { "sub": user.id }   ← "sub" = convention standard JWT

# Différence clé :
#   TUTURE  → stocke le username (email) dans le token
#   NLP/CV  → stocke l'id unique de l'utilisateur (plus sécurisé)


# =============================================================================
#  3. DÉCODAGE DU TOKEN / VÉRIFICATION
#  Objectif : lire le token reçu et trouver l'utilisateur correspondant
# =============================================================================

# --- TUTURE ---
# def token_required(f):                   ← décorateur Flask manuel
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         token = request.headers["Authorization"].split(" ")[1]
#         try:
#             jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
#         except jwt.ExpiredSignatureError:
#             return jsonify({"error": "Token expired"}), 401
#         except jwt.InvalidTokenError:
#             return jsonify({"error": "Invalid token"}), 401
#         return f(*args, **kwargs)
#     return decorated
#
# Limite : on vérifie juste que le token est valide,
#          mais on ne récupère pas l'objet User dans la base.

# --- NLP/CV ---
# async def get_current_user(credentials=Depends(security), db=Depends(get_db)):
#     token = credentials.credentials
#     user_id = decode_token(token)              ← extrait l'id du payload
#     if user_id is None:
#         raise HTTPException(401, "Token invalide ou expiré")
#     user = db.query(User).filter(User.id == user_id).first()  ← cherche en base
#     if user is None:
#         raise HTTPException(401, "Utilisateur non trouvé")
#     return user                                ← renvoie l'objet User complet
#
# Avantage : on récupère l'utilisateur complet, utilisable dans chaque route.
# FastAPI injecte automatiquement via Depends() — pas besoin de décorateur.


# =============================================================================
#  4. ROUTE LOGIN
#  Objectif : vérifier les identifiants et renvoyer un token
# =============================================================================

# --- TUTURE ---
# @app.route("/api/login", methods=["POST"])
# def login():
#     data = request.get_json()
#     email, password = data.get("username"), data.get("password")
#
#     conn = sqlite3.connect("database.db")      ← connexion SQLite brute
#     cur.execute("SELECT password FROM users WHERE email = ?", (email,))
#     user = cur.fetchone()
#     conn.close()
#
#     if user and check_password_hash(user[0], password):
#         return jsonify({"token": generate_token(email)})
#     return jsonify({"error": "Invalid credentials"}), 401

# --- NLP/CV ---
# def login_user(db: Session, login_data: LoginRequest) -> TokenResponse:
#     user = db.query(User).filter(User.username == login_data.username).first()
#     if not user or not verify_password(login_data.password, user.hashed_password):
#         raise HTTPException(401, "Nom d'utilisateur ou mot de passe incorrect")
#     token = create_access_token(data={"sub": user.id})
#     return TokenResponse(access_token=token, token_type="bearer")
#
# Différence : SQLAlchemy ORM au lieu de SQL brut → plus lisible et sécurisé
#              (protection automatique contre les injections SQL)


# =============================================================================
#  5. ROUTE INSCRIPTION (register)
#  Présente uniquement dans NLP/CV — absente dans TUTURE Flask_API.py
# =============================================================================

# --- NLP/CV ---
# def register_user(db: Session, signup_data: SignupRequest) -> TokenResponse:
#     existing = db.query(User).filter(
#         (User.username == signup_data.username) | (User.email == signup_data.email)
#     ).first()
#     if existing:
#         raise HTTPException(400, "Cet utilisateur ou email existe déjà")
#
#     user = User(
#         id=str(uuid.uuid4()),
#         username=signup_data.username,
#         email=signup_data.email,
#         hashed_password=get_password_hash(signup_data.password),
#         role=UserRole.USER,
#         status=AccountStatus.ACTIVE,
#     )
#     db.add(user) ; db.commit()
#     token = create_access_token(data={"sub": user.id})
#     return TokenResponse(access_token=token, token_type="bearer")


# =============================================================================
#  6. PROTECTION D'UNE ROUTE
#  Objectif : bloquer l'accès si l'utilisateur n'est pas connecté
# =============================================================================

# --- TUTURE ---
# @app.route("/api/estimate", methods=['POST'])
# @token_required                          ← décorateur placé sur la route
# def estimate_api():
#     ...

# --- NLP/CV ---
# @router.post('/image')
# async def post_image(current_user = Depends(get_current_user)):
#     ...                                  ← current_user disponible ici
#
# Avantage FastAPI : current_user contient l'objet User complet
#                   → on peut vérifier son rôle, son statut, etc.


# =============================================================================
#  7. SCHÉMAS DES DONNÉES (Pydantic) — absent dans TUTURE
#  Objectif : valider automatiquement les données reçues par l'API
# =============================================================================

# --- NLP/CV uniquement ---
# class SignupRequest(BaseModel):
#     username: str
#     email: str
#     password: str
#
# class LoginRequest(BaseModel):
#     username: str
#     password: str
#
# class TokenResponse(BaseModel):
#     access_token: str
#     token_type: str
#
# FastAPI valide automatiquement que les champs sont présents et du bon type.
# Si un champ manque → 422 Unprocessable Entity renvoyé automatiquement.


# =============================================================================
#  RÉSUMÉ COMPARATIF
# =============================================================================
#
#  Fonction              TUTURE (Flask)          NLP/CV (FastAPI)
#  ──────────────────────────────────────────────────────────────────────
#  Hash password         werkzeug                passlib/bcrypt (robuste)
#  Génération token      PyJWT, 2h, email        python-jose, 7j, user.id
#  Vérif token           décorateur manuel       Depends(get_current_user)
#  Récupération user     non (juste le token)    oui (objet User complet)
#  Base de données       SQLite brut             SQLAlchemy ORM
#  Validation données    manuelle                Pydantic automatique
#  Inscription           absente                 register_user()
#  Doc API               Flasgger YAML manuel    OpenAPI auto /docs
#  Protection route      @token_required         Depends(get_current_user)
# =============================================================================
