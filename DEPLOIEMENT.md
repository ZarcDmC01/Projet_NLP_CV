# Déploiement de l'application NLP & CV — Retour d'expérience

## Contexte

L'application se compose de deux services FastAPI distincts :
- **FastAPI_UI.py** : interface utilisateur avec templates Jinja2 (login, signup, upload d'image, affichage de légende)
- **FastAPI_API.py** : backend ML avec PyTorch (ResNet34 + LSTM), authentification JWT, base de données SQLite

L'objectif était de rendre l'application accessible en ligne depuis un dépôt GitHub unique (`ZarcDmC01/Projet_NLP_CV`, branche `Jesse`).

---

## Architecture finale retenue

```
Utilisateur
    │
    ▼
Vercel (FastAPI_UI.py)          ← UI, templates HTML, pas de ML
    │  API_URL env var
    ▼
Render (FastAPI_API.py)         ← ML, auth, SQLite, ResNet34 + LSTM
```

- **Vercel** héberge uniquement l'UI (léger, serverless Python)
- **Render** héberge l'API ML (Docker-like, supporte les gros packages Python)
- **Un seul repo GitHub** surveille les deux déploiements

---

## Pistes explorées

### Railway.app
- Envisagé comme alternative à Render pour l'API
- Écarté car payant sans tier gratuit suffisant pour un projet de démo

### Vercel seul (UI + API)
- Tentative initiale de tout mettre sur Vercel
- **Echec** : Vercel utilise `uv` comme resolver de dépendances, qui ne supporte pas correctement `--extra-index-url` (PyTorch CPU). L'installation de `torch` échouait systématiquement
- **Conclusion** : Vercel n'est pas adapté pour du ML avec PyTorch

### Deux repos GitHub séparés
- Vercel avait créé automatiquement un repo `projet-nlp-cv` (avec tirets) lors de son flow "Clone" au lieu d'importer `Projet_NLP_CV` (avec underscores)
- Les commits allaient vers `Projet_NLP_CV` mais Render regardait `projet-nlp-cv` → désynchronisation totale
- **Résolution** : reconnexion manuelle de Render au bon repo, suppression du repo dupliqué

---

## Problèmes rencontrés et solutions

### 1. PyTorch introuvable sur Vercel (uv + extra-index-url)

**Problème** : `uv` (le resolver Vercel) cherchait `numpy==2.4.6` sur le PyTorch index (`download.pytorch.org/whl/cpu`) où il n'existe pas, bloquant toute l'installation.

**Solution** : Séparer les dépendances en deux fichiers :
- `requirements.txt` → UI uniquement (pas de PyTorch, pas d'`--extra-index-url`)
- `requirements-api.txt` → API avec PyTorch CPU (`--extra-index-url https://download.pytorch.org/whl/cpu`)

> Note : le paramètre `requirementsFile` dans `vercel.json` (`"config": {"requirementsFile": "..."}`) **ne fonctionne pas** — Vercel lit toujours `requirements.txt`. La vraie solution est de rendre `requirements.txt` UI-only.

---

### 2. Python 3.14 sur Render — pas de wheel PyTorch

**Problème** : Render utilisait Python 3.14 par défaut. PyTorch n'a pas de wheel pour cette version → installation impossible.

**Solution** : Ajouter la variable d'environnement `PYTHON_VERSION=3.11.9` dans le dashboard Render.

> Note : un fichier `.python-version` à la racine **ne fonctionne pas** sur Render — il est ignoré. De plus, ce fichier entrait en conflit avec Vercel (qui générait un `pyproject.toml` exigeant `==3.12.*`). Il a donc été supprimé.

---

### 3. Build command Render — mauvais fichier de requirements

**Problème** : Render utilisait `pip install -r requirements.txt` (UI only), donc `torch`, `torchvision`, `pandas`, `nltk` manquaient.

**Solution** : Changer le build command dans le dashboard Render :
```
pip install -r requirements-api.txt
```

---

### 4. URL de l'API non transmise à l'UI

**Problème** : Les templates HTML appelaient `http://localhost:8000` en dur → ne fonctionnait pas en production.

**Solution** :
- `FastAPI_UI.py` lit la variable d'environnement `API_URL` avec fallback `localhost:8000` pour le dev local :
  ```python
  API_URL = os.environ.get("API_URL", "http://localhost:8000")
  ```
- Chaque route passe `api_url` au template : `context={"api_url": API_URL}`
- `base.html` injecte la valeur en JS : `<script>const API_URL = "{{ api_url }}";</script>`
- Les templates remplacent les URLs hardcodées par `` `${API_URL}/...` ``
- Variable `API_URL=https://projet-nlp-cv.onrender.com` configurée dans Vercel pour **tous les environnements** (Production + Preview + Development)

---

### 5. Variable d'env Vercel — Production only

**Problème** : `API_URL` était définie uniquement pour l'environnement "Production" mais la branche `Jesse` tournait en **Preview** → variable non lue, fallback `localhost:8000`.

**Solution** : Modifier la variable dans Vercel pour qu'elle s'applique aux 3 environnements. La variable "Sensitive" ne peut pas être éditée après création → suppression et recréation sans le toggle Sensitive.

---

### 6. CORS — URLs de preview Vercel non autorisées

**Problème** : L'URL de preview Vercel (`projet-nlp-cv-git-jesse-jessesteven26-4590s-projects.vercel.app`) n'était pas dans la liste CORS de l'API Render → browser bloquait les requêtes.

**Diagnostic** : Erreur "Failed to fetch" dans la console browser, masquant le vrai problème (la réponse 500 n'avait pas de headers CORS → browser la rejetait).

**Solution** : Utiliser `allow_origin_regex` dans le middleware CORS de FastAPI pour couvrir toutes les URLs Vercel dynamiquement :
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001", "https://projetnlpcv.vercel.app"],
    allow_origin_regex=r"https://projet-nlp-cv.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)
```

> Attention : le préfixe de l'URL Vercel est `projet-nlp-cv` (avec tirets), pas `projetnlpcv` (sans tirets). Le regex initial était incorrect.

---

### 7. SQLite — chemin relatif cassé sur Render

**Problème** : `DATABASE_URL = "sqlite:///./nlp_cv.db"` utilise un chemin relatif au répertoire courant. Sur Render, le CWD peut différer selon le contexte → 500 Internal Server Error sur tous les endpoints d'authentification.

**Solution** : Chemin absolu basé sur l'emplacement de `models.py` :
```python
_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nlp_cv.db")
DATABASE_URL = f"sqlite:///{_db_path}"
```

> Note : SQLite sur Render free tier est **éphémère** — la base est remise à zéro à chaque redéploiement. Pour un projet de démo c'est acceptable ; pour la production il faudrait PostgreSQL.

---

### 8. bcrypt 4.x incompatible avec passlib 1.7.4

**Problème** : `passlib[bcrypt]==1.7.4` sans version fixée pour `bcrypt` installait bcrypt 4.x, qui a supprimé l'attribut `__about__` utilisé par passlib → 500 sur `/auth/register` et `/auth/login`.

**Solution** : Pincer bcrypt à une version compatible dans `requirements-api.txt` :
```
passlib[bcrypt]==1.7.4
bcrypt==3.2.2
```

---

### 9. Réponses 500 sans headers CORS

**Problème** : Quand FastAPI/Starlette levait une exception non gérée, le middleware CORS ne s'appliquait pas à la réponse d'erreur → le browser voyait "Failed to fetch" au lieu du vrai message d'erreur, rendant le debug impossible.

**Solution** : Ajouter un handler global d'exception qui retourne du JSON (donc avec CORS headers) :
```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
```

---

### 10. Chemin du features store — dossier Data inexistant sur Render

**Problème** : `image_service.py` essayait d'écrire les features extraites dans `Data/features_api.pkl`. Le dossier `Data/` est gitignore et n'existe pas sur Render → 500 à l'upload d'image.

**Solution** : Utiliser `/tmp/` sur Linux (Render), sinon la racine du projet :
```python
FEATURES_STORE = (
    Path("/tmp/features_api.pkl") if Path("/tmp").exists()
    else Path(__file__).parent.parent.parent / "features_api.pkl"
)
```

---

## Fichiers créés ou modifiés pour le déploiement

| Fichier | Action | Raison |
|---|---|---|
| `requirements.txt` | Modifié | UI uniquement, sans PyTorch ni extra-index-url |
| `requirements-api.txt` | Créé | API avec torch CPU, bcrypt pinné |
| `vercel.json` | Créé | Route tout vers FastAPI_UI.py |
| `.vercelignore` | Créé | Exclut API, modèles, notebooks de Vercel |
| `FastAPI_UI.py` | Modifié | API_URL via env var, chemin absolu templates |
| `FastAPI_API.py` | Modifié | CORS regex Vercel, global exception handler |
| `models.py` | Modifié | Chemin absolu SQLite |
| `models/caption_model_best.pth` | Ajouté au repo | Modèle LSTM (19MB), nécessaire sur Render |
| `vocab/Flickr8k.token.txt` | Ajouté au repo | Vocabulaire (3.2MB), nécessaire sur Render |
| `Route_API/Model/model.py` | Modifié | Chemin absolu vers models/ |
| `Route_API/Model/model_service.py` | Modifié | Chemin absolu vers vocab/ |
| `Route_API/Image/image_service.py` | Modifié | Features store dans /tmp/ sur Linux |
| `.gitignore` | Modifié | Ajout de *.db |

---

## Variables d'environnement nécessaires

### Vercel
| Variable | Valeur | Environnements |
|---|---|---|
| `API_URL` | `https://projet-nlp-cv.onrender.com` | Production + Preview + Development |

### Render
| Variable | Valeur |
|---|---|
| `PYTHON_VERSION` | `3.11.9` |

---

## Résultat final

- **UI Vercel** : déploiement serverless, build en 13s, Preview sur branche Jesse
- **API Render** : LSTM chargé, ResNet34 téléchargé au démarrage, SQLite créé à la volée
- Pipeline complet fonctionnel : login → upload image → extraction ResNet34 → génération LSTM → affichage légende
