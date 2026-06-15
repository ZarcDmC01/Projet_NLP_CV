# Démarrage du projet

## 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 2. Démarrer les serveurs

Ouvrir **deux terminaux** dans le dossier du projet.

**Terminal 1 — API (port 8000) :**
```bash
python FastAPI_API.py
```

**Terminal 2 — UI (port 8001) :**
```bash
python FastAPI_UI.py
```

## 3. Accéder à l'application

| URL | Description |
|-----|-------------|
| http://localhost:8001/index | Interface utilisateur |
| http://localhost:8001/load_image | Charger une image |
| http://localhost:8000/docs | Documentation API (Swagger) |

## 4. Surveillance automatique (optionnel)

Lance l'autowatcher pour auto-commiter/pusher dès qu'un fichier surveillé est modifié :

```bash
python .github/workflows/autowatcher.py
```
