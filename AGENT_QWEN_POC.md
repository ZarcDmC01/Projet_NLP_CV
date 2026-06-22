# Agent Qwen Autonome — Document de préparation POC

## Vision

Un agent IA autonome, auto-améliorant, tournant h/24 gratuitement dans le cloud.
Pas de GPU, pas de coût, pas besoin que ton PC soit allumé.

---

## Stack retenue

| Brique | Service | Coût |
|---|---|---|
| Modèle Qwen | Hugging Face Hub | Gratuit |
| Agent / API | Render (Background Worker) | Gratuit |
| Orchestration pipelines | n8n sur Render | Gratuit |
| Keep-alive | UptimeRobot | Gratuit |
| UI (optionnel) | Vercel | Gratuit |
| Code | GitHub | Gratuit |
| **Total** | | **0€** |

---

## Architecture

```
GitHub (code)
    ↓ push → redéploiement auto
Render Worker (Qwen API)          Render Service (n8n)
    ↑                                   ↓
    └───────── pipeline n8n ────────────┘
                    ↑
            UptimeRobot (ping /10 min)
                    
Vercel UI (optionnel, si interface nécessaire)
```

---

## Choix du modèle

**Qwen 1.5B** → recommandé pour démarrer
- ~3GB RAM quantifié 4bit (dans les clous Render Starter si nécessaire)
- Suffisant pour raisonnement simple, résumé, génération de code court
- Disponible sur Hugging Face : `Qwen/Qwen1.5-1.8B-Chat`

**Alternative si RAM trop juste :**
- Appel API Qwen (Alibaba Cloud) → zéro RAM, ~gratuit pour faibles volumes
- ou `Qwen/Qwen1.5-0.5B-Chat` → ~1GB

---

## Ce que fait l'agent (scope POC)

- [ ] Reçoit un prompt / contexte via n8n
- [ ] Génère une réponse avec Qwen
- [ ] Évalue sa propre réponse (auto-critique simple)
- [ ] Si score insuffisant → rejoue avec prompt amélioré (1-2 itérations max)
- [ ] Retourne le résultat final (log, webhook, stockage)

> L'auto-amélioration ici = boucle de réflexion interne, pas de fine-tuning du modèle.  
> Le fine-tuning vient dans une v2 si le POC est concluant.

---

## Fichiers à créer (beaucoup moins que le projet NLP/CV)

```
agent/
├── main.py              ← endpoint FastAPI /generate
├── agent.py             ← logique agent (prompt → réponse → auto-critique)
├── model.py             ← chargement Qwen depuis HuggingFace
└── requirements.txt     ← transformers, fastapi, uvicorn, torch+cpu
```

Pas de :
- templates HTML
- base de données
- authentification
- split requirements UI/API

---

## Points techniques à ne pas oublier (leçons du projet NLP/CV)

- **Chemins absolus** partout (`os.path.abspath(__file__)`)
- **PYTHON_VERSION=3.11.9** sur Render
- **CORS** si UI Vercel → `allow_origin_regex`
- **Build command Render** : `pip install -r requirements.txt`
- **torch CPU** : `torch==2.x.x+cpu` + `--extra-index-url https://download.pytorch.org/whl/cpu`
- **Global exception handler** → erreurs lisibles dès le début

---

## Étapes POC

1. **Tester Qwen en local** → charger le modèle, générer une réponse, mesurer la RAM
2. **Créer le repo GitHub** + structure de base
3. **Déployer l'API sur Render** → endpoint `/generate` qui répond
4. **Installer n8n sur Render** → workflow simple (cron → appel API → log)
5. **Brancher UptimeRobot** → les deux services restent éveillés
6. **Implémenter la boucle auto-critique** dans `agent.py`
7. **Tester end-to-end** → trigger → agent → réponse améliorée

---

## Différence avec le projet NLP/CV

| | NLP/CV | Agent Qwen |
|---|---|---|
| Complexité déploiement | Élevée (2 stacks, ML lourd) | Faible (1 worker suffit) |
| Split requirements | Obligatoire | Probablement pas nécessaire |
| UI | Jinja2 + templates | Optionnelle |
| Auth | JWT + SQLite | Pas nécessaire pour le POC |
| Modèle | Entraîné custom (LSTM) | Pré-entraîné HuggingFace |
| Galère estimée | 10/10 | 3/10 |

---

## Pour plus tard (si POC concluant)

- Fine-tuning Qwen sur tes données → Hugging Face Training API
- Mémoire persistante → base vectorielle (ChromaDB, Qdrant free tier)
- Multi-agents → plusieurs workers Render spécialisés
- Interface sérieuse → Vercel UI
- Auto-amélioration réelle → l'agent push son propre code sur GitHub → redéploiement auto

---

> Test auto-improve Claude → SAO HUD (déjà en place)  
> Ce POC = version open source / self-hosted de la même idée
