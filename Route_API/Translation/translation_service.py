import os

MODEL_NAME = "llama-3.3-70b-versatile"

_SYSTEM_PROMPT = (
    "Tu es un traducteur. Traduis le texte suivant de l'anglais vers le français. "
    "Réponds uniquement avec la traduction, sans commentaire ni guillemets."
)


class TranslationService:
    """Traduit une légende anglaise en français via l'API Groq (Llama 3.3).
    Dégrade proprement (retourne None) si GROQ_API_KEY est absent ou si l'appel échoue,
    pour ne jamais faire échouer la génération de légende elle-même."""

    _client = None

    @classmethod
    def _get_client(cls):
        if cls._client is None:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                return None
            from groq import Groq
            cls._client = Groq(api_key=api_key)
        return cls._client

    @classmethod
    def translate_to_french(cls, text: str) -> str | None:
        client = cls._get_client()
        if client is None or not text:
            return None
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.2,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            print(f"[TranslationService] Échec traduction Groq : {exc}")
            return None
