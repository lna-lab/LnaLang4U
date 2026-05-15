from .config import settings


def get_models():
    """Return OpenAI-style model list with aliases."""
    models = [
        {
            "id": settings.served_model,
            "object": "model",
            "created": 1767225600,
            "owned_by": "lnalang4u",
            "max_model_len": settings.context_length,
        }
    ]
    for alias in settings.model_aliases:
        models.append({
            "id": alias,
            "object": "model",
            "created": 1767225600,
            "owned_by": "lnalang4u",
            "max_model_len": settings.context_length,
        })
    return models


def get_anthropic_models():
    """Return Anthropic-style model list."""
    models = [{
        "type": "model",
        "id": settings.served_model,
        "display_name": "DeepSeek V4 Flash (via LnaLang4U)",
        "created_at": "2026-05-15T00:00:00Z",
    }]
    for alias in settings.model_aliases:
        models.append({
            "type": "model",
            "id": alias,
            "display_name": f"{alias} (alias for {settings.served_model})",
            "created_at": "2026-05-15T00:00:00Z",
        })
    return {"data": models}
