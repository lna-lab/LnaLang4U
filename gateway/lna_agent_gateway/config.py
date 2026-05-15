import os
from typing import Optional


class Settings:
    upstream_base_url: str = os.environ.get("UPSTREAM_OPENAI_BASE_URL", "http://127.0.0.1:9000/v1")
    upstream_api_key: str = os.environ.get("UPSTREAM_API_KEY", "local-dev")
    host: str = os.environ.get("LNA_GATEWAY_HOST", "0.0.0.0")
    port: int = int(os.environ.get("LNA_GATEWAY_PORT", "9010"))
    served_model: str = os.environ.get("LNA_SERVED_MODEL", "deepseek-v4-flash")
    context_length: int = int(os.environ.get("LNA_CONTEXT_LENGTH", "1048576"))
    model_aliases_raw: str = os.environ.get("LNA_MODEL_ALIASES", "")

    @property
    def model_aliases(self) -> dict[str, str]:
        aliases = {}
        for pair in self.model_aliases_raw.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                aliases[k.strip()] = v.strip()
        return aliases

    def resolve_model(self, model: Optional[str]) -> str:
        if not model:
            return self.served_model
        return self.model_aliases.get(model, model)


settings = Settings()
