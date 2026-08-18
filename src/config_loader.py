import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class ConfigLoader:
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = os.path.join(Path(__file__).parent.parent, "config")
        self.config_dir = Path(config_dir)
        self.domain_config = self._load_yaml("domain_config.yaml")
        self.retrieval_config = self._load_yaml("retrieval_config.yaml")
        self.security_config = self._load_yaml("security_policy.yaml")

    def _load_yaml(self, filename: str) -> dict:
        filepath = self.config_dir / filename
        if not filepath.exists():
            return {}
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_domain_config(self) -> dict:
        return self.domain_config

    def get_retrieval_config(self) -> dict:
        return self.retrieval_config

    def get_security_config(self) -> dict:
        return self.security_config
