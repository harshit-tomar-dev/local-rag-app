"""
Configuration module loading from .env and environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root if present
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    PROJECT_NAME: str = "Local RAG Assistant"
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", str(BASE_DIR / "data" / "rag.db"))
    UPLOADS_DIR: str = os.getenv("UPLOADS_DIR", str(BASE_DIR / "data" / "uploads"))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")

    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "extractive").lower()
    LLM_MODEL: str = os.getenv("LLM_MODEL", "extractive")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")

    # Multi-Provider Dedicated API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

    # Embedding Settings (Step 8)
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "none").lower()
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434")

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


settings = Settings()
