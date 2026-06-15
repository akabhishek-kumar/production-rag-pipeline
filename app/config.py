from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    groq_api_key: str = "gsk_..."
    groq_model: str = "llama-3.1-8b-instant"

    # Storage — persistent Chroma on disk
    persist_dir: str = "./chroma_db"
    docs_dir: str = "./docs"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    retriever_k: int = 5

    # Quality gates
    grade_threshold: int = 6
    max_retries: int = 2
    recursion_limit: int = 25


settings = Settings()
