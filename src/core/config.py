from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    jina_api_key: str
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    
    # Đọc cấu hình từ file .env
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Tạo một instance duy nhất để import ở mọi nơi
settings = Settings()