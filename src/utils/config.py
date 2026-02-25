"""
Configuration module that manages application settings.
"""

import os
from typing import Optional


class Config:
    """
    Configuration class that holds all application settings.
    Values are loaded from environment variables or defaults.
    """
    
    def __init__(self):
        """Initialize configuration values from environment variables."""
        
        # Audio processing settings
        self.AUDIO_FILE = os.getenv("AUDIO_FILE", "meeting_recording.mp3")
        self.HF_TOKEN = os.getenv("HF_TOKEN")  # Required for pyannote.audio
        self.WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
        self.AUDIO_LANGUAGE = os.getenv("AUDIO_LANGUAGE", "ru")  # Language code (e.g., 'en', 'ru')
        
        # LLM settings
        self.USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"
        self.LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Required if using Groq instead of Ollama
        self.GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
        
        # Validation
        self._validate_config()
    
    def _validate_config(self):
        """Validate that required configuration values are present."""
        if not self.HF_TOKEN:
            raise ValueError(
                "HF_TOKEN environment variable is required for pyannote.audio. "
                "Get your token from https://huggingface.co/settings/tokens"
            )
        
        if not self.USE_OLLAMA and not self.GROQ_API_KEY:
            raise ValueError(
                "Either USE_OLLAMA=true or GROQ_API_KEY must be set to configure LLM provider"
            )