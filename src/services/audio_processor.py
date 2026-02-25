"""
Audio processing service that handles speaker diarization and speech-to-text conversion.
"""

import os
from typing import List, Dict, Any
from pathlib import Path

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Segment
import torch

from src.utils.config import Config


class AudioProcessor:
    """
    Service class for processing audio files with speaker diarization and transcription.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the audio processor with configuration.
        
        Args:
            config: Configuration object containing settings
        """
        self.config = config
        self.diarization_pipeline = None
        self.whisper_model = None
        self._setup_models()
    
    def _setup_models(self):
        """Initialize diarization pipeline and Whisper model."""
        # Initialize diarization pipeline
        if self.config.HF_TOKEN:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.config.HF_TOKEN
            )
        else:
            raise ValueError("Hugging Face token is required for diarization")
        
        # Determine device based on availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "float32"
        
        # Initialize Whisper model
        self.whisper_model = WhisperModel(
            self.config.WHISPER_MODEL_SIZE, 
            device=device, 
            compute_type=compute_type
        )
    
    def process_audio(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Process audio file with both diarization and transcription.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of dictionaries containing speaker, text, start and end times
        """
        print(f"Loading audio: {audio_path}...")
        
        # Run diarization
        print("Running diarization...")
        diarization = self.diarization_pipeline(audio_path)
        
        # Run transcription with word-level timestamps
        print("Running transcription...")
        segments, info = self.whisper_model.transcribe(
            audio_path, 
            word_timestamps=True,
            language=self.config.AUDIO_LANGUAGE
        )
        segments = list(segments)
        
        # Match speakers to transcribed segments
        transcript_with_speakers = []
        
        for segment in segments:
            start, end = segment.start, segment.end
            text = segment.text
            
            # Find which speaker was talking during this time
            speakers = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Check for overlap between diarization turn and transcription segment
                if turn.end > start and turn.start < end:
                    speakers.append(speaker)
            
            # Take the primary speaker for this segment (simplified approach)
            speaker_label = speakers[0] if speakers else "UNKNOWN"
            
            transcript_with_speakers.append({
                "speaker": speaker_label,
                "text": text.strip(),
                "start": start,
                "end": end
            })
        
        return transcript_with_speakers