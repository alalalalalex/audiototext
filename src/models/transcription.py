"""
Data models for transcription results.
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class TranscriptSegment:
    """
    Represents a single segment of the transcript with speaker information.
    """
    speaker: str
    text: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None


@dataclass
class TranscriptResult:
    """
    Complete transcription result with metadata.
    """
    segments: List[TranscriptSegment]
    total_duration: float
    language: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class DiarizationResult:
    """
    Result of speaker diarization.
    """
    speaker_labels: List[str]
    speaker_times: List[tuple]  # List of (start_time, end_time) tuples
    num_speakers: int