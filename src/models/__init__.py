"""
Models package for the audio recognition and summarization project.
"""

from .transcription import TranscriptSegment, TranscriptResult, DiarizationResult

__all__ = [
    'TranscriptSegment',
    'TranscriptResult',
    'DiarizationResult'
]