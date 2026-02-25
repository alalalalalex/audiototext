"""
Main entry point for the audio recognition and summarization project.
This module orchestrates the entire pipeline:
1. Audio processing with speaker diarization
2. Speech-to-text transcription
3. Text formatting
4. Conversation summarization using LLM
"""

import os
import logging
from typing import List, Dict, Optional

from src.services.audio_processor import AudioProcessor
from src.services.text_formatter import TextFormatter
from src.services.summarizer import Summarizer
from src.utils.config import Config


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """
    Main execution function that runs the complete pipeline.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    config = Config()
    
    # Initialize services
    audio_processor = AudioProcessor(config)
    text_formatter = TextFormatter()
    summarizer = Summarizer(config)
    
    # Validate input file exists
    if not os.path.exists(config.AUDIO_FILE):
        logger.error(f"Audio file not found: {config.AUDIO_FILE}")
        raise FileNotFoundError(f"Audio file not found: {config.AUDIO_FILE}")
    
    logger.info(f"Starting processing of audio file: {config.AUDIO_FILE}")
    
    # Step 1: Process audio and perform diarization + transcription
    logger.info("Step 1: Processing audio and performing diarization...")
    transcript_data = audio_processor.process_audio(config.AUDIO_FILE)
    
    # Step 2: Format the transcript for better readability
    logger.info("Step 2: Formatting transcript...")
    formatted_transcript = text_formatter.format_transcript(transcript_data)
    
    # Save full transcript to file
    transcript_file = "transcript.txt"
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(formatted_transcript)
    logger.info(f"Full transcript saved to {transcript_file}")
    
    # Step 3: Generate summary using LLM
    logger.info("Step 3: Generating summary with LLM...")
    summary = summarizer.generate_summary(formatted_transcript)
    
    # Print results
    print("\n" + "="*50)
    print("CONVERSATION SUMMARY")
    print("="*50)
    print(summary)
    print("="*50)


if __name__ == "__main__":
    main()