"""
Text formatting service that prepares transcribed text for LLM processing.
"""

import nltk
from typing import List, Dict, Any
from nltk.tokenize import sent_tokenize


class TextFormatter:
    """
    Service class for formatting transcribed text to make it suitable for LLM processing.
    """
    
    def __init__(self):
        """
        Initialize the text formatter and ensure required NLTK data is available.
        """
        self._ensure_nltk_resources()
    
    def _ensure_nltk_resources(self):
        """Ensure required NLTK resources for tokenization are downloaded."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')
    
    def format_transcript(self, transcript_data: List[Dict[str, Any]]) -> str:
        """
        Format the transcript data into a readable text format with speaker labels.
        
        Args:
            transcript_data: List of dictionaries containing speaker, text, start, end
            
        Returns:
            Formatted transcript as a string with speaker labels
        """
        formatted_text = ""
        
        for item in transcript_data:
            # Split text into sentences for better readability
            sentences = sent_tokenize(item['text'], language='russian')
            for sentence in sentences:
                formatted_text += f"{item['speaker']}: {sentence}\n"
        
        return formatted_text
    
    def split_text_by_context_window(self, text: str, max_tokens: int = 7000) -> List[str]:
        """
        Split text into chunks that fit within the LLM's context window.
        
        Args:
            text: Input text to be split
            max_tokens: Maximum number of tokens per chunk
            
        Returns:
            List of text chunks
        """
        # This is a simplified implementation
        # In a real application, we'd use a proper tokenizer to count tokens
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            # Approximate token count (typically 1 word ~ 1.3 tokens)
            word_length = len(word)
            
            if current_length + word_length > max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = word_length
                else:
                    # Handle very long single words/phrases
                    chunks.append(word)
                    current_chunk = []
                    current_length = 0
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks