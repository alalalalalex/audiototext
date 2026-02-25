"""
Summarization service that uses LLM to generate conversation summaries.
"""

import logging
from typing import Optional

import ollama
from groq import Groq

from src.utils.config import Config


class Summarizer:
    """
    Service class for generating summaries using various LLM providers.
    Supports both local Ollama models and remote APIs like Groq.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the summarizer with configuration.
        
        Args:
            config: Configuration object containing settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients based on configuration
        self.ollama_client = None
        self.groq_client = None
        
        if config.USE_OLLAMA:
            self.ollama_client = ollama
        elif config.GROQ_API_KEY:
            self.groq_client = Groq(api_key=config.GROQ_API_KEY)
    
    def generate_summary(self, text: str) -> str:
        """
        Generate a summary of the provided text using configured LLM.
        
        Args:
            text: Text to summarize
            
        Returns:
            Generated summary
        """
        self.logger.info("Generating summary with LLM...")
        
        # Create the prompt for the LLM
        prompt = self._create_summary_prompt(text)
        
        # Choose the appropriate LLM provider based on configuration
        if self.groq_client:
            return self._generate_with_groq(prompt)
        elif self.ollama_client:
            return self._generate_with_ollama(prompt)
        else:
            raise ValueError("No valid LLM provider configured (either Ollama or Groq API key needed)")
    
    def _create_summary_prompt(self, text: str) -> str:
        """
        Create a structured prompt for conversation summarization.
        
        Args:
            text: Original text to summarize
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
        You are a professional secretary. Below is a transcript of a conversation between multiple participants.
        Your task:
        1. Make a brief summary of the conversation (3-5 sentences).
        2. Highlight key agreements or tasks (as bullet points).
        3. Identify the overall tone of the conversation.
        
        Conversation transcript:
        {text}
        """
        return prompt.strip()
    
    def _generate_with_ollama(self, prompt: str) -> str:
        """
        Generate summary using local Ollama model.
        
        Args:
            prompt: Prompt to send to the model
            
        Returns:
            Generated summary
        """
        if not self.ollama_client:
            raise ValueError("Ollama client not initialized")
        
        response = self.ollama_client.chat(
            model=self.config.LLAMA_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ]
        )
        
        return response['message']['content']
    
    def _generate_with_groq(self, prompt: str) -> str:
        """
        Generate summary using Groq API.
        
        Args:
            prompt: Prompt to send to the model
            
        Returns:
            Generated summary
        """
        if not self.groq_client:
            raise ValueError("Groq client not initialized")
        
        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=self.config.GROQ_MODEL,
        )
        
        return chat_completion.choices[0].message.content