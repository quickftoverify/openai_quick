#!/usr/bin/env python3
"""
OpenAI API Utility Module

A utility module providing a convenient wrapper class for OpenAI API interactions.
Supports:
- Chat completions
- Text completions
- Image generation
- Text embeddings

Requirements:
- pip install openai python-dotenv
- Set OPENAI_API_KEY in .env file or environment variable

Usage:
    from openai_app import OpenAIClient
    
    client = OpenAIClient()
    response = client.chat_completion([
        {"role": "user", "content": "Hello!"}
    ])
"""

import os
import json
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import openai
from openai import OpenAI

# Load environment variables
load_dotenv()

class OpenAIClient:
    """A wrapper class for OpenAI API interactions"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI client
        
        Args:
            api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "gpt-3.5-turbo",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Create a chat completion
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (e.g., 'gpt-3.5-turbo', 'gpt-4')
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-2)
            stream: Whether to stream the response
            
        Returns:
            Response from OpenAI API
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )
            return response
        except Exception as e:
            print(f"Error in chat completion: {e}")
            raise
    
    def text_completion(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo-instruct",
        max_tokens: int = 150,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Create a text completion (legacy completions endpoint)
        
        Args:
            prompt: The prompt to complete
            model: Model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Response from OpenAI API
        """
        try:
            response = self.client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response
        except Exception as e:
            print(f"Error in text completion: {e}")
            raise
    
    def generate_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        quality: str = "standard"
    ) -> Dict[str, Any]:
        """
        Generate an image using DALL-E
        
        Args:
            prompt: Description of the image to generate
            n: Number of images to generate
            size: Size of the generated images
            quality: Quality of the generated images
            
        Returns:
            Response from OpenAI API
        """
        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=n,
                size=size,
                quality=quality
            )
            return response
        except Exception as e:
            print(f"Error in image generation: {e}")
            raise
    
    def get_embeddings(
        self,
        text: str,
        model: str = "text-embedding-ada-002"
    ) -> Dict[str, Any]:
        """
        Get embeddings for text
        
        Args:
            text: Text to get embeddings for
            model: Embedding model to use
            
        Returns:
            Response from OpenAI API
        """
        try:
            response = self.client.embeddings.create(
                model=model,
                input=text
            )
            return response
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            raise
