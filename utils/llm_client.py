"""Unified LLM Client supporting OpenAI, Anthropic, and SiliconFlow backends."""

import os
import sys
import json
from typing import Dict, Any, Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY,
    SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL,
    LLM_BACKEND, MODEL_NAME, CLAUDE_MODEL
)


class UnifiedLLMClient:
    """
    Unified interface for OpenAI, Anthropic, and SiliconFlow LLMs.
    Allows seamless switching between backends for experiments.
    """
    
    def __init__(
        self, 
        backend: str = None, 
        model_name: str = None,
        temperature: float = 0.0
    ):
        """
        Initialize the LLM client.
        
        Args:
            backend: "openai", "anthropic", or "siliconflow" (defaults to config.LLM_BACKEND)
            model_name: Model name (defaults to config.MODEL_NAME or CLAUDE_MODEL)
            temperature: Generation temperature (default 0.0 for deterministic)
        """
        self.backend = backend or LLM_BACKEND
        self.temperature = temperature
        
        if self.backend == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model_name = model_name or MODEL_NAME
        elif self.backend == "siliconflow":
            from openai import OpenAI
            # SiliconFlow uses OpenAI-compatible API
            self.client = OpenAI(
                api_key=SILICONFLOW_API_KEY,
                base_url=SILICONFLOW_BASE_URL
            )
            self.model_name = model_name or MODEL_NAME
        elif self.backend == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")
            
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not set in config.py")
            
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.model_name = model_name or CLAUDE_MODEL
        else:
            raise ValueError(f"Unknown backend: {self.backend}. Use 'openai', 'anthropic', or 'siliconflow'")
    
    def chat(
        self, 
        system_prompt: str, 
        user_prompt: str,
        max_tokens: int = 2000,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Send a chat request to the LLM.
        
        Args:
            system_prompt: System message
            user_prompt: User message
            max_tokens: Maximum tokens in response
            response_format: Optional response format (OpenAI only, e.g., {"type": "json_object"})
        
        Returns:
            Response text from the LLM
        """
        if self.backend in ["openai", "siliconflow"]:
            return self._openai_chat(system_prompt, user_prompt, max_tokens, response_format)
        else:
            return self._anthropic_chat(system_prompt, user_prompt, max_tokens)
    
    def _openai_chat(
        self, 
        system_prompt: str, 
        user_prompt: str,
        max_tokens: int,
        response_format: Optional[Dict]
    ) -> str:
        """OpenAI/SiliconFlow chat completion."""
        # Check if model is GPT-5 or reasoning model (o1, o3)
        is_new_model = any(x in self.model_name.lower() for x in ['gpt-5', 'o1', 'o3'])
        is_siliconflow = self.backend == "siliconflow"
        
        kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        }
        
        # GPT-5 and reasoning models have different parameter requirements
        if is_new_model and not is_siliconflow:
            kwargs["max_completion_tokens"] = max_tokens
            # GPT-5 only supports temperature=1 (don't set it explicitly)
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = self.temperature
        
        if response_format and not is_new_model and not is_siliconflow:
            # Response format may not be supported on SiliconFlow or new models
            kwargs["response_format"] = response_format
        
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    
    def _anthropic_chat(
        self, 
        system_prompt: str, 
        user_prompt: str,
        max_tokens: int
    ) -> str:
        """Anthropic chat completion."""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature
        )
        return response.content[0].text
    
    def chat_json(
        self, 
        system_prompt: str, 
        user_prompt: str,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Send a chat request and parse JSON response.
        
        For OpenAI: Uses response_format={"type": "json_object"}
        For SiliconFlow/Anthropic: Parses JSON from response text
        
        Returns:
            Parsed JSON dict
        """
        if self.backend == "openai":
            # OpenAI supports structured JSON output
            response = self._openai_chat(
                system_prompt + "\n\nRespond in valid JSON format.",
                user_prompt,
                max_tokens,
                {"type": "json_object"}
            )
        elif self.backend == "siliconflow":
            # SiliconFlow: Add JSON instruction to prompt (may not support response_format)
            response = self._openai_chat(
                system_prompt + "\n\nRespond ONLY with valid JSON, no additional text.",
                user_prompt,
                max_tokens,
                None  # Don't use response_format for SiliconFlow
            )
        else:
            # Anthropic: Add JSON instruction to prompt
            response = self._anthropic_chat(
                system_prompt + "\n\nRespond ONLY with valid JSON, no additional text.",
                user_prompt,
                max_tokens
            )
        
        # Parse JSON
        try:
            # Handle potential markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            # Return raw response if JSON parsing fails
            return {"raw_response": response, "parse_error": str(e)}
    
    @property
    def info(self) -> Dict[str, str]:
        """Get client info for logging."""
        return {
            "backend": self.backend,
            "model": self.model_name,
            "temperature": self.temperature
        }


def get_llm_client(
    backend: str = None, 
    model_name: str = None
) -> UnifiedLLMClient:
    """
    Factory function to create an LLM client.
    
    Args:
        backend: "openai", "anthropic", or "siliconflow"
        model_name: Specific model name
    
    Returns:
        UnifiedLLMClient instance
    """
    return UnifiedLLMClient(backend=backend, model_name=model_name)


# Convenience functions for quick experiments
def test_openai(prompt: str = "Say hello in 5 words.") -> str:
    """Quick test for OpenAI connection."""
    client = UnifiedLLMClient(backend="openai")
    return client.chat("You are helpful.", prompt)


def test_anthropic(prompt: str = "Say hello in 5 words.") -> str:
    """Quick test for Anthropic connection."""
    client = UnifiedLLMClient(backend="anthropic")
    return client.chat("You are helpful.", prompt)


def test_siliconflow(model: str = "deepseek-ai/DeepSeek-V3", prompt: str = "Say hello in 5 words.") -> str:
    """Quick test for SiliconFlow connection."""
    client = UnifiedLLMClient(backend="siliconflow", model_name=model)
    return client.chat("You are helpful.", prompt)


if __name__ == "__main__":
    # Test script
    print("Testing LLM clients...")
    
    print("\n--- OpenAI Test ---")
    try:
        result = test_openai()
        print(f"OpenAI ({MODEL_NAME}): {result}")
    except Exception as e:
        print(f"OpenAI error: {e}")
    
    print("\n--- Anthropic Test ---")
    try:
        result = test_anthropic()
        print(f"Anthropic ({CLAUDE_MODEL}): {result}")
    except Exception as e:
        print(f"Anthropic error: {e}")
    
    print("\n--- SiliconFlow Test ---")
    try:
        result = test_siliconflow()
        print(f"SiliconFlow (DeepSeek-V3): {result}")
    except Exception as e:
        print(f"SiliconFlow error: {e}")

