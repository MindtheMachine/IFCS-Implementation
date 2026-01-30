"""
LLM Provider Abstraction Layer
Supports multiple LLM backends with unified interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import os
from pathlib import Path


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = None,
        system: Optional[str] = None,
        seed: Optional[int] = None
    ) -> str:
        """Generate response from LLM

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (None = provider default)
            top_p: Nucleus sampling (None = provider default)
            system: System prompt (optional)

        Returns:
            Generated text response
        """
        raise NotImplementedError

    def generate_batch(
        self,
        prompt: str,
        n: int,
        max_tokens: int = 2000,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = None,
        system: Optional[str] = None,
        seed: Optional[int] = None
    ) -> List[str]:
        """Generate multiple candidates (default: n sequential calls)."""
        return [
            self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                system=system,
                seed=seed
            )
            for _ in range(n)
        ]

    def capabilities(self) -> Dict[str, bool]:
        """Advertise provider feature support."""
        return {
            "temperature": True,
            "top_p": True,
            "seed": False,
            "system": True,
            "batch": False,
        }

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name for output organization"""
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = None,
        system: Optional[str] = None,
        seed: Optional[int] = None
    ) -> str:
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p

        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def get_model_name(self) -> str:
        return self.model

    def capabilities(self) -> Dict[str, bool]:
        return {
            "temperature": True,
            "top_p": True,
            "seed": False,
            "system": True,
            "batch": False,
        }


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-4, etc.)"""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI provider requires 'openai' package. "
                "Install with: pip install openai"
            )

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = None,
        system: Optional[str] = None,
        seed: Optional[int] = None
    ) -> str:
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if seed is not None:
            kwargs["seed"] = seed

        response = self.client.chat.completions.create(
            **kwargs
        )

        return response.choices[0].message.content

    def get_model_name(self) -> str:
        return self.model.replace("/", "-")  # Sanitize for folder names

    def capabilities(self) -> Dict[str, bool]:
        return {
            "temperature": True,
            "top_p": True,
            "seed": True,
            "system": True,
            "batch": False,
        }


class HuggingFaceProvider(LLMProvider):
    """HuggingFace Inference API provider"""

    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3.1-70B-Instruct"):
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "HuggingFace provider requires 'huggingface_hub' package. "
                "Install with: pip install huggingface_hub"
            )

        self.client = InferenceClient(token=api_key)
        self.model = model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = None,
        system: Optional[str] = None,
        seed: Optional[int] = None
    ) -> str:
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat_completion(
            messages=messages,
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else None,
            top_p=top_p
        )

        return response.choices[0].message.content

    def get_model_name(self) -> str:
        # Convert "meta-llama/Llama-3.1-70B-Instruct" to "Llama-3.1-70B-Instruct"
        return self.model.split("/")[-1]

    def capabilities(self) -> Dict[str, bool]:
        return {
            "temperature": True,
            "top_p": True,
            "seed": False,
            "system": True,
            "batch": False,
        }


class OllamaProvider(LLMProvider):
    """Ollama local model provider"""

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434"):
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "Ollama provider requires 'ollama' package. "
                "Install with: pip install ollama"
            )

        self.client = ollama.Client(host=base_url)
        self.model = model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = None,
        system: Optional[str] = None,
        seed: Optional[int] = None
    ) -> str:
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        options = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p

        response = self.client.chat(
            model=self.model,
            messages=messages,
            options=options
        )

        return response['message']['content']

    def get_model_name(self) -> str:
        return self.model

    def capabilities(self) -> Dict[str, bool]:
        return {
            "temperature": True,
            "top_p": True,
            "seed": False,
            "system": True,
            "batch": False,
        }


class LLMProviderFactory:
    """Factory for creating LLM providers from configuration"""

    @staticmethod
    def create_from_env() -> LLMProvider:
        """Create LLM provider from environment variables

        Environment Variables:
            LLM_PROVIDER: anthropic|openai|huggingface|ollama (default: anthropic)
            LLM_MODEL: Model name (provider-specific)
            LLM_API_KEY: API key (not needed for ollama)
            OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)

            Legacy (backward compatibility):
            ANTHROPIC_API_KEY: Anthropic API key

        Returns:
            Configured LLM provider
        """
        LLMProviderFactory._reload_env()
        provider_type = os.getenv("LLM_PROVIDER", "anthropic").lower()

        # Provider-specific model defaults
        default_models = {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4-turbo",
            "huggingface": "meta-llama/Llama-3.1-70B-Instruct",
            "ollama": "llama3.1"
        }

        model = os.getenv("LLM_MODEL", default_models.get(provider_type))

        if provider_type == "anthropic":
            api_key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "Anthropic provider requires LLM_API_KEY or ANTHROPIC_API_KEY environment variable"
                )
            return AnthropicProvider(api_key=api_key, model=model)

        elif provider_type == "openai":
            api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI provider requires LLM_API_KEY or OPENAI_API_KEY environment variable"
                )
            return OpenAIProvider(api_key=api_key, model=model)

        elif provider_type == "huggingface":
            api_key = os.getenv("LLM_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
            if not api_key:
                raise ValueError(
                    "HuggingFace provider requires LLM_API_KEY or HUGGINGFACE_API_KEY environment variable"
                )
            return HuggingFaceProvider(api_key=api_key, model=model)

        elif provider_type == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return OllamaProvider(model=model, base_url=base_url)

        else:
            raise ValueError(
                f"Unknown LLM provider: {provider_type}. "
                f"Supported: anthropic, openai, huggingface, ollama"
            )

    @staticmethod
    def _reload_env():
        """Reload .env each run and purge stale provider vars."""
        env_path = Path(".env")
        if not env_path.exists():
            return

        keys_to_clear = [
            "LLM_PROVIDER",
            "LLM_MODEL",
            "LLM_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "HUGGINGFACE_API_KEY",
            "OLLAMA_HOST",
            "OLLAMA_BASE_URL",
        ]
        for key in keys_to_clear:
            os.environ.pop(key, None)

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value

    @staticmethod
    def create_from_config(config: 'TrilogyConfig') -> LLMProvider:
        """Create LLM provider from TrilogyConfig

        Falls back to environment variables if not specified in config
        """
        # If config has new llm_provider field, use it
        if hasattr(config, 'llm_provider') and config.llm_provider:
            return config.llm_provider

        # Otherwise, create from environment
        return LLMProviderFactory.create_from_env()
