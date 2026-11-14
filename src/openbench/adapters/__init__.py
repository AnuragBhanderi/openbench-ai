"""Model providers and adapters package for OpenBench AI."""

from openbench.adapters.base import (
    PROVIDER_REGISTRY,
    BaseModelAdapter,
    GenerationConfig,
    ProviderHealthStatus,
    create_provider,
    list_registered_providers,
    register_provider,
)
from openbench.adapters.huggingface import HuggingFaceAdapter
from openbench.adapters.local import DeterministicLookupAdapter, LocalCallableAdapter
from openbench.adapters.mock import MockModelProvider
from openbench.adapters.ollama import DEFAULT_OLLAMA_ENDPOINT, OllamaAdapter, OllamaConnectionError
from openbench.adapters.openai_compatible import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_OPENAI_COMPATIBLE_ENDPOINT,
    OpenAICompatibleAdapter,
    OpenAICompatibleConnectionError,
)

# Register all built-in providers
register_provider(MockModelProvider)
register_provider(OllamaAdapter)
register_provider(OpenAICompatibleAdapter, aliases=["openai", "openai-compatible"])
register_provider(HuggingFaceAdapter, aliases=["huggingface", "hf"])
register_provider(LocalCallableAdapter)
register_provider(DeterministicLookupAdapter)

__all__ = [
    "BaseModelAdapter",
    "DEFAULT_API_KEY_ENV",
    "DEFAULT_OLLAMA_ENDPOINT",
    "DEFAULT_OPENAI_COMPATIBLE_ENDPOINT",
    "DeterministicLookupAdapter",
    "GenerationConfig",
    "HuggingFaceAdapter",
    "LocalCallableAdapter",
    "MockModelProvider",
    "OllamaAdapter",
    "OllamaConnectionError",
    "OpenAICompatibleAdapter",
    "OpenAICompatibleConnectionError",
    "PROVIDER_REGISTRY",
    "ProviderHealthStatus",
    "create_provider",
    "list_registered_providers",
    "register_provider",
]
