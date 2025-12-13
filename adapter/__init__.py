"""
Adapter module for VLLM integration with DSPy.
"""

# Local imports - constants only (no circular dependencies)
from adapter.adapter_constants import ScoringTarget, XMLTag
from adapter.constraints import GranularityType, ResponseLength
from lm.lm_constants import MessageKey, MessageRole, SamplingParam

__all__ = [
	# Constants/Enums (always available, no circular imports)
	"GranularityType",
	"MessageKey",
	"MessageRole",
	"ResponseLength",
	"SamplingParam",
	"ScoringTarget",
	"XMLTag",
]

# Note: Classes like LocalVLLMAdapter, VLLMGeneratorAdapter, etc.
# should be imported directly from their modules to avoid circular imports:
#   from adapter.vllm_adapter import LocalVLLMAdapter
#   from adapter.vllm_generator_adapter import VLLMGeneratorAdapter
