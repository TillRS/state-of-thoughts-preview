"""
Centralized constants for the dspy_reasoning project.

Core constants that are used across multiple modules.
Module-specific constants are stored in their respective modules:
- adapter/adapter_constants.py: Adapter-specific constants
- predict/controller_constants.py: Controller-specific constants (ControllerType)
- predict/tree_of_thoughts/tree_parameters.py: Tree of Thoughts parameters
- tree/tree_constants.py: Tree and ToT-specific constants
- signatures/field_constants.py: Signature field constants
- experiments/argument_generation/arg_gen_constants.py: Argument generation constants
- lm/lm_constants.py: Language model constants (FinishReason, TaskType)
"""
# Standard library imports
import enum
import logging
from pathlib import Path

# ============================================================================
# Enums
# ============================================================================


class GPU(enum.StrEnum):
	"""Enum for GPU types."""

	# NVIDIA GPUs
	A100_40_GB = "A100_40GB"
	A100_80_GB = "A100_80GB"
	H100_80_GB = "H100_80GB"

	def __str__(self) -> str:
		return self.value


class OpenSourceModel(enum.StrEnum):
	"""Enum for open source language models."""

	# Qwen Models
	QWEN_3_4B = "Qwen3-4B"  # Requires a T4 or better to run
	QWEN_3_4B_INSTRUCT_2507 = (
		"Qwen3-4B-Instruct-2507"  # Requires a T4 or better to run
	)
	QWEN_3_4B_4_BIT = (
		"Qwen3-4B-unsloth-bnb-4bit"  # Requires a T4 or better to run
	)
	QWEN_3_8B_4_BIT = (
		"Qwen3-8B-unsloth-bnb-4bit"  # Requires a T4 or better to run (~7 tok/s)
	)
	QWEN_3_14B_4_BIT = (
		"Qwen3-14B-unsloth-bnb-4bit"  # Requires an A100 or better to run
	)
	QWEN_3_30B_A3B = (
		"Qwen3-30B-A3B"  # Requires an A100 with 80GB of VRAM to run
	)
	QWEN_3_30B_A3B_INSTRUCT_2507 = (
		"Qwen3-30B-A3B-Instruct-2507"  # Requires an A100 with 80GB of VRAM to run
	)
	QWEN_3_RERANKER_4B = (
		"Qwen3-Reranker-4B"  # Requires a T4 or better to run
	)
	QWEN_3_RERANKER_8B = (
		"Qwen3-Reranker-8B"  # Requires an A100 or better to run
	)

	def __str__(self) -> str:
		return self.value


class ModelProvider(enum.StrEnum):
	"""
	Enum for open source language model providers.
	We support the providers of the open source models listed in OpenSourceModel.
	"""

	UNSLOTH = "unsloth"
	META_LLAMA = "meta-llama"
	QWEN = "Qwen"

	def __str__(self) -> str:
		return self.value


class CandidateGenerationMethod(enum.StrEnum):
	"""
	Methods for generating candidate reasoning steps.

	SINGLE_CANDIDATE_CALLS: Instructs the model to generate a single output per call and derives
		multiple candidates by repeatedly calling the model. This method depends on
		temperature-induced diversity, meaning it relies on relatively high generator temperature
		to produce different outputs over multiple calls.
	MULTI_CANDIDATE_CALL: Directly instructs the model to generate multiple outputs in a single
		call. This method leverages the model's ability to follow instructions for producing
		diverse reasoning steps without depending on temperature-based sampling.
	"""

	SINGLE_CANDIDATE_CALLS = "single_candidate_calls"
	MULTI_CANDIDATE_CALL = "multi_candidate_call"


class Verbosity(enum.StrEnum):
	"""Enum for verbosity levels."""
	DEBUG = "debug"
	INFO = "info"
	WARNING = "warning"
	ERROR = "error"


# Mapping from Verbosity enum to logging levels
VERBOSITY_TO_LOGGING_LEVEL: dict[Verbosity, int] = {
	Verbosity.DEBUG: logging.DEBUG,
	Verbosity.INFO: logging.INFO,
	Verbosity.WARNING: logging.WARNING,
	Verbosity.ERROR: logging.ERROR,
}

# ============================================================================
# Path Constants
# ============================================================================

CURRENT_DIR = Path(__file__).resolve().parent
