"""Constants specific to adapters."""
# Standard library imports
import enum
import re


class XMLTag(enum.StrEnum):
	"""Enum for XML tags."""
	THINKING_START = "<thinking>"
	THINKING_END = "</thinking>"
	STEP_START = "<step>"
	STEP_END = "</step>"
	ANSWER_START = "<answer>"
	ANSWER_END = "</answer>"


class ScoringTarget(enum.StrEnum):
	"""Enum for scoring targets."""
	REASONING = "reasoning"
	OUTPUT = "output"
	ACTION = "action"


class FinalOutputKind(enum.StrEnum):
	"""Kind of final output instruction to include in system prompt."""

	SYNTHESIS_STRICT = "synthesis_strict"
	# Very faithful to the reasoning steps.
	# Preserve content, structure, ordering, and phrasing as closely as possible.
	# Essentially a detailed synthesis of the existing reasoning.

	SYNTHESIS_FAITHFUL = "synthesis_faithful"
	# Faithful to the ideas and reasoning steps, but allows light rephrasing.
	# Ordering and structure should remain the same, with minimal stylistic edits.

	SYNTHESIS_RESTRUCTURED = "synthesis_restructured"
	# Maintains the same broad ideas and reasoning,
	# but allows rephrasing and restructuring for clarity and coherence.

	CONCLUSION = "conclusion"
	# Allows the model to produce the best possible final answer based on the reasoning.
	# Prioritizes clarity and quality over strict faithfulness to structure or phrasing.


class AdapterErrorKey(enum.StrEnum):
	"""Keys for adapter error reporting."""
	FAILED_PARSING = "failed_parsing"
	RAW_OUTPUT = "raw_output"
	ERROR = "error"


# Adapter configuration constant
GENERATOR_ADAPTER_NAME = "VLLMGeneratorAdapter"

# Pattern constants
FIELD_HEADER_PATTERN = re.compile(r"## (\w+)")
