"""Constants specific to signature fields and metadata."""
import enum


class EvaluationMetric(enum.StrEnum):
	"""Enum for evaluation metrics."""
	PERSUASIVENESS = "persuasiveness"
	COHERENCE = "coherence"
	RELEVANCE = "relevance"
	CORRECTNESS = "correctness"
	CLARITY = "clarity"
	EFFICIENCY = "efficiency"
	COMPLETENESS = "completeness"
	SOUNDNESS = "soundness"
	PROMISE = "promise"


class FieldMetadata(enum.StrEnum):
	"""Enum for field metadata."""
	DESCRIPTION = "description"
	JSON_SCHEMA_EXTRA = "json_schema_extra"
	CONSTRAINTS = "constraints"
	DESC = "desc"
	FORMAT = "format"
	PREFIX = "prefix"
	DSPY_TYPE = "__dspy_field_type"


class FieldType(enum.StrEnum):
	"""Enum for field types."""
	INPUT = "input"
	REASONING = "reasoning"
	OUTPUT = "output"


class ConstraintField(enum.StrEnum):
	"""Enum for constraint fields."""
	GRANULARITY = "granularity"
	BOUNDS = "bounds"


# Default reasoning field name
DEFAULT_REASONING_FIELD_NAME = "reasoning_step"
