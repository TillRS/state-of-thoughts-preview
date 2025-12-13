"""Constants specific to local language models."""
import enum
from typing import Literal


class SamplingParam(enum.StrEnum):
	"""Enum for sampling parameters."""
	STOP = "stop"
	TEMPERATURE = "temperature"
	MAX_TOKENS = "max_tokens"
	TOP_P = "top_p"
	MIN_P = "min_p"
	TOP_K = "top_k"
	N = "n"
	USE_BEAM_SEARCH = "use_beam_search"


class MessageRole(enum.StrEnum):
	"""Enum for message roles in chat completions."""
	SYSTEM = "system"
	USER = "user"
	ASSISTANT = "assistant"
	TOOL = "tool"


class MessageKey(enum.StrEnum):
	"""Enum for message dictionary keys."""
	ROLE = "role"
	CONTENT = "content"


class FinishReason(enum.StrEnum):
	"""Enum for finish reason values in completion responses."""
	STOP = "stop"
	LENGTH = "length"
	CONTENT_FILTER = "content_filter"
	TOOL_CALLS = "tool_calls"


class ChoiceKey(enum.StrEnum):
	"""Enum for choice dictionary keys."""
	FINISH_REASON = "finish_reason"
	INDEX = "index"
	MESSAGE = "message"
	TEXT = "text"
	SCORE = "score"


class UsageKey(enum.StrEnum):
	"""Enum for usage dictionary keys."""
	PROMPT_TOKENS = "prompt_tokens"
	COMPLETION_TOKENS = "completion_tokens"
	TOTAL_TOKENS = "total_tokens"


class TaskType(enum.StrEnum):
	"""Enum for task types within vLLM."""
	AUTO = "auto"
	GENERATE = "generate"
	SCORE = "score"
	EMBEDDING = "embedding"
	EMBED = "embed"
	CLASSIFY = "classify"
	REWARD = "reward"


# Type aliases for backward compatibility
TASK_TYPES = Literal[
	TaskType.AUTO,
	TaskType.GENERATE,
	TaskType.SCORE,
	TaskType.EMBEDDING,
	TaskType.EMBED,
	TaskType.CLASSIFY,
	TaskType.REWARD,
]

CHAT_TEMPLATE_FORMATS = Literal["auto", "string", "openai"]

# Parameters for initializing vLLM models
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_SEED: int = 42
DEFAULT_MAX_MODEL_LEN: int = 8192
DEFAULT_GPU_MEMORY_UTILIZATION: float = 0.9

# Result dictionary key
RESULT_CHOICES: str = "choices"

# Parameters for chat template kwargs
ENABLE_THINKING: str = "enable_thinking"

