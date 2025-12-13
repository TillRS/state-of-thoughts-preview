"""Constants specific to controllers."""
# Standard library imports
import enum
from dataclasses import dataclass
from typing import Any


class ControllerActionParameters(enum.StrEnum):
	"""Enum for controller action parameters."""
	ACTION = "action"
	ARGUMENTS = "action_arguments"
	CONSIDERATIONS = "considerations"


class ActionSpaceJsonKeys(enum.StrEnum):
	"""Keys used in action space JSON configuration files."""
	DIMENSION_NAME = "name"
	DIMENSION_DEFINITION = "definition"
	DIMENSION_CHOICES = "choices"
	CHOICE_INTERNAL_REASONING = "internal_reasoning"
	CHOICE_PREFIX = "prefix"


class ControllerContinueReasoningChoice(enum.StrEnum):
	"""Enum for controller continue reasoning choices."""
	CONTINUE_REASONING = "continue_reasoning"
	FINISH = "finish"


class ControllerOutputParameters(enum.StrEnum):
	"""Enum for controller output parameters."""
	TRAJECTORY = "controller_output_trajectory"
	OUTPUTS = "controller_outputs"
	INTERNAL_REASONING = "internal_reasoning"
	UNIQUE_ACTION_RESPONSE_COUNT = "unique_action_response_count"
	TOOL_DESCRIPTIONS = "tool_descriptions"
	PREFIX = "prefix"
	DEFINITION = "definition"
	SCORE = "score"


class ControllerConfig(enum.StrEnum):
	"""Enum for controller configuration fields."""
	NUMBER_OF_ADDITIONAL_REASONING_STEPS = "number_of_additional_reasoning_steps"
	REASONING_FIELD_NAME = "existing_reasoning"
	MAX_ITERS = "max_iters"


@dataclass
class ReasoningIntervention:
	"""
	Represents an intervention over the next reasoning step.

	A ReasoningIntervention is the result of executing a controller tool. It specifies
	how to influence the next reasoning generation step through internal guidance
	and/or a literal text prefix.

	Attributes:
		continue_reasoning: Whether to continue reasoning (True) or generate final output (False).
		internal_reasoning: A first-person analysis of what to do next that is injected into
			the next generation to guide the model's thinking. Empty string if not provided.
		prefix: A literal text to inject at the start of the next generation to guide the
			model's thinking. Empty string if not provided.
	"""

	continue_reasoning: bool
	internal_reasoning: str = ""
	prefix: str = ""

	def to_dict(self) -> dict[str, bool | str]:
		"""
		Convert the ReasoningIntervention to a dictionary.

		Returns:
			dict[str, bool | str]: A dictionary representation of the intervention.
		"""
		return {
			ControllerContinueReasoningChoice.CONTINUE_REASONING: self.continue_reasoning,
			ControllerOutputParameters.INTERNAL_REASONING: self.internal_reasoning,
			ControllerOutputParameters.PREFIX: self.prefix,
		}


class ControllerType(enum.StrEnum):
	"""
	Enumeration representing the different types of controllers available.

	GENERATOR: Uses a generator model to generate controller actions.
	RERANKER: Uses a reranker model to score and select controller actions.
	"""

	GENERATOR = "generator"
	RERANKER = "reranker"


@dataclass
class ControllerOutput:
	"""
	Unified controller output containing both the decision and its execution result.

	The decision fields (action, action_arguments, considerations) represent the controller's choice.
	The execution fields (continue_reasoning, prefix, etc.) are populated by executing the tool.

	Attributes:
	    action: The name of the selected tool/action.
	    action_arguments: Dictionary of arguments to pass to the tool.
	    tool_descriptions: String describing the tool and its parameters.
	    considerations: The reasoning behind selecting this action.
	    continue_reasoning: Whether to continue reasoning (True) or generate final output (False).
	    internal_reasoning: Optional internal reasoning to guide the next generation step.
	    prefix: Optional prefix text to inject at the start of the next generation.
	    tool_execution_error: Optional error message if tool execution failed.
	    failed_tool: Optional name of the tool that failed to execute.
	    unique_action_response_count: Number of times this unique action+arguments combination was chosen.
	"""

	# Decision fields (from ControllerChoice)
	action: str
	action_arguments: dict[str, Any]  # Dictionary of parameters passed into the tool call.
	tool_descriptions: str  # Formatted string describing the tool and its parameters.
	continue_reasoning: bool  # Continue reasoning if True, generate final output if False.
	considerations: str = "N/A"  # Default is "N/A" (occurs if using a reranker-based controller).

	# Execution result fields (from ControllerIntervention - auto-populated by tool execution)
	internal_reasoning: str = ""  # Represents internal reasoning of the model before next generation.
	prefix: str = ""  # A text prefix that the model must continue from in the next generation.
	tool_execution_error: str = ""  # An error message produced by the tool execution.
	failed_tool: str = ""  # The name of the tool that failed to execute.

	# Shared field
	unique_action_response_count: int = 1  # Number of times this unique action+arguments combination was chosen.
	score: float | None = None  # The score assigned to this action (if using a reranker).

	@classmethod
	def from_choice_dict(
		cls,
		choice_dict: dict[str, Any],
		intervention: ReasoningIntervention,
		tool_execution_error: str = "",
	) -> "ControllerOutput":
		"""
		Create ControllerOutput from a choice dict and intervention.

		Parameters:
		    choice_dict: Dict with ControllerActionParameters.ACTION, ControllerActionParameters.ARGUMENTS,
		        ControllerActionParameters.CONSIDERATIONS, and optionally
		        ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT and
		        ControllerOutputParameters.TOOL_DESCRIPTIONS
		    intervention: ReasoningIntervention from tool execution
		    tool_execution_error: Error message if tool execution failed (default: "")

		Returns:
		    ControllerOutput with all fields populated
		"""
		return cls(
			action=choice_dict[ControllerActionParameters.ACTION],
			action_arguments=choice_dict[ControllerActionParameters.ARGUMENTS],
			tool_descriptions=choice_dict[ControllerOutputParameters.TOOL_DESCRIPTIONS],
			considerations=choice_dict[ControllerActionParameters.CONSIDERATIONS],
			continue_reasoning=intervention.continue_reasoning,
			internal_reasoning=intervention.internal_reasoning,
			prefix=intervention.prefix,
			unique_action_response_count=choice_dict.get(
				ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT, 1
			),
			tool_execution_error=tool_execution_error,
			failed_tool=choice_dict[ControllerActionParameters.ACTION] if tool_execution_error else "",
		)


# Tool name constants
DEFAULT_REASONING_INTERVENTION_TOOL_NAME = "intervene_on_next_reasoning_step"
