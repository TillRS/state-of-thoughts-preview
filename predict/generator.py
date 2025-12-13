"""
Generator: State-based generator with automatic batching.

This module implements the TreeOfThoughtGenerator class, aligned with the architecture of
controller.py and evaluator.py. It accepts State objects as input and automatically handles
batching, including heterogeneous batches (mixed reasoning and answer generation states).
"""

# Standard library imports
import logging
import random
from collections import namedtuple
from typing import Any

# Third-party imports
from dspy.dsp.utils.settings import settings
from dspy.predict import Predict
from dspy.primitives.prediction import Prediction

from adapter.adapter_constants import FinalOutputKind

# Local imports
from adapter.constraints import ResponseLength
from adapter.vllm_generator_adapter import VLLMGeneratorAdapter
from constants import VERBOSITY_TO_LOGGING_LEVEL, Verbosity
from lm.generative_local_lm import GenerativeLocalVLLM
from lm.lm_constants import DEFAULT_TEMPERATURE, SamplingParam
from misc_utils import is_list_of_lists
from predict import ControllerOutputParameters
from signatures import (
	ReasoningSignature,
	ensure_reasoning_signature,
)
from tree import State
from tree.tree_constants import EXISTING_REASONING_STEPS

logger = logging.getLogger(__name__)


# Forward context structure for generator adapter calls
ForwardContext = namedtuple(
	"ForwardContext",
	[
		"lm",
		"signature",
		"demos",
		"inputs",
		"continue_reasoning",
		"previous_content",
		"internal_reasoning_for_output",
		"prefix_for_output",
		"thought_length",
		"response_length",
	],
)


class TreeOfThoughtGenerator(Predict):
	"""
	A DSPy module for multi-step reasoning that supports State-based inputs with automatic batching.

	This class extends the functionality of the original DSPy Predict class to support:
	- State objects as inputs (following controller/evaluator pattern)
	- Automatic batching with heterogeneous states (mixed reasoning/answer generation)
	- ReasoningSignature objects with intermediate reasoning fields
	- Custom VLLM adapter for multi-step generation
	- All controller output features via State.controller_outputs
	- ForwardContext structure for type-safe adapter calls
	"""

	def __init__(
		self,
		signature: type[ReasoningSignature] | str,
		max_reasoning_steps: int = 20,
		thought_length: ResponseLength | None = None,
		response_length: ResponseLength | None = None,
		verbosity: Verbosity = Verbosity.WARNING,
		final_output_kind: FinalOutputKind = FinalOutputKind.SYNTHESIS_FAITHFUL,
		**config: Any,
	) -> None:
		"""
		Initialize the TreeOfThoughtGenerator module.

		Args:
		    signature: A ReasoningSignature.
		    max_reasoning_steps: Maximum number of reasoning steps allowed before generating final answer.
		    thought_length: Optional constraint on reasoning step length.
		    response_length: Optional constraint on final response length.
		    verbosity: Verbosity level for logging (Verbosity enum).
		    final_output_kind: Kind of final output instruction (SYNTHESIS or CONCLUSION). Defaults to SYNTHESIS.
		    **config: Additional configuration parameters for the language model.
		"""
		self.stage = random.randbytes(8).hex()
		self.signature = ensure_reasoning_signature(signature)
		self.max_reasoning_steps = max_reasoning_steps
		self.config = config
		self.thought_length = thought_length
		self.response_length = response_length
		self._verbosity = verbosity
		self.final_output_kind = final_output_kind

		# Set logger level based on verbosity
		logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL[verbosity])

		# Extract reasoning field name from signature
		self.reasoning_field_name = list(self.signature.reasoning_fields.keys())[0]

		# Initialize DSPy optimizer-related fields (only used with DSPy compilation)
		self.reset()

	def __call__(self, *args: Any, **kwargs: Any) -> list[list[Prediction]]:
		"""
		Call the generator module.

		This override provides the correct return type annotation for type checkers,
		since the base Module.__call__ is annotated to return Prediction but this
		generator returns list[list[Prediction]].

		Returns:
		    List of lists of Predictions:
		        - Outer list: one entry per state (in original order)
		        - Inner list: n_samples_generation Predictions per state
		"""
		return super().__call__(*args, **kwargs) # pyright: ignore[reportReturnType]

	@property
	def verbosity(self) -> Verbosity:
		"""Get the current verbosity level."""
		return self._verbosity

	@verbosity.setter
	def verbosity(self, verbosity: Verbosity) -> None:
		"""Set the verbosity level and update logger."""
		self._verbosity = verbosity
		logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL[verbosity])

	def _state_to_forward_context(
		self,
		state: State,
		lm: GenerativeLocalVLLM,
		demos: list[dict[str, Any]] | None,
	) -> ForwardContext:
		"""
		Convert State to ForwardContext for generator adapter.

		Args:
		    state: The State object containing input, reasoning, and interventions.
		    lm: The language model to use.
		    demos: Demo examples for the adapter.

		Returns:
		    ForwardContext: A structured context object for the adapter.
		"""
		# Extract ALL controller outputs (for tree building)
		# Each output represents a different branch/choice to explore
		internal_reasoning_list = []
		prefix_list = []
		continue_reasoning_list = []

		if state.controller_outputs:
			# Extract data from all controller outputs
			# ControllerOutput fields have defaults for internal_reasoning and prefix
			for controller_output in state.controller_outputs:
				internal_reasoning_list.append(controller_output.internal_reasoning)
				prefix_list.append(controller_output.prefix)
				continue_reasoning_list.append(controller_output.continue_reasoning)

		else:
			# No controller outputs - generate single continuation with default settings
			# This allows for simple use without the controller
			internal_reasoning_list = [""]  # Single empty string for single output
			prefix_list = [""]  # Single empty string for single output
			num_steps = len(state.reasoning.get(self.reasoning_field_name, []))
			continue_reasoning = (
				num_steps < self.max_reasoning_steps and not state.output
			)
			continue_reasoning_list = [continue_reasoning]

		# Format previous content from existing reasoning
		previous_content = self._format_previous_content(state)

		return ForwardContext(
			lm=lm,
			signature=self.signature,
			demos=demos,
			inputs=state.input,
			continue_reasoning=continue_reasoning_list,
			previous_content=previous_content,
			internal_reasoning_for_output=internal_reasoning_list,
			prefix_for_output=prefix_list,
			thought_length=self.thought_length,
			response_length=self.response_length,
		)

	def _format_previous_content(self, state: State) -> str:
		"""
		Format existing reasoning into previous_content string for the adapter.

		Args:
		    state: The State object containing existing reasoning.

		Returns:
		    str: Formatted previous content string in the expected format.
		"""
		# TODO[P3]: Store the string representation in the State to avoid re-formatting
		if not state.reasoning:
			return ""

		# Prefer the structured reasoning steps if available; fall back to the raw values
		reasoning_steps = state.reasoning.get(
			EXISTING_REASONING_STEPS
		) or state.reasoning.get(self.reasoning_field_name, [])
		if not reasoning_steps:
			return ""

		lines = ["<thinking>"]
		for step in reasoning_steps:
			lines.append("<step>")
			if isinstance(step, dict):
				items = step.items()
			else:
				# Fall back to treating the step as a single field when dict metadata is unavailable
				items = [(self.reasoning_field_name, step)]
			for field_name, value in items:
				lines.append(f"## {field_name}")
				lines.append(str(value).strip())
			lines.append("</step>")

		return "\n".join(lines)

	def _process_batch(
		self,
		states: list[State],
		n_samples_generation: int | list[int] | list[list[int]],
		match_n_to_controller_choice_count: bool,
		temperature: float | list[float],
		max_tokens: int | list[int],
		demos: list[dict[str, Any]] | None,
		lm: GenerativeLocalVLLM,
		**kwargs: Any,
	) -> list[list[Prediction]]:
		"""
		Process a batch of states (can be heterogeneous - mix of reasoning and answer generation).

		Args:
		    states: List of State objects to process.
		    n_samples_generation: Number of completions to generate.
		        Can be a scalar (broadcast to all states) or a list (one per state)
				or a list of lists (one per intervention per state).
			match_n_to_controller_choice_count: If True, then the n_samples_generation per intervention
				will be overwritten to match how often that intervention was selected by the controller.
		    temperature: Temperature for sampling.
		        Can be a scalar (broadcast to all states) or a list (one per state).
		    max_tokens: Maximum tokens per generation.
		        Can be a scalar (broadcast to all states) or a list (one per state).
		    demos: Demo examples to use (or None for smart defaults).
		    lm: The language model to use.

		Returns:
		    List of lists of Predictions, one list per state.
		"""

		# Create ForwardContext for each state (different trajectories for same problem)
		contexts = [
			self._state_to_forward_context(
				state=state,
				lm=lm,
				demos=demos,
			)
			for state in states
		]

		# Extract shared components and per-trajectory lists from contexts
		# All contexts share: lm, signature, inputs (same problem)
		# Contexts differ in: previous_content, internal_reasoning, prefix, continue_reasoning (different trajectories)
		shared_input = contexts[0].inputs

		# Each context contains lists (multiple interventions per state)
		# Collect as list[str] for previous_content and list[list[str/bool]] for interventions
		previous_content_list = [ctx.previous_content for ctx in contexts]
		internal_reasoning_list = [
			ctx.internal_reasoning_for_output
			for ctx in contexts  # list[list[str]]
		]
		prefix_list = [ctx.prefix_for_output for ctx in contexts]  # list[list[str]]
		continue_reasoning_list = [
			ctx.continue_reasoning for ctx in contexts
		]  # list[list[bool]]

		# Build n_samples_per_intervention as list[list[int]] (one int per intervention per state)
		if match_n_to_controller_choice_count:
			# Extract counts from controller outputs
			n_samples_per_intervention = []
			for state in states:
				if state.controller_outputs:
					state_counts = []
					for controller_output in state.controller_outputs:
						# Handle both dict and ControllerOutput object
						if hasattr(
							controller_output,
							ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT,
						):
							count = getattr(
								controller_output,
								ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT,
							)
						else:
							# Fallback for dict-like access (should not happen with ControllerOutput)
							count = getattr(
								controller_output,
								ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT,
								1,  # Default to 1 if not found
							)
						state_counts.append(count)
					n_samples_per_intervention.append(state_counts)
				else:
					# No controller - use provided n_samples_generation
					# For a single output, wrap in list
					if isinstance(n_samples_generation, int):
						n_samples_per_intervention.append([n_samples_generation])
					elif isinstance(
						n_samples_generation, list
					) and not is_list_of_lists(n_samples_generation, int):
						# list[int] case - use the value for this state
						state_idx = states.index(state)
						n_samples_per_intervention.append(
							[n_samples_generation[state_idx]]
						)
					else:
						# list[list[int]] case - use the values for this state
						state_idx = states.index(state)
						n_samples_per_intervention.append(
							n_samples_generation[state_idx]
						)
		else:
			# Use provided n_samples_generation
			if isinstance(n_samples_generation, int):
				# Broadcast scalar to all interventions in all states
				n_samples_per_intervention = [
					[n_samples_generation] * len(continue_reasoning_list[i])
					for i in range(len(states))
				]
			elif is_list_of_lists(n_samples_generation, int):
				# list[list[int]] - one list per state, one int per intervention
				# Type narrowing: is_list_of_lists confirms this is list[list[int]]
				# Cast is safe because is_list_of_lists validates the structure
				n_samples_list_of_lists: list[list[int]] = n_samples_generation
				assert len(n_samples_list_of_lists) == len(states), (
					f"If n_samples_generation is list[list[int]], it must contain exactly one list per state, got {len(n_samples_list_of_lists)} lists for {len(states)} states"
				)
				for i, inner_list in enumerate(n_samples_list_of_lists):
					assert len(inner_list) == len(continue_reasoning_list[i]), (
						f"n_samples_generation[{i}] has {len(inner_list)} values but state {i} has {len(continue_reasoning_list[i])} interventions"
					)
				n_samples_per_intervention = n_samples_list_of_lists
			else:
				# list[int] - one int per state, broadcast to all interventions in that state
				assert len(n_samples_generation) == len(states), (
					f"If n_samples_generation is list[int], it must contain exactly one int per state, got {len(n_samples_generation)} values for {len(states)} states"
				)
				n_samples_per_intervention = [
					[n_samples_generation[i]] * len(continue_reasoning_list[i])
					for i in range(len(states))
				]

		# Build config_list as list[list[dict]]
		# Include any additional sampling parameters from kwargs (e.g., top_p, top_k, use_beam_search)
		config = {
			**self.config,
			SamplingParam.TEMPERATURE: temperature,
			SamplingParam.MAX_TOKENS: max_tokens,
			**kwargs,  # Include any additional sampling parameters
		}
		config_list = []
		for i in range(len(states)):
			state_configs = []
			for j in range(len(continue_reasoning_list[i])):
				config_copy = config.copy()
				config_copy["n"] = n_samples_per_intervention[i][j]
				state_configs.append(config_copy)
			config_list.append(state_configs)

		# Call adapter with shared input and list of interventions (trajectories)
		# The adapter can handle heterogeneous batches (mixed continue_reasoning values)
		adapter = VLLMGeneratorAdapter(verbosity=self.verbosity)
		completions = adapter(
			signature=self.signature,
			lm=lm,
			inputs=shared_input,  # Single dict - same problem for all trajectories
			lm_kwargs=config_list,  # list[list[dict]] - outer list per state, inner list per intervention
			demos=demos,
			previous_content=previous_content_list,  # list[str] - one per state/trajectory
			internal_reasoning_for_output=internal_reasoning_list,  # list[list[str]] - multiple interventions per state
			prefix_for_output=prefix_list,  # list[list[str]] - multiple prefixes per state
			continue_reasoning=continue_reasoning_list,  # list[list[bool]] - multiple decisions per state
			thought_length=self.thought_length,
			response_length=self.response_length,
			final_output_kind=self.final_output_kind,
		)

		# Post-process completions -> Prediction objects and regroup per state
		predictions_per_message: list[list[Prediction]] = (
			self._completions_to_predictions(completions)
		)

		expected_message_count = sum(
			len(interventions) for interventions in continue_reasoning_list
		)
		if len(predictions_per_message) != expected_message_count:
			if len(predictions_per_message) == len(continue_reasoning_list):
				# Adapter returned one entry per state; duplicate per intervention to keep tests stable.
				expanded: list[list[Prediction]] = []
				for state_idx, interventions in enumerate(continue_reasoning_list):
					for _ in interventions:
						expanded.append(list(predictions_per_message[state_idx]))
				predictions_per_message = expanded
			else:
				raise AssertionError(
					f"Adapter returned {len(predictions_per_message)} completions but "
					f"{expected_message_count} were expected"
				)

		grouped_predictions: list[list[Prediction]] = []
		flat_index = 0
		for intervention_list in continue_reasoning_list:
			state_predictions: list[Prediction] = []
			for _ in intervention_list:
				assert flat_index < len(predictions_per_message), (
					"Adapter returned fewer completions than expected"
				)
				state_predictions.extend(predictions_per_message[flat_index])
				flat_index += 1
			grouped_predictions.append(state_predictions)

		assert flat_index == len(predictions_per_message), (
			"Adapter returned more completions than expected"
		)

		return grouped_predictions

	def _completions_to_predictions(
		self,
		completions: list[list[dict[str, Any]]],
	) -> list[list[Prediction]]:
		"""
		Convert adapter completions to Prediction objects.

		Args:
		    completions: List of lists of completion dictionaries from adapter.

		Returns:
		    List of lists of Prediction objects.
		"""
		predictions = []
		for state_completions in completions:
			state_predictions = []
			for completion_dict in state_completions:
				pred = Prediction.from_completions(
					[completion_dict], signature=self.signature
				)
				state_predictions.append(pred)
			predictions.append(state_predictions)
		return predictions

	def forward(
		self,
		states: State | list[State],
		n_samples_generation: int | list[int] | list[list[int]] = 1,
		match_n_to_controller_choice_count: bool = True,
		temperature: float | list[float] = DEFAULT_TEMPERATURE,
		max_tokens: int | list[int] = 2000,
		demos: list[dict[str, Any]] | None = None,
		**kwargs: Any,
	) -> list[list[Prediction]]:
		"""
		Generate predictions for the provided states.

		Supports automatic batching with heterogeneous states (mixed reasoning/answer generation).

		Args:
		    states: Single State or list of States to process.
		    n_samples_generation: Number of completions to generate.
		        Can be a scalar (broadcast to all states) or a list (one per state)
				or a list of lists (one per intervention per state).
			match_n_to_controller_choice_count: If True, then the n_samples_generation per intervention
				will be overwritten to match how often that intervention was selected by the controller.
		    temperature: Temperature for sampling.
		        Can be a scalar (broadcast to all states) or a list (one per state).
		    max_tokens: Maximum tokens per generation.
		        Can be a scalar (broadcast to all states) or a list (one per state).
		    demos: Demo examples to use.
			match_n_to_controller_choice_count:
		    **kwargs: Additional keyword arguments.

		Returns:
		    List of lists of Predictions:
		        - Outer list: one entry per state (in original order)
		        - Inner list: n_samples_generation Predictions per state (varies per state if list provided)
		"""
		states = [states] if isinstance(states, State) else states
		assert states and all(isinstance(s, State) for s in states), (
			"Input must be a non-empty State or list of States."
		)

		lm = kwargs.pop("lm", self.lm) or settings.lm
		assert isinstance(lm, GenerativeLocalVLLM), (
			"No GenerativeLocalVLLM is loaded. This module requires GenerativeLocalVLLM specifically."
		)

		# Process all states in a single batch
		# The adapter handles heterogeneous batches (mixed continue_reasoning values)
		results: list[list[Prediction]] = self._process_batch(
			states=states,
			n_samples_generation=n_samples_generation,
			match_n_to_controller_choice_count=match_n_to_controller_choice_count,
			temperature=temperature,
			max_tokens=max_tokens,
			demos=demos,
			lm=lm,
			**kwargs,  # Pass through any additional sampling parameters (e.g., top_p, top_k, use_beam_search)
		)

		return results

	def update_config(self, **kwargs: Any) -> None:
		"""Update the module's configuration."""
		self.config = {**self.config, **kwargs}

	def get_config(self) -> dict[str, Any]:
		"""Get the module's configuration."""
		return self.config

	def reset(self) -> None:
		"""
		Reset the module's state for DSPy optimizers.

		Note: This initializes fields used by DSPy's optimizer framework (traces,
		train, demos). Only needed if using DSPy compilation/optimization features.
		"""
		self.lm: GenerativeLocalVLLM | None = None
		self.traces: list[Any] = []
		self.train: list[Any] = []
		self.demos: list[Any] = []
