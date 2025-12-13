"""
Controller: A DSPy module that determines the next action to take when solving a reasoning problem.
The controller's choice of action maps to a textual prefix which steers a subsequent LLM generation.
"""

# Standard library imports
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_type_hints

# Third-party imports
import dspy
from dspy import Tool

# Local imports
from constants import Verbosity
from lm.lm_constants import SamplingParam
from misc_utils import (
	parse_base_signature,
	parse_literal,
	safe_parse_dict,
	stringify,
)
from predict.controller_constants import (
	DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
	ControllerActionParameters,
	ControllerConfig,
	ControllerOutputParameters,
)
from predict.controller_utils import (
	DEFAULT_TOOL,
	FINISH_TOOL,
	ActionSpaceConfig,
	ControllerPrediction,
	ForcedChoiceFunction,
	ReasoningIntervention,
	create_literal_from_dict,
	create_reasoning_intervention_from_choices,
	execute_tool_safely,
	load_action_space_json,
	remove_duplicate_actions_with_counts,
	return_action_if_single_option,
	sanitize_param_name,
)
from predict.local_predict import LocalPredict
from signatures import (
	ReasoningSignature,
	ensure_reasoning_signature,
)
from tree import State
from tree.tree_constants import ReasoningState

logger = logging.getLogger(__name__)


class TreeOfThoughtsController(dspy.Module):
	"""
	A module for explicitly controlling aspects of the next action within an Additive Reasoning Tree of Thoughts.

	Supports single-candidate generation which produces one action + argument(s) combination.
	"""

	@staticmethod
	def _create_combined_action_tool_func(
		configs: list[ActionSpaceConfig]
	) -> Callable[[dict[str, str]], ReasoningIntervention]:
		"""
		Create a function that takes a dictionary of choices and returns a ReasoningIntervention.

		The tool takes one choice per dimension. When executed, it returns a ReasoningIntervention,
		which reflects whether to continue reasoning or generate final output. If continuing reasoning,
		it also includes an internal reasoning and/or a prefix to inject at the start of the next
		generation.

		Parameters:
			configs: List of ActionSpaceConfig objects, one per dimension.

		Returns:
			A tool function for use in a dspy.Tool that returns a ReasoningIntervention.
		"""
		def tool_func(**kwargs) -> ReasoningIntervention:
			"""Creates a ReasoningIntervention from the Controller's choices."""
			intervention = create_reasoning_intervention_from_choices(configs, kwargs)
			return intervention
		return tool_func

	@staticmethod
	def _load_action_spaces_and_create_combined_tool(
		json_paths: list[str | Path],
		tool_name: str = DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
	) -> Tool:
		"""
		Load multiple action space JSONs and create a SINGLE combined tool.

		For the generative controller, we create ONE tool with (potentially) multiple parameters,
		where each parameter corresponds to one dimension (action space JSON).
		The generative model is expected to generate one choice for each parameter given the
		permitted choices for each parameter.

		Parameters:
			json_paths: List of paths to action space JSON files. The generative controller must
				select one choice for each parameter (corresponding to one json file/dimension).
			tool_name: Name for the combined tool (default: DEFAULT_REASONING_INTERVENTION_TOOL_NAME).

		Returns:
			A single Tool with (potentially) multiple parameters.
		"""
		# Load all configs
		configs: list[ActionSpaceConfig] = []
		for path in json_paths:
			configs.append(load_action_space_json(path))

		# Create parameter names from dimension names
		param_names = [sanitize_param_name(config.name) for config in configs]
		assert len(param_names) == len(set(param_names)), "Parameters must have unique names"

		# Create combined tool function
		tool_func: Callable[[dict[str, str]], ReasoningIntervention] = (
			TreeOfThoughtsController._create_combined_action_tool_func(configs)
		)

		# Build combined description
		desc_lines = [
			"Determine the best choice in each dimension (corresponding to an argument of this function) to improve the quality of the next reasoning step.",
			"You **must** select **one** choice for **each** of the provided dimensions.",
			"Be mindful of the impact that each choice has on the next reasoning step.",
			"When providing a choice, avoid enumeration or unnecessary prefixes/suffixes -- just provide your choice directly."
		]
		tool_desc = "\n".join(desc_lines)

		# Build args dict (parameter descriptions)
		args: dict[str, str] = {}
		type_overrides: dict[str, Any] = {}
		for config, param_name in zip(configs, param_names, strict=True):
			# Build detailed argument description
			arg_desc_lines = [config.definition, "Options:"]
			for choice_name, choice_data in config.choices.items():
				definition = choice_data.get(ControllerOutputParameters.DEFINITION, "")
				arg_desc_lines.append(f'- "{choice_name}": {definition}')
			args[param_name] = "\n".join(arg_desc_lines).strip()
			type_overrides[param_name] = create_literal_from_dict(config.choices)

		logger.debug(
			f"Created combined tool '{tool_name}' with {len(configs)} dimensions: "
			f"`{[c.name for c in configs]}`"
		)
		return dspy.Tool(
			name=tool_name,
			func=tool_func,
			desc=tool_desc,
			args=args,
			arg_types=type_overrides,
		)

	def create_basic_tools(
		self,
		provided_tools: list[dspy.Tool] | None,
		action_space_paths: list[str | Path] | None,
		early_stopping_enabled: bool,
		reasoning_intervention_tool_name: str = DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
	) -> list[dspy.Tool]:
		"""
		Create the basic tools list for the controller.

		Creates tools from action_space_paths and/or provided_tools. If neither is provided,
		uses DEFAULT_TOOL. Optionally adds FINISH_TOOL if early stopping is enabled.

		Parameters:
			provided_tools: List of dspy.Tool instances provided by the user, or None.
			action_space_paths: Paths to action space JSON files, or None.
			early_stopping_enabled: Whether to include the early stopping tool.
			reasoning_intervention_tool_name: Name for the reasoning intervention tool.

		Returns:
			List of dspy.Tool instances to use in the controller.
		"""
		tools: list[dspy.Tool] = []

		# Create combined tool from action_space_paths if provided
		if action_space_paths is not None:
			# For generative controller: create ONE combined tool with multiple parameters
			# This tool is meant to perform interventions on the next reasoning step of the
			# generator.
			combined_tool: Tool = self._load_action_spaces_and_create_combined_tool(
				action_space_paths,
				tool_name=reasoning_intervention_tool_name,
			)
			tools.append(combined_tool)
			logger.info(
				f"Created combined tool '{combined_tool.name}' with "
				f"{len(combined_tool.args)} dimension parameters from action space JSONs"
			)

		# Add provided tools
		if provided_tools is not None:
			for tool in provided_tools:
				if isinstance(tool, dspy.Tool):
					tools.append(tool)
				else:  # tool is a callable. Create a dspy.Tool from it.
					assert hasattr(tool, "__name__"), "Tool must have a name"
					assert hasattr(tool, "__doc__"), "Tool must have a description"
					assert get_type_hints(tool) is not None, "Tool must have annotations"
					tools.append(dspy.Tool(tool))

		# Fall back to DEFAULT_TOOL if no tools were provided
		if not tools:
			tools = [DEFAULT_TOOL]

		if early_stopping_enabled:
			tools.append(FINISH_TOOL)

		return tools

	def __init__(
		self,
		signature: type[ReasoningSignature],
		max_reasoning_steps: int,
		tools: list[Callable | Tool] | None = None,
		action_space_paths: list[str | Path] | None = None,
		forced_choice_function: ForcedChoiceFunction = return_action_if_single_option,
		early_stopping_enabled: bool = True,
		verbosity: Verbosity = Verbosity.WARNING,
	) -> None:
		"""
		Initialize the TreeOfThoughtsController.

		Parameters:
			signature (dspy.Signature): The base signature for the reasoning task.
			max_reasoning_steps (int): The maximum number of reasoning steps allowed.
			tools (list[Callable | Tool] | None): A list of functions (callable objects), or
				`dspy.Tool` instances. Can be used alone or combined with action_space_paths.
				If both are None, uses DEFAULT_TOOL.
			action_space_paths (list[str | Path] | None): Paths to action space JSON files. Each JSON
				defines a dimension (e.g., structure, style, or subtopic) with choices that can be
				selected. Creates ONE combined tool with parameters for each dimension. Can be used
				alone or combined with tools.
			forced_choice_function (ForcedChoiceFunction): A function that takes available tools
				(dict[str, Tool]) and state, returning a list of (action_name, action_arguments,
				considerations) tuples or None if no forced choice. The considerations string
				explains why this action was chosen.
			early_stopping_enabled (bool): Whether to include the early stopping tool.
			verbosity (Verbosity): Verbosity level for logging (Verbosity enum).
		"""
		super().__init__()

		self.base_signature = ensure_reasoning_signature(signature)
		self.input_field_names = list(self.base_signature.input_fields.keys())
		self.output_field_names = list(self.base_signature.output_fields.keys())
		self.reasoning_field_name = list(self.base_signature.reasoning_fields.keys())[0]
		self.max_reasoning_steps = max_reasoning_steps
		self.forced_choice_function = forced_choice_function
		self._verbosity = verbosity

		#TODO[P2]: Add support for more complex forced choice functions that can return
		# interventions with internal_reasoning and prefix fields populated, not just
		# continue_reasoning.

		# Create basic tools from provided tools, action space JSONs, or use default
		tools_list: list[dspy.Tool] = self.create_basic_tools(
			provided_tools=tools,
			action_space_paths=action_space_paths,
			early_stopping_enabled=early_stopping_enabled,
		)

		# Create dict of tools by name
		self.tools = {tool.name: tool for tool in tools_list}

		# Check if any tool has arguments - if not, we don't need ARGUMENTS in output
		self.tools_have_arguments = any(
			tool.args and len(tool.args) > 0 for tool in self.tools.values()
		)

		# Create single-candidate predictor
		self.decide_next_step_single = LocalPredict(
			signature=self._create_controller_signature_single_candidate(),
			verbose=verbosity,
		)

	@property
	def verbosity(self) -> Verbosity:
		"""Get the current verbosity level."""
		return self._verbosity

	@verbosity.setter
	def verbosity(self, verbosity: Verbosity) -> None:
		"""Set the verbosity level."""
		self._verbosity = verbosity


	def _create_tool_instructions(self) -> list[str]:
		"""
		Create instructions for available tools.
		Following the DSPy ReACT module:
			https://github.com/stanfordnlp/dspy/blob/103d3d7b336c58c3ab659d002b2a7b57766937c2/dspy/predict/react.py#L41
		"""
		tool_instructions = []
		for idx, tool in enumerate(self.tools.values()):
			desc = (
				f", whose description is <desc>{tool.desc}</desc>."
				if tool.desc
				else "."
			).replace("\n", "  ")
			args = getattr(tool, "args", {})
			if len(args) > 0:
				desc += f" It takes the following arguments in JSON format: {args}."
			tool_instructions.append(f"({idx + 1}) `{tool.name}`{desc}")
		return tool_instructions

	def _create_controller_signature_single_candidate(self) -> dspy.Signature:
		"""
		Create signature for the controller to generate a candidate an action and considerations.
		"""
		inputs, outputs = parse_base_signature(
			input_field_names=self.input_field_names,
			output_field_names=self.output_field_names,
		)
		instructions = [self.base_signature.instructions]

		instructions.extend(
			[
				f"""
You are given {inputs} and your goal is to finish with {outputs}.
To accomplish this goal, you will need to reason about the problem step by step rather than generating {outputs} directly.
You have up to `{ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS}` additional steps to reason about the problem before generating {outputs}.
Refer to the existing reasoning stored in `{self.reasoning_field_name}` to inform your next step.
You may choose from the following actions:
{self._create_tool_instructions()}
""",
			]
		)
		# Create input fields without type_ parameters (they're ignored anyway)
		input_fields = {
			**self.base_signature.input_fields,
			self.reasoning_field_name: dspy.InputField(
				desc=f"The existing reasoning steps towards producing {outputs}."
			),
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: dspy.InputField(
				desc=f"The maximum number of additional reasoning steps you can take before you must produce {outputs}."
			),
		}

		# Create output fields
		output_fields = {
			ControllerActionParameters.CONSIDERATIONS: dspy.OutputField(
				desc="The thought process behind choosing the next action to take.",
			),
			ControllerActionParameters.ACTION: dspy.OutputField(
				desc=f"The selected action (function) to take. The chosen action influences the next reasoning step towards producing {outputs}."
			),
		}

		# Conditionally create constants.ACTION_ARGUMENTS field
		if self.tools_have_arguments:
			output_fields[ControllerActionParameters.ARGUMENTS] = dspy.OutputField(
				desc="The input arguments for the selected action function. "
				"Return an empty dictionary for actions that do not require any arguments."
			)

		# Fields are stored as tuples: (field_type, field_info)
		signature_fields = {}
		for name, field in self.base_signature.input_fields.items():
			field_type = (
				field.annotation
				if hasattr(field, "annotation") and field.annotation
				else str
			)
			signature_fields[name] = (field_type, field)

		signature_fields[self.reasoning_field_name] = (
			str, input_fields[self.reasoning_field_name]
		)
		signature_fields[ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS] = (
			int, input_fields[ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS]
		)
		signature_fields[ControllerActionParameters.CONSIDERATIONS] = (
			str, output_fields[ControllerActionParameters.CONSIDERATIONS]
		)
		signature_fields[ControllerActionParameters.ACTION] = (
			str, output_fields[ControllerActionParameters.ACTION]
		)

		if self.tools_have_arguments:
			signature_fields[ControllerActionParameters.ARGUMENTS] = (
				dict[str, Any],
				output_fields[ControllerActionParameters.ARGUMENTS],
			)

		return dspy.Signature(signature_fields, "\n".join(instructions))

	def _state_to_controller_input(self, state: State) -> dict[str, Any]:
		"""
		Convert the state to the input for the decision-making controller.

		Parameters:
		    state (State): The current state of the tree of thoughts.

		Returns:
		    dict[str, Any]: The input for the decision-making controller.
		"""
		input_fields = {**state.model_dump()[ReasoningState.INPUT]}
		existing_reasoning = state.reasoning.get(self.reasoning_field_name, [])
		number_of_existing_reasoning_steps = len(existing_reasoning)
		input_fields[self.reasoning_field_name] = stringify(existing_reasoning)
		input_fields[ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS] = (
			self.max_reasoning_steps - number_of_existing_reasoning_steps
		)
		return input_fields

	def create_predictions_from_forced_choices(
		self,
		forced_results_list: list[tuple[str, dict[str, Any], str]],
	) -> list[ControllerPrediction]:
		"""Create predictions from forced choices.

		Args:
		    forced_results_list (list[tuple[str, dict[str, Any], str]]): List of tuples containing
				the action name (strings), arguments (dictionaries mapping argument names for the
				tool to their values), and considerations (strings representing the thought
				process leading to the choice of the action).

		Returns:
		    list[ControllerPrediction]: List of ControllerPrediction objects reflecting the chosen
				interventions to perform on generations from a given state.
		"""
		forced_preds = []
		for action_name, action_arguments, considerations in forced_results_list:
			assert action_name in self.tools, f"Tool '{action_name}' not in {self.tools.keys()}."
			tool = self.tools[action_name]
			intervention, error_message = execute_tool_safely(tool, action_arguments)
			forced_preds.append(
				ControllerPrediction(
					tool=tool,
					chosen_values=action_arguments,
					intervention=intervention,
					considerations=considerations,
					tool_execution_error=error_message,
				)
			)
		return forced_preds

	def create_controller_predictions(
		self,
		prediction: dspy.Prediction,
	) -> list[ControllerPrediction]:
		"""
		Convert the controller output to a list of ControllerPrediction objects.

		Args:
		    prediction (dspy.Prediction): The prediction object produced by the underlying
				generative LLM. The prediction includes objects corresponding to the output
				fields of the controller's signature.

		Returns:
		    list[ControllerPrediction]: List of ControllerPrediction objects.
		"""
		# Helper to ensure list
		# TODO[P3]: Remove helpers like this, and move to misc_utils.py. Avoid defining helpers
		# inside of other functions.
		def to_list(x):
			return x if isinstance(x, list) else [x]

		actions = to_list(prediction.completions[ControllerActionParameters.ACTION])
		considerations = to_list(prediction.completions[ControllerActionParameters.CONSIDERATIONS])
		args_list = (
			to_list(prediction.completions[ControllerActionParameters.ARGUMENTS])
			if self.tools_have_arguments
			else [{}] * len(actions)
		)

		# Validate equal completion field lengths
		assert len(actions) == len(considerations) == len(args_list), (
			f"Mismatch in completion lengths: actions={len(actions)}, "
			f"considerations={len(considerations)}, args={len(args_list)}"
		)

		# Prepare dictionaries for deduplication
		output_dicts = []
		for i, (action, rationale, args_raw) in enumerate(zip(actions, considerations, args_list, strict=True)):
			args = safe_parse_dict(args_raw) if self.tools_have_arguments else {}
			assert action and isinstance(action, str) and action.strip(), (
				f"Model generated empty or invalid action at index {i}: '{action}'. "
				f"Available tools: {list(self.tools.keys())}. "
			)
			action = parse_literal(action).strip("\"'`")
			assert action in self.tools, (
				f"Model generated unknown action '{action}' at index {i}. "
				f"Available tools: {list(self.tools.keys())}. "
			)
			output_dicts.append({
				ControllerActionParameters.ACTION: action,
				ControllerActionParameters.ARGUMENTS: args,
				ControllerActionParameters.CONSIDERATIONS: rationale,
			})

		# Deduplicate
		unique_outputs = remove_duplicate_actions_with_counts(output_dicts)

		controller_predictions = []
		for output in unique_outputs:
			action = output[ControllerActionParameters.ACTION]
			args = output[ControllerActionParameters.ARGUMENTS]
			rationale = output[ControllerActionParameters.CONSIDERATIONS]
			count = output[ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT]

			tool = self.tools[action]
			intervention, error = execute_tool_safely(tool, args)

			controller_predictions.append(
				ControllerPrediction(
					tool=tool,
					chosen_values=args,
					intervention=intervention,
					considerations=rationale,
					tool_execution_error=error,
					num_occurrences=count,
				)
			)

		return controller_predictions


	def get_controller_predictions_from_forced_choices(
		self,
		state: State,
		n_samples_generation: int,
	) -> list[ControllerPrediction] | None:
		"""Get forced controller predictions for a given state if available.

		Forced predictions are either stored in the (parent) state before the controller is
		called or generated by the forced choice function (passed into the controller at
		initialization).

		Args:
		    state (State): The current state of the tree of thoughts.
		    n_samples_generation (int): The number of samples to generate.

		Returns:
		    list[ControllerPrediction] | None: The forced controller predictions for the given
				state.
		"""
		forced_choices = (
			state.forced_controller_outputs
			or self.forced_choice_function(self.tools, state)
		)
		if not forced_choices:
			return None
		forced_preds = self.create_predictions_from_forced_choices(forced_choices)
		
		# Broadcast single forced choice if necessary
		if len(forced_preds) == 1 and n_samples_generation > 1:
			forced_preds = forced_preds * n_samples_generation

		assert len(forced_preds) == n_samples_generation, (
			f"Forced choices must have length {n_samples_generation}, but got {len(forced_preds)}"
		)
		return forced_preds

	def _generate_lm_predictions(
		self,
		lm_states_with_indices: list[tuple[int, State]],
		n_samples_generation: int,
		temperature: float,
		max_tokens: int,
		demos: list[dict[str, Any]] | None,
		controller_sampling_params: dict[str, Any] | None,
	) -> dict[int, list[ControllerPrediction]]:
		"""
		Generate predictions using the language model for states that require it.

		Args:
			lm_states_with_indices: List of tuples (original_index, state).
			n_samples_generation: Number of actions to generate per state (duplicates allowed).
			temperature: Sampling temperature.
			max_tokens: Max tokens for generation.
			demos: Examples for the prompt.
			controller_sampling_params: Additional sampling parameters.

		Returns:
			dict[int, list[ControllerPrediction]]: Map of original index to predictions.
		"""
			# Extract states and build batch inputs
		lm_states = [state for _, state in lm_states_with_indices]
		batch_inputs = [self._state_to_controller_input(s) for s in lm_states]

		# Auto-batching kwargs
		input_keys = list(batch_inputs[0].keys())
		batched_kwargs = {k: [inp[k] for inp in batch_inputs] for k in input_keys}

		config = controller_sampling_params or {}
		config.update(
			{
				SamplingParam.N: n_samples_generation,
				SamplingParam.TEMPERATURE: temperature,
				SamplingParam.MAX_TOKENS: max_tokens,
			}
		)

		# TODO[P3]: Avoid using `forward(...)` and instead directly call the predictor
		predictions = self.decide_next_step_single.forward(
			config=config, demos=demos, **batched_kwargs
		)

		# Allow one-to-one or one-to-N matching; flexible for different predictor behaviors
		if len(predictions) != len(lm_states_with_indices):
			raise ValueError(f"Expected {len(lm_states_with_indices)} preds, got {len(predictions)}")

		results = {}
		for (orig_idx, _), pred in zip(lm_states_with_indices, predictions, strict=True):
			lm_preds: list[ControllerPrediction] = self.create_controller_predictions(pred)

			# Ensure exact count by cycling
			assert sum([pred.num_occurrences for pred in lm_preds]) == n_samples_generation, (
				f"Expected {n_samples_generation} predictions, "
				f"got {sum([pred.num_occurrences for pred in lm_preds])}"
			)
			results[orig_idx] = lm_preds

		return results

	def forward(
		self,
		states: State | list[State],
		n_samples_generation: int = 1,
		temperature: float = 0.7,
		max_tokens: int = 2000,
		demos: list[dict[str, Any]] | None = None,
		controller_sampling_params: dict[str, Any] | None = None,
		**kwargs,
	) -> list[list[ControllerPrediction]]:
		"""
		Forward method that automatically uses batch processing where applicable.

		Parameters:
		    states (Union[State, List[State]]): Single state or list of states to process.
		    n_samples_generation (int): Number of generations per state.
		    temperature (float): Temperature for generation.
		    max_tokens (int): Maximum tokens per generation.
		    demos (Optional[List[dict[str, Any]]]): List of demo inputs for the controller.
		    controller_sampling_params (Optional[dict[str, Any]]): Additional sampling parameters specific to controller
		        (e.g., top_p, top_k, min_p, use_beam_search). These will be merged into the config.
		    **kwargs: Additional keyword arguments.

		Returns:
		    list[list[ControllerPrediction]]: Outer list has one entry per input state.
		        Inner list contains candidate actions (each leading to a distinct child node
		        via controlled generation). Each ControllerPrediction contains: tool,
		        chosen_values, intervention, considerations, tool_execution_error, and
		        num_occurrences.
		"""
		# TODO[P2]: Add support for multi-candidate generation.
		states = states if isinstance(states, list) else [states]

		# Separate states into forced choices and those needing LM calls
		forced_results = {}  			# dict[original_index -> result]
		lm_states_with_indices = []  	# list of (original_index, state) tuples

		# 1. Identify states with forced choices -- they need not be processed by an LLM.
		for i, state in enumerate(states):
			if forced := self.get_controller_predictions_from_forced_choices(
				state=state, n_samples_generation=n_samples_generation,
			):
				forced_results[i] = forced
			else:
				lm_states_with_indices.append((i, state))

		# 2. Generate LM predictions
		lm_results = {}
		if lm_states_with_indices:
			lm_results = self._generate_lm_predictions(
				lm_states_with_indices=lm_states_with_indices,
				n_samples_generation=n_samples_generation,
				temperature=temperature,
				max_tokens=max_tokens,
				demos=demos,
				controller_sampling_params=controller_sampling_params,
			)

		# 3. Merge results for LLM generations and forced choices
		return [lm_results.get(i, forced_results.get(i)) for i in range(len(states))]
