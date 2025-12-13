# Standard library imports
import copy
import logging
from json import JSONDecodeError
from typing import Any

# Third-party imports
import dspy.utils
from dspy.adapters.utils import (
	format_field_value,
	get_annotation_name,
	parse_value,
)
from dspy.utils.callback import BaseCallback
from vllm import SamplingParams

# Local imports
from adapter.adapter_constants import FIELD_HEADER_PATTERN
from adapter.constraints import (
	ResponseLength,
	format_response_length_instruction,
)
from adapter.prompts import SIMPLE_MAIN_TEMPLATE
from adapter.utils import format_field_description
from lm.generative_local_lm import ChatCompletionResponse, GenerativeLocalVLLM
from lm.lm_constants import MessageKey, MessageRole, SamplingParam
from signatures import ReasoningSignature
from tree.tree_constants import ReasoningState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LocalVLLMAdapter:
	def __init__(self, callbacks: list[BaseCallback] | None = None):
		self.callbacks = callbacks or []

	def __call__(
		self,
		signature: type[ReasoningSignature],
		lm: GenerativeLocalVLLM,
		inputs: dict[str, Any] | list[dict[str, Any]],
		lm_kwargs: dict[str, Any] | list[dict[str, Any]],
		demos: list[dict[str, Any]] | list[list[dict[str, Any]]] | None = None,
		response_length: ResponseLength | None = None,
		verbose: bool = True,
	) -> list[list[dict[str, Any]]]:
		"""
		Produces a list of LLM outputs given a collection of formatted inputs.

		Args:
		    lm: The GenerativeLocalVLLM instance to use for generating responses.
		    lm_kwargs: Additional keyword arguments to pass to the GenerativeLocalVLLM's forward method.
		        Can be a single dict (applied to all inputs) or a list of dicts (one per input).
		    signature: The DSPy signature defining the input and output fields of the task.
			demos: Either a list of few-shot examples (demos) or a list of lists of demos (i.e., a batch of
				few-shot examples -- one per input). Each demo should be a dictionary containing ReasoningState.INPUT and
				ReasoningState.OUTPUT keys, where ReasoningState.INPUT is a dictionary of input fields (will be represented with user
				messages) and ReasoningState.OUTPUT is a dictionary of output fields (will be represented with assistant
				messages).
		    inputs: Either a single input dictionary or a list of input dictionaries, where each dictionary
				contains the input fields defined in the signature. If a list of dictionaries is provided,
				the length of this list must match the length of `demos` if `demos` is a list of lists (this
				corresponds with the "batch dimension" of the inputs).
		    response_length: Optional ResponseLength object specifying constraints on the length of the response.
		    verbose: Boolean indicating whether to log verbose outputs during execution.

		Returns:
			A list of lists of dictionaries, where the outer list corresponds to the inputs (i.e., has length
			equal to the number of inputs), and each inner list corresponds to the completion attempts for
			each the respective input. Each inner list may contain multiple completion attempts if `n` is specified
			in `lm_kwargs` (i.e., if `n > 1` in `vllm.SamplingParams(...)`).
		"""
		if demos is None:
			demos = []
		logger.setLevel(logging.DEBUG if verbose else logging.INFO)
		messages = self.format(
			signature=signature,
			demos=demos,
			inputs=inputs,
			response_length=response_length,
		)

		# Determine batch size
		batch_size = len(messages)

		# Handle lm_kwargs validation and processing
		if isinstance(lm_kwargs, list):
			if not isinstance(inputs, list):
				raise ValueError(
					"When lm_kwargs is a list, inputs must also be a list."
				)
			if len(lm_kwargs) != batch_size:
				raise ValueError(
					f"When lm_kwargs is a list, it must have the same length as inputs. "
					f"Got {len(lm_kwargs)} lm_kwargs but {batch_size} inputs."
				)

			# Batch mode with per-input kwargs - call LM separately for each input
			all_outputs: list[ChatCompletionResponse] = []
			for _i, (single_messages, single_lm_kwargs) in enumerate(
				zip(messages, lm_kwargs, strict=False)
			):
				sampling_params, _ = self._get_sampling_params(
					single_lm_kwargs
				)
				# Call forward() with a single conversation (not wrapped in list)
				output: ChatCompletionResponse = lm.forward(
					messages=single_messages,
					sampling_params=sampling_params,
					**single_lm_kwargs,
				)
				all_outputs.append(output)

			return self._call_post_process(all_outputs, signature)
		else:
			# Single lm_kwargs - create one SamplingParams for all inputs
			sampling_params = self._prepare_sampling_params(lm_kwargs)

			# Use batch() for multiple conversations, forward() for single
			if batch_size == 1:
				output = lm.forward(
					messages=messages[0],
					sampling_params=sampling_params,
					**lm_kwargs,
				)
				outputs: list[ChatCompletionResponse] = [output]
			else:
				outputs = lm.batch(
					messages=messages,
					sampling_params=sampling_params,
					**lm_kwargs,
				)

			assert all(isinstance(output, ChatCompletionResponse) for output in outputs), (
				"All outputs must be ChatCompletionResponse objects, not RerankResponse objects"
			)
			return self._call_post_process(outputs, signature)

	def _call_post_process(
		self,
		outputs: list[ChatCompletionResponse],  # Size: [batch]
		signature: type[ReasoningSignature],
	) -> list[list[dict[str, Any]]]:  			# Size: [batch, num_completions]
		"""
		Parse each completion attempt for one or more outputs.

		Args:
			outputs: A list of ChatCompletionResponse objects, where each represents a generated output,
				parsed according to the provided signature. Each output may contain multiple
				completion attempts if `n` is specified in `lm_kwargs` and is greater than 1.
			signature: The DSPy signature defining the task's inputs, outputs, and instructions.

		Returns:
			A list of list of dictionaries. Entries in the outer list represent the outputs corresponding
			to each input (i.e., the original batch size, which is equal in length to `len(outputs)`).
			Entries in the inner list represent the completion attempts for each output (when `n > 1`
			within `lm_kwargs` -- this is translated to `n > 1` in `vllm.SamplingParams(...)`).
		"""
		result = []
		for example in outputs:
			values = []
			for candidate in example["choices"]:
				candidate_logprobs = (
					candidate["logprobs"] if "logprobs" in candidate else None
				)
				candidate_text = candidate["text"]
				value: dict[str, Any] = self.parse(signature, candidate_text)
				if candidate_logprobs is not None:
					value["logprobs"] = candidate_logprobs
				values.append(value)
			result.append(values)
		return result

	def format_single(
		self,
		signature: type[ReasoningSignature],
		inputs: dict[str, Any],
		demos: list[dict[str, Any]] | None = None,
		response_length: ResponseLength | None = None,
	) -> list[dict[str, Any]]:
		"""Format a single input into messages for the LM call.

		Args:
		    signature: The DSPy signature defining the task's inputs, outputs, and instructions.
		    demos: A list of few-shot examples.
		    inputs: The input arguments for a single instance.
		    response_length: Optional response length constraints.

		Returns:
		    A list of messages for a single conversation.
		"""
		if demos is None:
			demos = []
		field_descriptions = format_field_description(signature)
		response_length_instruction = format_response_length_instruction(
			response_length=response_length
		)
		# Format instruction as a separate line only if it has content
		response_length_instruction_formatted = (
			f"\n\n{response_length_instruction}" if response_length_instruction else ""
		)
		system_message = SIMPLE_MAIN_TEMPLATE.format(
			task_instructions=signature.instructions,
			field_descriptions=field_descriptions,
			response_length_instruction=response_length_instruction,
			response_length_instruction_formatted=response_length_instruction_formatted,
		).strip()
		messages: list[dict[str, Any]] = []
		messages.append({
			MessageKey.ROLE: MessageRole.SYSTEM,
			MessageKey.CONTENT: system_message,
		})
		messages.extend(self.format_demos(signature=signature, demos=demos))
		content = self.format_user_message_content(
			signature=signature, inputs=inputs, main_request=True
		)
		messages.append({
			MessageKey.ROLE: MessageRole.USER,
			MessageKey.CONTENT: content,
		})
		return messages

	def format(
		self,
		signature: type[ReasoningSignature],
		inputs: dict[str, Any] | list[dict[str, Any]],
		demos: list[dict[str, Any]] | list[list[dict[str, Any]]],
		response_length: ResponseLength | None = None,
	) -> list[list[dict[str, Any]]]:
		"""Format the input messages for the (local, VLLM-based) LM call.

		This method converts the DSPy structured input along with few-shot examples into multiturn
		messages as expected by the LM. Supports both individual and batch processing.

		Messages will have the following structure:
		```
		[
		    {"role": "system", "content": system_message},
		    # Begin few-shot examples
		    {"role": "user", "content": few_shot_example_1_input},
		    {"role": "assistant", "content": few_shot_example_1_output},
		    {"role": "user", "content": few_shot_example_2_input},
		    {"role": "assistant", "content": few_shot_example_2_output},
		    ...
		    # End few-shot examples
		    {"role": "user", "content": current_input},
		]
		```

		The system message should contain a description of the task (i.e., description of input and
		output fields), how to derive a solution for the task, rules to follow, and a template
		for how to structure the response.

		Args:
		    signature: The DSPy signature defining the task's inputs, outputs, and instructions.
		    inputs: The input arguments to the DSPy module.
			demos: A list of few-shot examples.
		    response_length: The response length constraints
		Returns:
		    A list of multiturn message lists, one for each input.
		"""
		# Normalize inputs to list format
		if isinstance(inputs, dict):
			# If inputs is a single dictionary, convert it to a list with one element
			inputs = [inputs]
		elif not isinstance(inputs, list):
			raise TypeError(
				f"Expected inputs to be a list or a dict, got {type(inputs)}"
			)

		# Handle demos - check if it's batch demos or single demos for all
		# TODO[P3]: Add support for custom demos for different inputs/trajectories for a single input
		# NOTE: We assume that we get one list of demos for all inputs of the batch
		batch_demos: list[list[dict[str, Any]]]

		if demos and len(demos) > 0 and isinstance(demos[0], list):
			# Batch demos mode - each input gets its own demo list
			batch_demos = demos
			if len(batch_demos) != len(inputs):
				raise ValueError(
					f"When demos is a list of lists, it must have the same length as inputs. "
					f"Got {len(batch_demos)} demo lists but {len(inputs)} inputs."
				)
		else:
			# Single demos mode - same demos applied to all inputs
			single_demos: list[dict[str, Any]] = demos
			batch_demos = [single_demos] * len(inputs)

		messages: list[list[dict[str, Any]]] = []
		inputs_copy = copy.deepcopy(inputs)

		for i, input_dict in enumerate(inputs_copy):
			single_messages = self.format_single(
				signature=signature,
				demos=batch_demos[i],
				inputs=input_dict,
				response_length=response_length,
			)
			messages.append(single_messages)
		return messages

	def format_demos(
		self,
		signature: type[ReasoningSignature],
		demos: list[dict[str, Any]],
	) -> list[dict[str, str]]:
		"""Format the in-context examples into a list of messages.

		Transforms each demo into a pair of user and assistant messages, where the
		user message contains the inputs and the assistant message contains the
		outputs.

		Args:
		    signature: The DSPy signature for which to format the few-shot examples.
		    demos: A list of examples. Each example is a dictionary containing:
		        - INPUT: A dictionary mapping input field names to their values
		        - OUTPUT: A dictionary mapping output field names to their values

		Returns:
		    A list of messages alternating between user and assistant roles.
		"""
		# Validate that all demos are complete
		for i, demo in enumerate(demos):
			# Check that demo has the required keys
			assert set(demo.keys()).issuperset({ReasoningState.INPUT, ReasoningState.OUTPUT}), (
				f"Demo {i} is missing one or more required keys ('input', 'output')"
			)

			# Check that input dictionary contains all required input fields
			input_fields_set = set(signature.input_fields.keys())
			demo_input_fields_set = set(demo[ReasoningState.INPUT].keys())
			assert demo_input_fields_set.issuperset(input_fields_set), (
				f"Demo {i} input is missing required fields: {input_fields_set - demo_input_fields_set}"
			)

			# Check that output dictionary contains all required output fields
			output_fields_set = set(signature.output_fields.keys())
			demo_output_fields_set = set(demo[ReasoningState.OUTPUT].keys())
			assert demo_output_fields_set.issuperset(output_fields_set), (
				f"Demo {i} output is missing required fields: {output_fields_set - demo_output_fields_set}"
			)

		messages = []
		# Format each demo into user and assistant messages
		for demo in demos:
			# Create user message with input fields
			messages.append(
				{
					MessageKey.ROLE: MessageRole.USER,
					MessageKey.CONTENT: self.format_user_message_content(
						signature,
						inputs=demo[ReasoningState.INPUT],
						main_request=False,
						prefix=signature.instructions,
					),
				}
			)
			# Create assistant message with outputs
			messages.append(
				{
					MessageKey.ROLE: MessageRole.ASSISTANT,
					MessageKey.CONTENT: self.format_demo_assistant_message(
						signature=signature, demo=demo
					),
				}
			)
		return messages

	def format_demo_assistant_message(
		self,
		signature: type[ReasoningSignature],
		demo: dict[str, Any],
	) -> str:
		"""Format an assistant message for an in-context example.

		Creates a properly formatted assistant message with output fields using ## headers.

		Args:
		    signature: The DSPy signature defining the fields.
		    demo: A dictionary containing input and output keys.

		Returns:
		    A formatted string representing the assistant's response with proper headers.
		"""
		message_parts = []

		# Add each output field
		for field_name, field_info in signature.output_fields.items():
			message_parts.append(f"## {field_name}")
			# Format the output value according to its field info
			formatted_value = format_field_value(field_info, demo[ReasoningState.OUTPUT][field_name])
			message_parts.append(formatted_value)

		# Join all parts with double newlines for readability
		return "\n\n".join(message_parts)

	def format_user_message_content(
		self,
		signature: type[ReasoningSignature],
		inputs: dict[str, Any],
		prefix: str = "",
		suffix: str = "",
		main_request: bool = False,
	) -> str:
		"""
		Format the content of the user message.

		The user prompt instructs the language model (assistant) to solve a single instance of the task
		defined in the system message.

		Args:
		    signature (Type[Signature]): The DSPy signature defining the expected input/output fields.
		    inputs (Dict[str, Any]): The input arguments to the DSPy module.
		    prefix (str): Optional prefix to prepend to the user message.
		    suffix (str): Optional suffix to append to the user message.
		    main_request (bool): Whether this is the main request for the task. If True, it will include
		        output requirements in the message.

		Returns:
		    str: The formatted user message content.
		"""
		messages = [prefix if prefix else signature.instructions]
		for k, v in signature.input_fields.items():
			if k in inputs:
				value = inputs.get(k)
				formatted_field_value = format_field_value(field_info=v, value=value)
				messages.append(f"## {k}\n{formatted_field_value}")
		if main_request:
			output_requirements = self.user_message_output_requirements(signature)
			if output_requirements is not None:
				messages.append(output_requirements)
		messages.append(suffix)
		return "\n\n".join(messages).strip()

	def user_message_output_requirements(
		self, signature: type[ReasoningSignature]
	) -> str:
		"""Returns a simplified format reminder for the language model.

		In chat-based interactions, language models may lose track of the required output format
		as the conversation context grows longer. This method generates a concise reminder of
		the expected output structure that can be included in user messages.

		Args:
		    signature (Type[Signature]): The DSPy signature defining the expected input/output fields.

		Returns:
		    str: A simplified description of the required output format.

		Note:
		    This is a more lightweight version of `format_field_structure` specifically designed
		    for inline reminders within chat messages.
		"""

		def type_info(v: Any) -> str:
			if v.annotation is not str:
				return f" (must be formatted as a valid Python {get_annotation_name(v.annotation)})"
			else:
				return ""

		message = (
			"Respond with the corresponding output fields, starting with the field "
		)
		message += ", then ".join(
			f"`## {f}`{type_info(v)}" for f, v in signature.output_fields.items()
		)
		return message

	def _call_post_process_batch(
		self,
		outputs: list[dict[str, Any]],  # Size: [batch]
		signature: type[ReasoningSignature],
	) -> list[list[dict[str, Any]]]:  # Size: [batch, num_completions]
		"""
		Parse each completion attempt for one or more outputs.

		Args:
		    outputs: A list of dictionaries, where each dictionary represents a generated output, parsed
		    according to the provided signature. Each output dictionary may contain multiple
		    completion attempts if `n` is specified in `lm_kwargs` and is greater than 1.
		    signature: The DSPy signature defining the task's inputs, outputs, and instructions.

		Returns:
		A list of list of dictionaries. Entries in the outer list represent the outputs corresponding
		to each input (i.e., the original batch size, which is equal in length to `len(outputs)`).
		Entries in the inner list represent the completion attempts for each output (when `n > 1`
		within `lm_kwargs` -- this is translated to `n > 1` in `vllm.SamplingParams).
		"""
		result = []
		for example in outputs:
			values = []
			for candidate in example["choices"]:
				candidate = candidate["text"]
				candidate_logprobs = (
					candidate["logprobs"] if "logprobs" in candidate else None
				)
				value = self.parse(signature, candidate)
				if candidate_logprobs is not None:
					value["logprobs"] = candidate_logprobs
				values.append(value)
			result.append(values)
		return result

	def _get_sampling_params(
		self,
		kwargs: dict[str, Any],
		temperature: float = 0.0,
		max_tokens: int = 1000,
	) -> tuple[SamplingParams, dict[str, Any]]:
		"""
		Extract sampling parameters from kwargs and return a SamplingParams object.

		Args:
		    kwargs: A dictionary of keyword arguments that may contain sampling parameters.
		    temperature: The temperature to use for sampling. Defaults to 0.0.
		    max_tokens: The maximum number of tokens to generate. Defaults to 1000.

		Returns:
		    A tuple containing:
		        - SamplingParams object initialized with the provided or default parameters.
		        - Remaining kwargs that are not sampling parameters.
		"""

		sampling_fields = set(SamplingParams.__annotations__)
		sampling_params = {}  # The parameters used to initialize vllm.SamplingParams
		remaining_kwargs = kwargs.copy()

		for parameter_name in list(remaining_kwargs.keys()):
			if parameter_name in sampling_fields:
				parameter_value = remaining_kwargs.pop(
					parameter_name
				)  # Sampling parameters should not be passed into other methods
				if parameter_name == SamplingParam.TEMPERATURE:
					sampling_params[parameter_name] = parameter_value
				elif parameter_name == SamplingParam.MAX_TOKENS:
					sampling_params[parameter_name] = parameter_value
				else:
					sampling_params[parameter_name] = parameter_value

		# Set defaults if not provided
		if SamplingParam.TEMPERATURE not in sampling_params:
			sampling_params[SamplingParam.TEMPERATURE] = temperature
		if SamplingParam.MAX_TOKENS not in sampling_params:
			sampling_params[SamplingParam.MAX_TOKENS] = max_tokens
		# Default include_stop_str_in_output to True unless explicitly set
		if "include_stop_str_in_output" not in sampling_params:
			sampling_params["include_stop_str_in_output"] = True

		return (
			SamplingParams(**sampling_params),
			remaining_kwargs,  # Remaining kwargs that are not sampling parameters
		)

	def _prepare_sampling_params(
		self, lm_kwargs: dict[str, Any] | list[dict[str, Any]]
	) -> SamplingParams | list[SamplingParams]:
		"""
		Prepare sampling parameters for either single or batch processing.

		Args:
		    lm_kwargs: Either a single dict or list of dicts containing LM parameters.

		Returns:
		    Either a single SamplingParams object or a list of SamplingParams objects.
		"""

		if isinstance(lm_kwargs, list):
			# Batch mode - create SamplingParams for each input
			sampling_params_list = []
			for kwargs in lm_kwargs:
				sampling_params, _ = self._get_sampling_params(kwargs)
				sampling_params_list.append(sampling_params)
			return sampling_params_list
		else:
			# Single mode - create one SamplingParams for all inputs
			sampling_params, _ = self._get_sampling_params(lm_kwargs)
			return sampling_params

	def parse(
		self, signature: type[ReasoningSignature], completion: str
	) -> dict[str, Any]:
		"""
		Parses the provided completion and extracts the relevant outputs.

		Args:
		    signature: The DSPy signature for which to parse the completion.
		    completion: The completion to parse.

		Returns:
		    A dictionary containing the parsed output fields.
		"""
		# Try JSON parsing first (for guided JSON generation)
		completion_stripped = completion.strip()
		if completion_stripped.startswith("{") and completion_stripped.endswith("}"):
			try:
				import json

				json_response: dict[str, Any] = json.loads(completion_stripped)

				# Check if all required fields are present
				expected_fields = set(signature.output_fields.keys())
				if set(json_response.keys()).issuperset(expected_fields):
					return json_response
			except JSONDecodeError:
				pass  # Fall through to original parsing

		sections: list[tuple[str | None, list[str]]] = [(None, [])]
		for line in completion.splitlines():
			match = FIELD_HEADER_PATTERN.match(line.strip())
			if match:
				# If the header pattern is found, split the rest of the line as content
				header = match.group(1)
				remaining_content = line[match.end() :].strip()
				sections.append(
					(header, [remaining_content] if remaining_content else [])
				)
			else:
				sections[-1][1].append(line)

		processed_sections = [(k, "\n".join(v).strip()) for k, v in sections]

		fields: dict[str, Any] = {}
		for k, v in processed_sections:
			if (
				k is not None
				and (k not in fields)
				and (k in signature.output_fields.keys())
			):
				try:
					annotation = signature.output_fields[k].annotation
					fields[k] = parse_value(v, annotation)
				except Exception as e:
					raise dspy.utils.exceptions.AdapterParseError(
						adapter_name="SimpleLocalVLLMAdapter",
						signature=signature,
						lm_response=completion,
						message=f"Failed to parse field {k} with value {v} from the LM response. Error message: {e}",
					) from e
		if fields.keys() != signature.output_fields.keys():
			raise dspy.utils.exceptions.AdapterParseError(
				adapter_name="SimpleLocalVLLMAdapter",
				signature=signature,
				lm_response=completion,
				parsed_result=fields,
			)
		return fields
