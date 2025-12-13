# A DSPy adapter for scoring/reranking tasks using LocalVLLM.
# Standard library imports
import logging
from typing import Literal

# Third-party imports
from dspy.utils.callback import BaseCallback

# Local imports
from adapter.adapter_constants import ScoringTarget
from constants import VERBOSITY_TO_LOGGING_LEVEL, Verbosity
from lm.lm_constants import MessageRole, TaskType
from lm.scoring_local_lm import RerankResponse, ScoringLocalVLLM
from signatures import ReasoningSignature
from tree import Input, Output, Reasoning

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)



class LocalVLLMScoringAdapter:
	"""
	Adapter for scoring/reranking tasks using LocalVLLM.

	This adapter is specifically configured for scoring tasks where the
	underlying model is initialized with task="score" for efficient
	query-document pair scoring. It provides methods for formatting inputs,
	executing batch scoring operations, and parsing results.
	"""

	def __init__(
		self,
		callbacks: list[BaseCallback] | None = None,
		message_start_token: str = "<|im_start|>",  # noqa: S107
		message_end_token: str = "<|im_end|>",      # noqa: S107
		assistant_prefix: str = "<think>\n\n</think>\n\n",
		verbosity: Verbosity = Verbosity.INFO,
	) -> None:
		"""
		Initialize the LocalVLLMScoringAdapter.

		Args:
			callbacks: Optional list of callbacks to execute during scoring
				operations. Callbacks should implement BaseCallback interface
				from DSPy.
			message_start_token: Token used to start a message block (e.g., "<|im_start|>").
				Defaults to Qwen3 format.
			message_end_token: Token used to end a message block (e.g., "<|im_end|>").
				Defaults to Qwen3 format.
			assistant_prefix: Prefix text for assistant messages in the suffix.
				Defaults to redacted reasoning format.
			verbosity: Verbosity level for logging. Defaults to INFO.
		"""
		self.callbacks: list[BaseCallback] = callbacks or []
		self._message_start_token: str = message_start_token
		self._message_end_token: str = message_end_token
		self._assistant_prefix: str = assistant_prefix
		self._verbosity: Verbosity = verbosity

	@property
	def assistant_prefix(self) -> str:
		return self._assistant_prefix

	@assistant_prefix.setter
	def assistant_prefix(self, value: str) -> None:
		self._assistant_prefix = value

	@property
	def message_start_token(self) -> str:
		return self._message_start_token

	@message_start_token.setter
	def message_start_token(self, value: str) -> None:
		self._message_start_token = value

	@property
	def message_end_token(self) -> str:
		return self._message_end_token

	@message_end_token.setter
	def message_end_token(self, value: str) -> None:
		self._message_end_token = value

	@property
	def verbosity(self) -> Verbosity:
		"""Get the current verbosity level."""
		return self._verbosity

	@verbosity.setter
	def verbosity(self, verbosity: Verbosity) -> None:
		"""Set the verbosity level and update logger."""
		self._verbosity = verbosity
		logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL[verbosity])

	def __call__(
		self,
		signature: type[ReasoningSignature],
		lm: ScoringLocalVLLM,
		input: Input,
		scoring_target: Literal[
			ScoringTarget.REASONING, ScoringTarget.OUTPUT, ScoringTarget.ACTION
		],
        reasoning_candidates: Reasoning | list[Reasoning] | None = None,
		output_candidates: Output | list[Output] | None = None,
        action_candidates: str | list[str] | None = None,
	) -> list[RerankResponse]:
		"""
		Score query-document pairs using the LocalVLLM model.

		This method normalizes inputs to lists and calls the underlying
		LocalVLLM's score method. A single query string or list of queries is
		scored against a single document string or list of documents.

		Args:
			signature: The DSPy signature defining the scoring task, specifying
				input/output field definitions for the scoring operation.
			lm: The LocalVLLM instance with task="score" to use for scoring.
				Must be initialized with TaskType.SCORE constant.
			input: The input for the reasoning task (e.g., a topic and a stance for an argument
                generation task, or a question for a question-answering task). The inputs
                provided must be specified as input fields in the provided signature.
			scoring_target: Literal specifying what the scoring task is:
				- REASONING: Evaluate reasoning trajectories or partial solutions
				- OUTPUT: Evaluate final output fields (complete solutions)
				- ACTION: Evaluate controller actions
			reasoning_candidates: Either a single reasoning candidate or a list of reasoning
                candidates. Will be normalized to list[Reasoning] for processing.
			output_candidates: Either a single output candidate or a list of output candidates.
				Will be normalized to list[Output] for processing.
			action_candidates: Either a single action candidate or list of action candidates.
				Will be normalized to list[str] for processing.

		Returns:
			A list of RerankResponse objects, one per query. Each
			RerankResponse contains scoring results for that query against all
			documents, including relevance scores and metadata.

		Raises:
			AssertionError: If scoring_target is invalid, or if the
				LocalVLLM instance is not configured for scoring.
		"""
		# Set logger level based on verbosity
		logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL[self.verbosity])

		# Validate target field name
		assert scoring_target in {ScoringTarget.REASONING, ScoringTarget.OUTPUT, ScoringTarget.ACTION}, (
		    "scoring_target must be one of 'reasoning', 'output', or 'action', "
		    f"but received '{scoring_target}'."
		)

		# Normalize inputs to always be lists
		# NOTE: Reasoning and Output are type aliases (dict types)
		norm_reasoning_candidates = (
			[reasoning_candidates]
			if reasoning_candidates is not None
			and isinstance(reasoning_candidates, dict)
			and not isinstance(reasoning_candidates, list)
			else reasoning_candidates
		)
		norm_output_candidates = (
			[output_candidates]
			if output_candidates is not None
			and isinstance(output_candidates, dict)
			and not isinstance(output_candidates, list)
			else output_candidates
		)
		norm_action_candidates = (
			[action_candidates] if isinstance(action_candidates, str)
			else action_candidates
		)

		# Validate that the LM is configured for scoring
		assert lm.task == TaskType.SCORE, (
			f"LocalVLLM instance must be initialized with task='score', but got task='{lm.task}'."
		)

		# Call the scoring method
		results = self._score_batch(
            signature=signature,
            lm=lm,
            input=input,
            scoring_target=scoring_target,
            reasoning_candidates=norm_reasoning_candidates,
            output_candidates=norm_output_candidates,
            action_candidates=norm_action_candidates,
        )
		return results

	def _score_batch(
		self,
		signature: type[ReasoningSignature],
		lm: ScoringLocalVLLM,
        input: Input,
		scoring_target: Literal[
			ScoringTarget.REASONING, ScoringTarget.OUTPUT, ScoringTarget.ACTION
		],
		reasoning_candidates: list[Reasoning] | None = None,
		output_candidates: list[Output] | None = None,
        action_candidates: list[str] | None = None,
	) -> list[RerankResponse]:
		"""
		Score queries against documents using the ScoringLocalVLLM instance.

		Formats queries and documents according to the signature, then calls
		the underlying ScoringLocalVLLM's score method, returning one RerankResponse
		per query.

		Args:
			signature: The DSPy signature for the scoring task. Used to format
				queries and documents before scoring.
			lm: The LocalVLLM instance with task="score" initialized for
				scoring operations.
			input: The input for the reasoning task (e.g., a topic and a stance
				for an argument generation task, or a question for a question-
				answering task). The inputs provided must be specified as input
				fields in the provided signature.
			scoring_target: What the scoring task is:
				- ScoringTarget.REASONING: Evaluate reasoning trajectories or partial solutions
				- ScoringTarget.OUTPUT: Evaluate final output fields (complete solutions)
				- ScoringTarget.ACTION: Evaluate controller actions
			reasoning_candidates: Optional list of reasoning candidates to score.
			output_candidates: Optional list of output candidates to score.
			action_candidates: Optional list of action candidates to score.

		Returns:
			A list of RerankResponse objects, one per query.
			Each RerankResponse contains the scoring results for that query against all documents.
		"""
		if scoring_target == ScoringTarget.REASONING:
			assert reasoning_candidates is not None, (
				"reasoning_candidates must be provided when scoring reasoning candidates."
			)
		elif scoring_target == ScoringTarget.OUTPUT:
			assert output_candidates is not None, (
				"output_candidates must be provided when scoring output candidates."
			)
		else: 	# scoring_target == ScoringTarget.ACTION:
			assert action_candidates is not None, (
				"action_candidates must be provided when scoring action candidates."
			)

		# Format queries and documents according to the signature
		formatted_queries = self.format_queries(
            signature=signature,
            input=input,
            scoring_target=scoring_target,
            reasoning_candidates=reasoning_candidates,
        )
		formatted_documents = self.format_documents(
            scoring_target=scoring_target,
            reasoning_candidates=reasoning_candidates,
            output_candidates=output_candidates,
            action_candidates=action_candidates,
        )

		# Log formatted queries and documents in DEBUG mode
		logger.debug(f"\n{'='*80}\nFORMATTED QUERIES ({len(formatted_queries)} total):\n{'='*80}")
		for i, query in enumerate(formatted_queries, 1):
			logger.debug(f"\n--- Query {i}/{len(formatted_queries)} ---\n{query}\n")
		logger.debug(f"\n{'='*80}\nFORMATTED DOCUMENTS ({len(formatted_documents)} total):\n{'='*80}")
		for i, doc in enumerate(formatted_documents, 1):
			logger.debug(f"\n--- Document {i}/{len(formatted_documents)} ---\n{doc}\n")
		logger.debug(f"{'='*80}\n")

		# Validate query-document matching based on scoring target
		num_queries = len(formatted_queries)
		num_documents = len(formatted_documents)

		# Score each query against all documents using broadcasting mode.
		# This mode gives us N responses (one per query), each with M scores (one per
		# document/action)
		if scoring_target == ScoringTarget.ACTION:
			# For ScoringTarget.ACTION: Expand N queries × M documents into N×M pairs where each query
			# is repeated M times (once per document). Then use broadcast_scores=True.
			# The score method groups scores by unique query, giving us N responses with M scores each.
			expanded_queries = []
			expanded_documents = []
			for query in formatted_queries:
				# Repeat this query M times (once for each document)
				for document in formatted_documents:
					expanded_queries.append(query)
					expanded_documents.append(document)

			# Score with broadcast_scores=True
			# The score method automatically groups scores by unique query
			return lm.score(
				queries=expanded_queries,
				documents=expanded_documents,
				use_tqdm=False,
				broadcast_scores=True,
			)

		# For ScoringTarget.REASONING and ScoringTarget.OUTPUT: Each query must have a corresponding document (1-to-1 matching)
		assert num_queries == num_documents, (
			f"For {scoring_target} scoring, the number of queries ({num_queries}) "
			f"must match the number of documents ({num_documents}). "
			f"Each query should be scored against its corresponding document."
		)

		# For ScoringTarget.REASONING/ScoringTarget.OUTPUT: Use pairwise mode (one response per query-document pair)
		rerank_responses: list[RerankResponse] = lm.score(
			queries=formatted_queries,
			documents=formatted_documents,
			use_tqdm=False,
			broadcast_scores=False,
		)

		return rerank_responses

	def _get_task_instruction(
		self,
		signature: type[ReasoningSignature],
		scoring_target: Literal[ScoringTarget.REASONING, ScoringTarget.OUTPUT, ScoringTarget.ACTION],
	) -> str:
		"""
		Generate the detailed task instruction for the scoring operation.

		Args:
			signature: The DSPy signature to extract information from.
			scoring_target: What the scoring task is:
				- ScoringTarget.REASONING: Evaluate reasoning trajectories or partial solutions
				- ScoringTarget.OUTPUT: Evaluate final output fields (complete solutions)
				- ScoringTarget.ACTION: Evaluate controller actions

		Returns:
			A string containing the detailed task instructions.
		"""
		if scoring_target == ScoringTarget.REASONING:
			label_description = "partial solution (reasoning) given the provided input"
		elif scoring_target == ScoringTarget.OUTPUT:
			label_description = "final output given the provided input and reasoning"
		else:	# scoring_target == ScoringTarget.ACTION
			label_description = "action to take given the provided input and partial solution (reasoning)"

		return (
			f"{signature.instructions.strip()} "
			f"Since this is a reasoning task, we are interested not only in the final output, "
			f"but also in the reasoning process that leads to it. "
			f"The user will provide you with inputs (under the \"# Inputs\" heading) and "
			f"{scoring_target.capitalize()} (under the \"# {scoring_target.capitalize()}\" heading). "
			f"Determine whether the candidate {scoring_target.capitalize()} is an effective "
			f"{label_description}."
		)

	def _get_user_message_query(
		self,
		input: Input,
		scoring_target: Literal[ScoringTarget.REASONING, ScoringTarget.OUTPUT, ScoringTarget.ACTION],
		reasoning_candidates: list[Reasoning] | None = None,
		candidate_index: int = 0,
	) -> str:
		"""
		Generate the query content for the user message based on the scoring target.

		* For ScoringTarget.REASONING target: Includes input fields + all but the most recent reasoning step (if
			there are multiple reasoning steps). If there is only one reasoning step in the
			reasoning trajectory, the user message will only include the input fields.
		* For ScoringTarget.OUTPUT target: Includes input fields + the entire reasoning trajectory.
		* For ScoringTarget.ACTION target: Includes input fields + the entire reasoning trajectory.

		Args:
			input: The input dictionary containing task-specific input fields.
			scoring_target: What the scoring task is:
				- ScoringTarget.REASONING: Evaluate reasoning trajectories or partial solutions
				- ScoringTarget.OUTPUT: Evaluate final output fields (complete solutions)
				- ScoringTarget.ACTION: Evaluate controller actions
			reasoning_candidates: Optional list of reasoning candidates.
			candidate_index: Index of the candidate to include.

		Returns:
			A formatted query string with appropriate sections.
		"""
		# Format input fields matching the document format (field: value)
		input_lines = [f"{k}: {v}" for k, v in input.items()]
		input_str = "\n".join(input_lines)

		message_parts = [f"# Inputs\n{input_str}"]

		# For ScoringTarget.REASONING, include all but most recent reasoning step (if multiple steps exist)
		if scoring_target == ScoringTarget.REASONING:
			# Only process reasoning if candidates are provided
			if reasoning_candidates is not None:
				# Get all but most recent reasoning step
				previous_reasoning = self._get_all_but_most_recent_reasoning(
					reasoning_candidates[candidate_index]
				)
				# Only include if there are previous steps (i.e., more than one step total)
				# Check if any field has non-empty values
				has_previous_steps = any(
					len(values) > 0 for values in previous_reasoning.values()
				)
				if has_previous_steps:
					reasoning_str = self._format_reasoning_dict(previous_reasoning)
					message_parts.append(f"# Reasoning\n{reasoning_str}")
		else:
			# For ScoringTarget.OUTPUT and ScoringTarget.ACTION, reasoning candidates are required
			if reasoning_candidates is not None:
				reasoning_str = self._format_reasoning_dict(reasoning_candidates[candidate_index])
				message_parts.append(f"# Reasoning\n{reasoning_str}")

		return "\n\n".join(message_parts)

	def _get_user_message_document(
		self,
		scoring_target: Literal[ScoringTarget.REASONING, ScoringTarget.OUTPUT, ScoringTarget.ACTION],
		reasoning_candidates: list[Reasoning] | None = None,
		output_candidates: list[Output] | None = None,
		action_candidates: list[str] | None = None,
		candidate_index: int = 0,
	) -> str:
		"""
		Generate the document content for scoring.

		For ScoringTarget.REASONING target: Returns the reasoning candidate.
		For ScoringTarget.OUTPUT target: Returns the most recent reasoning step + output.
		For ScoringTarget.ACTION target: Returns the chosen action.

		Args:
			scoring_target: What the scoring task is:
				- ScoringTarget.REASONING: Evaluate reasoning trajectories or partial solutions
				- ScoringTarget.OUTPUT: Evaluate final output fields (complete solutions)
				- ScoringTarget.ACTION: Evaluate controller actions
			reasoning_candidates: Optional list of reasoning candidates.
			output_candidates: Optional list of output candidates.
			action_candidates: Optional list of action candidate strings.
			candidate_index: Index of the candidate to include.

		Returns:
			A formatted document string with the candidate content.
		"""
		result = f"# {scoring_target.capitalize()}\n"
		if scoring_target == ScoringTarget.REASONING:
			result += self._format_reasoning_dict(reasoning_candidates[candidate_index]).strip()
		elif scoring_target == ScoringTarget.OUTPUT:
			result += self._format_output_dict(output_candidates[candidate_index]).strip()
		else:  # scoring_target == ScoringTarget.ACTION
			result += action_candidates[candidate_index].strip()
		return result

	def format_queries(
		self,
		signature: type[ReasoningSignature],
		scoring_target: Literal[ScoringTarget.REASONING, ScoringTarget.OUTPUT, ScoringTarget.ACTION],
		input: Input,
		reasoning_candidates: list[Reasoning] | None = None,
	) -> list[str]:
		"""
		Format queries from input and optional reasoning trajectories.

		Queries follow the template: {prefix}<Instruct>: {instruction}\n<Query>: {query}\n

		See guidance on VLLM reranker models (in this case qwen3) here:
		https://docs.vllm.ai/en/v0.9.2/examples/offline_inference/qwen3_reranker.html

		Query content varies by scoring target:
		- ScoringTarget.REASONING: Input fields + all but most recent reasoning step (if multiple steps exist)
		- ScoringTarget.OUTPUT: Input fields + entire reasoning trajectory
		- ScoringTarget.ACTION: Input fields + entire reasoning trajectory

		Args:
			signature: The DSPy signature for the scoring task. Used to extract
				the instruction description to construct a formatted prompt.
			scoring_target: What the scoring task is:
				- ScoringTarget.REASONING: Evaluate reasoning trajectories or partial solutions
				- ScoringTarget.OUTPUT: Evaluate final output fields (complete solutions)
				- ScoringTarget.ACTION: Evaluate controller actions
			input: The input dictionary containing task-specific input fields.
			reasoning_candidates: List of reasoning candidates.
				When provided, trajectories may be extracted for context in the query.

		Returns:
			A list of formatted query strings meant for the `score` method of reranker models.
		"""
		# Fixed prefix for the system message and user start
		prefix = (
			f"{self.message_start_token}{MessageRole.SYSTEM}\n"
			"Judge whether the Document meets the requirements based on the Query and the "
			f"Instruct provided. Note that the answer can only be \"yes\" or \"no\".{self.message_end_token}\n"
			f"{self.message_start_token}{MessageRole.USER}\n"
		)

		# Get instructions from signature
		instruction = self._get_task_instruction(signature, scoring_target)

		# Determine number of queries based on scoring target
		if scoring_target == ScoringTarget.OUTPUT:
			num_queries = len(reasoning_candidates)
		elif scoring_target == ScoringTarget.ACTION:
			num_queries = len(reasoning_candidates)
		else:	# scoring_target == ScoringTarget.REASONING
			num_queries = len(reasoning_candidates) if reasoning_candidates else 1

		formatted_queries = []
		for i in range(num_queries):
			query_content = self._get_user_message_query(
				input=input,
				scoring_target=scoring_target,
				reasoning_candidates=reasoning_candidates,
				candidate_index=i,
			)

			# Format: {prefix}<Instruct>: {instruction}\n<Query>: {query}\n
			formatted_query = f"{prefix}<Instruct>: {instruction}\n<Query>: {query_content}\n"
			formatted_queries.append(formatted_query)

		return formatted_queries

	def format_documents(
		self,
		scoring_target: Literal[ScoringTarget.REASONING, ScoringTarget.OUTPUT, ScoringTarget.ACTION],
		reasoning_candidates: list[Reasoning] | None = None,
		output_candidates: list[Output] | None = None,
		action_candidates: list[str] | None = None,
	) -> list[str]:
		"""
		Format documents/candidates according to the signature.

		Determines which type of solution is being scored based on whether an
		output field is provided (complete solution) or not (working solution):
		- **Complete Solution**: output_candidates is not None. Documents include
			both the reasoning trajectory and the final output fields.
		- **Working Solution**: output_candidates is None. Documents include only
			the reasoning trajectories or actions.

		Args:
			signature: The DSPy signature for the scoring task. Used to determine
				which candidates to format and how to format them.
			scoring_target: What the scoring task is:
				- ScoringTarget.REASONING: Evaluate reasoning trajectories or partial solutions
				- ScoringTarget.OUTPUT: Evaluate final output fields (complete solutions)
				- ScoringTarget.ACTION: Evaluate controller actions
			reasoning_candidates: Optional list of reasoning candidates (working
				solutions without final outputs).
			output_candidates: Optional list of output candidates (complete
				solutions with final outputs). When provided, indicates a
				complete solution scenario.
			action_candidates: Optional list of action candidates (for controller
				action scoring).

		Returns:
			A list of formatted document strings with suffix formatting applied.
		"""
		# Suffix for the document: end of user turn, start of assistant turn + think block
		suffix = (
			f"{self.message_end_token}\n"
			f"{self.message_start_token}{MessageRole.ASSISTANT}\n"
			f"{self.assistant_prefix}"
		)

		# Determine number of documents based on candidates
		if scoring_target == ScoringTarget.REASONING:
			if output_candidates is not None:
				num_docs = len(output_candidates)
			elif reasoning_candidates is not None:
				num_docs = len(reasoning_candidates)
			else:
				num_docs = 0
		elif scoring_target == ScoringTarget.OUTPUT:
			num_docs = len(output_candidates) if output_candidates else 0
		else:  # scoring_target == ScoringTarget.ACTION
			num_docs = len(action_candidates) if action_candidates else 0

		formatted_documents = []
		for i in range(num_docs):
			doc_content = self._get_user_message_document(
				scoring_target=scoring_target,
				reasoning_candidates=reasoning_candidates,
				output_candidates=output_candidates,
				action_candidates=action_candidates,
				candidate_index=i,
			)

			# Format: <Document>: {doc}{suffix}
			formatted_doc = f"<Document>: {doc_content}{suffix}"
			formatted_documents.append(formatted_doc)

		return formatted_documents

	def _format_output_dict(self, output: Output) -> str:
		"""
		Format an output dictionary as a string for document scoring.

		Args:
			output: Dictionary mapping output field names to their values.

		Returns:
			A formatted string representation of the output.
		"""
		lines = []
		for field_name, field_value in output.items():
			lines.append(f"{field_name}: {field_value}")
		return "\n".join(lines).strip()

	def _format_reasoning_dict(self, reasoning: Reasoning) -> str:
		"""
		Format a reasoning dictionary as a string for document scoring.

		Args:
			reasoning: Dictionary mapping reasoning field names to lists of
				reasoning steps.

		Returns:
			A formatted string representation of the reasoning trajectory.
		"""
		lines = []
		for field_name, field_values in reasoning.items():
			# field_values is typically a list of reasoning steps
			if isinstance(field_values, list):
				for step in field_values:
					lines.append(f"{field_name}: {step}")
			else:
				lines.append(f"{field_name}: {field_values}")
		return "\n".join(lines).strip()

	def _get_most_recent_reasoning_step(self, reasoning: Reasoning) -> str:
		"""
		Extract the most recent reasoning step from a reasoning dictionary.

		Args:
			reasoning: Dictionary mapping reasoning field names to lists of
				reasoning steps.

		Returns:
			A formatted string representation of the most recent reasoning step.
		"""
		lines = []
		for field_name, field_values in reasoning.items():
			if isinstance(field_values, list) and len(field_values) > 0:
				# Get the last step
				most_recent = field_values[-1]
				lines.append(f"{field_name}: {most_recent}")
			elif not isinstance(field_values, list):
				lines.append(f"{field_name}: {field_values}")
		return "\n".join(lines).strip()

	def _get_all_but_most_recent_reasoning(self, reasoning: Reasoning) -> Reasoning:
		"""
		Get all reasoning steps except the most recent one.

		Args:
			reasoning: Dictionary mapping reasoning field names to lists of
				reasoning steps.

		Returns:
			A reasoning dictionary with all but the most recent step.
		"""
		result: Reasoning = {}
		for field_name, field_values in reasoning.items():
			if isinstance(field_values, list) and len(field_values) > 1:
				# Return all but the last step
				result[field_name] = field_values[:-1]
			elif isinstance(field_values, list) and len(field_values) == 1:
				# If only one step, return empty list
				result[field_name] = []
			else:
				# Not a list, return as is (though this shouldn't happen)
				result[field_name] = field_values
		return result
