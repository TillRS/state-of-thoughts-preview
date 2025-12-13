# Standard library imports
from typing import Literal
from unittest.mock import MagicMock

# Third-party imports
import pytest

# Local imports
from adapter.adapter_constants import ScoringTarget
from adapter.vllm_scoring_adapter import LocalVLLMScoringAdapter
from predict.controller_constants import ControllerContinueReasoningChoice
from signatures import (
	GenerateArgumentWithReasoning,
	ReasoningSignature,
	SolveMathProblemWithReasoning,
)
from signatures.example_signatures import ArgumentField, MathField
from tree import Input, Output, Reasoning
from utilities_for_tests import MockScoringLocalVLLM


@pytest.fixture
def scoring_adapter() -> LocalVLLMScoringAdapter:
	"""Fixture for LocalVLLMScoringAdapter."""
	return LocalVLLMScoringAdapter()


@pytest.fixture
def mock_lm() -> MockScoringLocalVLLM:
	"""Fixture for a mock LocalVLLM instance configured for scoring."""
	return MockScoringLocalVLLM(rerank_responses=None)


class TestLocalVLLMScoringAdapterInit:
	"""Test cases for LocalVLLMScoringAdapter initialization."""

	@pytest.mark.parametrize(
		# Parameter names
		"callbacks,expected_callback_count",

		# Parameter values
		[
			pytest.param(
				None,  									# callbacks
				0,  									# expected_callback_count
				id="init_default_no_callbacks",
			),
			pytest.param(
				[MagicMock()],  						# callbacks
				1,  									# expected_callback_count
				id="init_with_single_callback",
			),
			pytest.param(
				[MagicMock(), MagicMock()],  			# callbacks
				2,  									# expected_callback_count
				id="init_with_multiple_callbacks",
			),
		],
	)
	def test_init(
		self,
		callbacks: list[MagicMock] | None,
		expected_callback_count: int,
	) -> None:
		"""
		Test LocalVLLMScoringAdapter initialization with various callback configurations.

		Args:
			callbacks: Optional list of mock callbacks to initialize the adapter with.
			expected_callback_count: Expected number of callbacks in the initialized adapter.
		"""
		adapter = LocalVLLMScoringAdapter(callbacks=callbacks)
		assert len(adapter.callbacks) == expected_callback_count
		if callbacks:
			for i, callback in enumerate(callbacks):
				assert adapter.callbacks[i] == callback

class TestFormatQueries:
	"""Test cases for format_queries method."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"signature",
			"input_data",
			"scoring_target",
			"reasoning_candidates",
			"expected_queries",
			"expected_error",
		],
		# Parameter values
		[
			# REASONING target tests with SolveMathProblemWithReasoning
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				{MathField.MATH_PROBLEM: "What is 2+2?"},  	# input_data
				ScoringTarget.REASONING,  					# scoring_target
				[  											# reasoning_candidates
					{MathField.MATH_OPERATION: ["First, add 2 and 2"]}
				],
				[  											# expected_queries
					# first candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Solve the provided math problem and return its answer. Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it. The user will provide you with inputs (under the "# Inputs" heading) and Reasoning (under the "# Reasoning" heading). Determine whether the candidate Reasoning is an effective partial solution (reasoning) given the provided input.
<Query>: # Inputs
math_problem: What is 2+2?
"""
				],
					None,  									# expected_error
					id="reasoning_single_candidate_math_problem",
				),
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				{											# input_data
					MathField.MATH_PROBLEM: "What is (5*6+3)/2?"
				},
				ScoringTarget.REASONING,  					# scoring_target
				[  											# reasoning_candidates
					{										# good first step
						MathField.MATH_OPERATION: ["Multiplying 5 by 6 yields 30"]
					},
					{										# bad first step
						MathField.MATH_OPERATION: ["5 * 6 = 33"]
					},
				],
				[  											# expected_queries
					# first candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Solve the provided math problem and return its answer. Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it. The user will provide you with inputs (under the "# Inputs" heading) and Reasoning (under the "# Reasoning" heading). Determine whether the candidate Reasoning is an effective partial solution (reasoning) given the provided input.
<Query>: # Inputs
math_problem: What is (5*6+3)/2?
""",
					# second candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Solve the provided math problem and return its answer. Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it. The user will provide you with inputs (under the "# Inputs" heading) and Reasoning (under the "# Reasoning" heading). Determine whether the candidate Reasoning is an effective partial solution (reasoning) given the provided input.
<Query>: # Inputs
math_problem: What is (5*6+3)/2?
""",
				],
				None,  										# expected_error
				id="reasoning_multiple_candidates_first_step_math_problem",
			),
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				{											# input_data
					MathField.MATH_PROBLEM: "What is (5*6+3)/2?"
				},
				ScoringTarget.REASONING,  					# scoring_target
				[  											# reasoning_candidates
					{	# good candidate with 2 reasoning steps
						MathField.MATH_OPERATION: [
							"Multiplying 5 by 6 yields 30",
							"Adding 3 to 30 yields 33"
						]
					},
					{	# bad candidate with 2 reasoning steps
						MathField.MATH_OPERATION: [
							"5 * 6 = 33",
							"33 + 3 = 36"
						]
					},
				],
				[  											# expected_queries
					# first candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Solve the provided math problem and return its answer. Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it. The user will provide you with inputs (under the "# Inputs" heading) and Reasoning (under the "# Reasoning" heading). Determine whether the candidate Reasoning is an effective partial solution (reasoning) given the provided input.
<Query>: # Inputs
math_problem: What is (5*6+3)/2?

# Reasoning
math_operation: Multiplying 5 by 6 yields 30
""",
					# second candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Solve the provided math problem and return its answer. Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it. The user will provide you with inputs (under the "# Inputs" heading) and Reasoning (under the "# Reasoning" heading). Determine whether the candidate Reasoning is an effective partial solution (reasoning) given the provided input.
<Query>: # Inputs
math_problem: What is (5*6+3)/2?

# Reasoning
math_operation: 5 * 6 = 33
""",
				],
				None,  										# expected_error
				id="reasoning_multiple_candidates_second_step_math_problem",
			),
			# REASONING target tests with GenerateArgumentWithReasoning
			pytest.param(
				GenerateArgumentWithReasoning,  			# signature
				{											# input_data
					ArgumentField.TOPIC: "AI does more harm than good",
					ArgumentField.STANCE: "PRO"
				},
				ScoringTarget.REASONING,  					# scoring_target
				[  											# reasoning_candidates
					{ArgumentField.CLAIM: ["AI improves efficiency"]}
				],
				[  											# expected_queries
					# first candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Generate an argument which takes the provided stance towards the provided topic. Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it. The user will provide you with inputs (under the "# Inputs" heading) and Reasoning (under the "# Reasoning" heading). Determine whether the candidate Reasoning is an effective partial solution (reasoning) given the provided input.
<Query>: # Inputs
topic: AI does more harm than good
stance: PRO
""",
				],
				None,  										# expected_error
				id="reasoning_single_candidate_generate_argument",
			),
			pytest.param(
				GenerateArgumentWithReasoning,  			# signature
				{											# input_data
					"topic": "Combatting climate change should be a top priority",
					"stance": "ANTI"
				},
				ScoringTarget.REASONING,  					# scoring_target
				[  											# reasoning_candidates
					{ArgumentField.CLAIM: ["Climate policies are too expensive"]},
					{ArgumentField.CLAIM: ["Economic impact is too severe"]},
					{ArgumentField.CLAIM: ["Alternative solutions exist"]},
				],
				[  											# expected_queries
					# first candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Generate an argument which takes the provided stance towards the provided topic. Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it. The user will provide you with inputs (under the "# Inputs" heading) and Reasoning (under the "# Reasoning" heading). Determine whether the candidate Reasoning is an effective partial solution (reasoning) given the provided input.
<Query>: # Inputs
topic: Combatting climate change should be a top priority
stance: ANTI
""",
					# second candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Generate an argument which takes the provided stance towards the provided topic. Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it. The user will provide you with inputs (under the "# Inputs" heading) and Reasoning (under the "# Reasoning" heading). Determine whether the candidate Reasoning is an effective partial solution (reasoning) given the provided input.
<Query>: # Inputs
topic: Combatting climate change should be a top priority
stance: ANTI
""",
					# third candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Generate an argument which takes the provided stance towards the provided topic. Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it. The user will provide you with inputs (under the "# Inputs" heading) and Reasoning (under the "# Reasoning" heading). Determine whether the candidate Reasoning is an effective partial solution (reasoning) given the provided input.
<Query>: # Inputs
topic: Combatting climate change should be a top priority
stance: ANTI
""",
				],
				None,  										# expected_error
				id="reasoning_multiple_candidates_generate_argument",
			),
			# OUTPUT target tests with SolveMathProblemWithReasoning
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				{MathField.MATH_PROBLEM: "What is 10/2?"},  # input_data
				ScoringTarget.OUTPUT,  						# scoring_target
				[  											# reasoning_candidates
					{MathField.MATH_OPERATION: ["Divide 10 by 2", "10/2 = 5"]}
				],
				[  											# expected_queries
					# first candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Solve the provided math problem and return its answer. Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it. The user will provide you with inputs (under the "# Inputs" heading) and Output (under the "# Output" heading). Determine whether the candidate Output is an effective final output given the provided input and reasoning.
<Query>: # Inputs
math_problem: What is 10/2?

# Reasoning
math_operation: Divide 10 by 2
math_operation: 10/2 = 5
""",
				],
			None,  										# expected_error
			id="output_single_candidate_with_reasoning_math_problem",
		),
		pytest.param(
			SolveMathProblemWithReasoning,  			# signature
			{MathField.MATH_PROBLEM: "What is 3*4?"},  	# input_data
			ScoringTarget.OUTPUT,  						# scoring_target
			[  											# reasoning_candidates
				{MathField.MATH_OPERATION: ["3 times 4"]},
				{MathField.MATH_OPERATION: ["Multiply 3 and 4"]},
			],
			[  											# expected_queries
				# First candidate
				"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Solve the provided math problem and return its answer. Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it. The user will provide you with inputs (under the "# Inputs" heading) and Output (under the "# Output" heading). Determine whether the candidate Output is an effective final output given the provided input and reasoning.
<Query>: # Inputs
math_problem: What is 3*4?

# Reasoning
math_operation: 3 times 4
""",
				# Second candidate
				"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Solve the provided math problem and return its answer. Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it. The user will provide you with inputs (under the "# Inputs" heading) and Output (under the "# Output" heading). Determine whether the candidate Output is an effective final output given the provided input and reasoning.
<Query>: # Inputs
math_problem: What is 3*4?

# Reasoning
math_operation: Multiply 3 and 4
""",
			],
			None,  										# expected_error
			id="output_multiple_candidates_with_reasoning_math_problem",
		),
		# OUTPUT target tests with GenerateArgumentWithReasoning
		pytest.param(
			GenerateArgumentWithReasoning,  			# signature
			{											# input_data
				"topic": "Renewable Energy", "stance": "PRO"
			},
			ScoringTarget.OUTPUT,  						# scoring_target
			[  											# reasoning_candidates
				{
					"claim": [
						"Renewable energy reduces carbon emissions",
						"It creates jobs",
					]
				}
			],
			[  											# expected_queries
				"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Generate an argument which takes the provided stance towards the provided topic. Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it. The user will provide you with inputs (under the "# Inputs" heading) and Output (under the "# Output" heading). Determine whether the candidate Output is an effective final output given the provided input and reasoning.
<Query>: # Inputs
topic: Renewable Energy
stance: PRO

# Reasoning
claim: Renewable energy reduces carbon emissions
claim: It creates jobs
""",
				],
			None,  										# expected_error
			id="output_single_candidate_with_reasoning_generate_argument",
		),
		# ACTION target tests with SolveMathProblemWithReasoning
		pytest.param(
			SolveMathProblemWithReasoning,  			# signature
			{MathField.MATH_PROBLEM: "What is 7+8?"},  	# input_data
			ScoringTarget.ACTION,  						# scoring_target
			[  											# reasoning_candidates
				{MathField.MATH_OPERATION: ["Start by adding 7 and 8"]}
			],
			[  											# expected_queries
				"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Solve the provided math problem and return its answer. Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it. The user will provide you with inputs (under the "# Inputs" heading) and Action (under the "# Action" heading). Determine whether the candidate Action is an effective action to take given the provided input and partial solution (reasoning).
<Query>: # Inputs
math_problem: What is 7+8?

# Reasoning
math_operation: Start by adding 7 and 8
""",
				],
			None,  										# expected_error
			id="action_single_candidate_with_reasoning_math_problem",
		),
		pytest.param(
			GenerateArgumentWithReasoning,  			# signature
			{											# input_data
				"topic": "Space Exploration", "stance": "ANTI"
			},
			ScoringTarget.ACTION,  						# scoring_target
			[  											# reasoning_candidates
				{ArgumentField.CLAIM: ["Space exploration is expensive"]},
				{ArgumentField.CLAIM: ["Resources better spent on Earth"]},
			],
			[  											# expected_queries
				# first candidate
				"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Generate an argument which takes the provided stance towards the provided topic. Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it. The user will provide you with inputs (under the "# Inputs" heading) and Action (under the "# Action" heading). Determine whether the candidate Action is an effective action to take given the provided input and partial solution (reasoning).
<Query>: # Inputs
topic: Space Exploration
stance: ANTI

# Reasoning
claim: Space exploration is expensive
""",
				# second candidate
				"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Generate an argument which takes the provided stance towards the provided topic. Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it. The user will provide you with inputs (under the "# Inputs" heading) and Action (under the "# Action" heading). Determine whether the candidate Action is an effective action to take given the provided input and partial solution (reasoning).
<Query>: # Inputs
topic: Space Exploration
stance: ANTI

# Reasoning
claim: Resources better spent on Earth
""",
			],
			None,  										# expected_error
			id="action_multiple_candidates_with_reasoning_generate_argument",
		),
		# Edge case: No reasoning candidates for REASONING target (should still work)
		pytest.param(
			SolveMathProblemWithReasoning,  			# signature
			{MathField.MATH_PROBLEM: "What is 1+1?"},  	# input_data
			ScoringTarget.REASONING,  					# scoring_target
			None,  										# reasoning_candidates
			[  											# expected_queries
				"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Solve the provided math problem and return its answer. Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it. The user will provide you with inputs (under the "# Inputs" heading) and Reasoning (under the "# Reasoning" heading). Determine whether the candidate Reasoning is an effective partial solution (reasoning) given the provided input.
<Query>: # Inputs
math_problem: What is 1+1?
""",
			],
			None,  										# expected_error
		id="reasoning_no_candidates_defaults_to_one_query",
			),
		],
	)
	def test_format_queries_generates_valid_prompts(
		self,
		scoring_adapter: LocalVLLMScoringAdapter,
		signature: type[ReasoningSignature],
		input_data: Input,
		scoring_target: Literal[
			ScoringTarget.REASONING, ScoringTarget.OUTPUT, ScoringTarget.ACTION
		],
		reasoning_candidates: Reasoning | list[Reasoning] | None,
		expected_queries: list[str] | None,
		expected_error: type[Exception] | None,
	) -> None:
		"""
		Test that format_queries generates exact expected output strings.

		Verifies that the formatted queries exactly match the expected output
		for various combinations of signatures, scoring targets, and candidates.

		Args:
			scoring_adapter: The scoring adapter instance for testing.
			signature: The scoring signature to test with.
			input_data: Input dictionary for the task.
			scoring_target: The scoring target (reasoning/output/action).
			reasoning_candidates: Reasoning candidates (single or list).
			expected_queries: List of expected query strings.
			expected_error: Expected exception type if error should occur.
		"""
		if expected_error is not None:
			with pytest.raises(expected_error):
				scoring_adapter.format_queries(
					signature=signature,
					input=input_data,
					scoring_target=scoring_target,
					reasoning_candidates=reasoning_candidates,
				)
		else:
			formatted_queries = scoring_adapter.format_queries(
				signature=signature,
				scoring_target=scoring_target,
				input=input_data,
				reasoning_candidates=reasoning_candidates,
			)

			# Should generate expected number of queries
			assert expected_queries is not None
			assert len(formatted_queries) == len(expected_queries), (
				f"Expected {len(expected_queries)} queries, got {len(formatted_queries)}"
			)

			# Each query should exactly match the expected string
			for query_idx, (formatted_query, expected_query) in enumerate(
				zip(formatted_queries, expected_queries, strict=True)
			):
				assert formatted_query == expected_query, (
					f"Query {query_idx} does not match expected output.\n"
					f"Expected:\n{expected_query}\n\n"
					f"Got:\n{formatted_query}"
				)


class TestFormatDocuments:
	"""Test cases for format_documents method."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"signature",
			"scoring_target",
			"reasoning_candidates",
			"output_candidates",
			"action_candidates",
			"expected_documents",
			"expected_error",
		],
		# Parameter values
		[
			# REASONING target tests with SolveMathProblemWithReasoning
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				ScoringTarget.REASONING,  					# scoring_target
				[  											# reasoning_candidates
					{MathField.MATH_OPERATION: ["First, add 2 and 2"]}
				],
				None,  										# output_candidates
				None,  										# action_candidates
				[  											# expected_documents
					"""<Document>: # Reasoning
math_operation: First, add 2 and 2<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="reasoning_single_candidate_math_problem",
			),
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				ScoringTarget.REASONING,  					# scoring_target
				[  											# reasoning_candidates
					{MathField.MATH_OPERATION: ["Multiplying 5 by 6 yields 30"]},
					{MathField.MATH_OPERATION: ["5 * 6 = 33"]},
				],
				None,  										# output_candidates
				None,  										# action_candidates
				[  											# expected_documents
					# first candidate
					"""<Document>: # Reasoning
math_operation: Multiplying 5 by 6 yields 30<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
					# second candidate
					"""<Document>: # Reasoning
math_operation: 5 * 6 = 33<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="reasoning_multiple_candidates_math_problem",
			),
			# REASONING target tests with GenerateArgumentWithReasoning
			pytest.param(
				GenerateArgumentWithReasoning,  			# signature
				ScoringTarget.REASONING,  					# scoring_target
				[  											# reasoning_candidates
					{ArgumentField.CLAIM: ["AI can automate dangerous tasks"]}
				],
				None,  										# output_candidates
				None,  										# action_candidates
				[  											# expected_documents
					"""<Document>: # Reasoning
claim: AI can automate dangerous tasks<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="reasoning_single_candidate_generate_argument",
			),
			pytest.param(
				GenerateArgumentWithReasoning,  			# signature
				ScoringTarget.REASONING,  					# scoring_target
				[  											# reasoning_candidates
					{ArgumentField.CLAIM: ["Climate change requires immediate action"]},
					{ArgumentField.CLAIM: ["Renewable energy is cost-effective"]},
				],
				None,  										# output_candidates
				None,  										# action_candidates
				[  											# expected_documents
					"""<Document>: # Reasoning
claim: Climate change requires immediate action<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
					"""<Document>: # Reasoning
claim: Renewable energy is cost-effective<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="reasoning_multiple_candidates_generate_argument",
			),
			# REASONING target with multiple reasoning steps
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				ScoringTarget.REASONING,  					# scoring_target
				[  											# reasoning_candidates
					{MathField.MATH_OPERATION: ["First step", "Second step", "Third step"]}
				],
				None,  										# output_candidates
				None,  										# action_candidates
				[  											# expected_documents
					"""<Document>: # Reasoning
math_operation: First step
math_operation: Second step
math_operation: Third step<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="reasoning_multiple_steps_single_candidate",
			),
			# OUTPUT target tests with SolveMathProblemWithReasoning
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				ScoringTarget.OUTPUT,  						# scoring_target
				[  											# reasoning_candidates
					{MathField.MATH_OPERATION: ["3 times 4"]}
				],
				[  											# output_candidates
					{"answer": "12"}
				],
				None,  										# action_candidates
				[  											# expected_documents
					"""<Document>: # Output
answer: 12<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="output_single_candidate_with_reasoning_math_problem",
			),
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				ScoringTarget.OUTPUT,  						# scoring_target
				[  											# reasoning_candidates
					{MathField.MATH_OPERATION: ["3 times 4"]},
					{MathField.MATH_OPERATION: ["Multiply 3 and 4"]},
				],
				[  											# output_candidates
					{"answer": "12"},
					{"answer": "12"},
				],
				None,  										# action_candidates
				[  											# expected_documents
					"""<Document>: # Output
answer: 12<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
					"""<Document>: # Output
answer: 12<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="output_multiple_candidates_with_reasoning_math_problem",
			),
			# OUTPUT target tests with GenerateArgumentWithReasoning
			pytest.param(
				GenerateArgumentWithReasoning,  			# signature
				ScoringTarget.OUTPUT,  						# scoring_target
				[  											# reasoning_candidates
					{ArgumentField.CLAIM: ["Renewable energy reduces carbon emissions"]}
				],
				[  											# output_candidates
					{"argument": "Renewable energy is essential for combating climate change."}
				],
				None,  										# action_candidates
				[  											# expected_documents
					"""<Document>: # Output
argument: Renewable energy is essential for combating climate change.<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="output_single_candidate_with_reasoning_generate_argument",
			),
			pytest.param(
				GenerateArgumentWithReasoning,  			# signature
				ScoringTarget.OUTPUT,  						# scoring_target
				[  											# reasoning_candidates
					{ArgumentField.CLAIM: ["Space exploration is expensive"]},
					{ArgumentField.CLAIM: ["Resources better spent on Earth"]},
				],
				[  											# output_candidates
					{"argument": "Space exploration diverts funds from pressing Earth issues."},
					{"argument": "We should prioritize solving problems on Earth first."},
				],
				None,  										# action_candidates
				[  											# expected_documents
					# first candidate
					"""<Document>: # Output
argument: Space exploration diverts funds from pressing Earth issues.<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
					# second candidate
					"""<Document>: # Output
argument: We should prioritize solving problems on Earth first.<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="output_multiple_candidates_with_reasoning_generate_argument",
			),
			# OUTPUT target with multiple output fields
			pytest.param(
				GenerateArgumentWithReasoning,  			# signature
				ScoringTarget.OUTPUT,  						# scoring_target
				[  											# reasoning_candidates
					{ArgumentField.CLAIM: ["Renewable energy reduces carbon emissions", "It creates jobs"]}
				],
				[  											# output_candidates
					{"argument": "Renewable energy is essential.", "conclusion": "We must transition now."}
				],
				None,  										# action_candidates
				[  											# expected_documents
					# first candidate
					"""<Document>: # Output
argument: Renewable energy is essential.
conclusion: We must transition now.<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="output_multiple_fields_single_candidate",
			),
			# ACTION target tests
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				ScoringTarget.ACTION,  						# scoring_target
				[  											# reasoning_candidates
					{MathField.MATH_OPERATION: ["Start by adding 7 and 8"]}
				],
				None,  										# output_candidates
				[  											# action_candidates
					ControllerContinueReasoningChoice.CONTINUE_REASONING
				],
				[  											# expected_documents
					# first candidate
					"""<Document>: # Action
continue_reasoning<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="action_single_candidate_with_reasoning_math_problem",
			),
			pytest.param(
				GenerateArgumentWithReasoning,  			# signature
				ScoringTarget.ACTION,  						# scoring_target
				[  											# reasoning_candidates
					{ArgumentField.CLAIM: ["Space exploration is expensive"]},
					{ArgumentField.CLAIM: ["Resources better spent on Earth"]},
				],
				None,  										# output_candidates
				[  											# action_candidates
					"select_style_structure",
					ControllerContinueReasoningChoice.FINISH,
				],
				[  											# expected_documents
					# first candidate
					"""<Document>: # Action
select_style_structure<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
					# second candidate
					"""<Document>: # Action
finish<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="action_multiple_candidates_with_reasoning_generate_argument",
			),
			# Edge case: REASONING with no candidates (should return empty list)
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				ScoringTarget.REASONING,  					# scoring_target
				None,  										# reasoning_candidates
				None,  										# output_candidates
				None,  										# action_candidates
				[],  										# expected_documents
				None,  										# expected_error
				id="reasoning_no_candidates_returns_empty",
			),
			# Edge case: OUTPUT with no candidates (should return empty list)
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				ScoringTarget.OUTPUT,  						# scoring_target
				None,  										# reasoning_candidates
				None,  										# output_candidates
				None,  										# action_candidates
				[],  										# expected_documents
				None,  										# expected_error
				id="output_no_candidates_returns_empty",
			),
			# Edge case: ACTION with no candidates (should return empty list)
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				ScoringTarget.ACTION,  						# scoring_target
				None,  										# reasoning_candidates
				None,  										# output_candidates
				None,  										# action_candidates
				[],  										# expected_documents
				None,  										# expected_error
				id="action_no_candidates_returns_empty",
			),
		],
	)
	def test_format_documents_generates_valid_prompts(
		self,
		scoring_adapter: LocalVLLMScoringAdapter,
		signature: type[ReasoningSignature],
		scoring_target: Literal[
			ScoringTarget.REASONING, ScoringTarget.OUTPUT, ScoringTarget.ACTION
		],
		reasoning_candidates: list[Reasoning] | None,
		output_candidates: list[Output] | None,
		action_candidates: list[str] | None,
		expected_documents: list[str],
		expected_error: type[Exception] | None,
	) -> None:
		"""
		Test that format_documents generates exact expected output strings.

		Verifies that the formatted documents exactly match the expected output
		for various combinations of signatures, scoring targets, and candidates.

		Args:
			scoring_adapter: The scoring adapter instance for testing.
			signature: The scoring signature to test with.
			scoring_target: The scoring target (reasoning/output/action).
			reasoning_candidates: List of reasoning candidates.
			output_candidates: List of output candidates.
			action_candidates: List of action candidate strings.
			expected_documents: List of expected document strings.
			expected_error: Expected exception type if error should occur.
		"""
		if expected_error is not None:
			with pytest.raises(expected_error):
				scoring_adapter.format_documents(
					scoring_target=scoring_target,
					reasoning_candidates=reasoning_candidates,
					output_candidates=output_candidates,
					action_candidates=action_candidates,
				)
		else:
			formatted_documents = scoring_adapter.format_documents(
				scoring_target=scoring_target,
				reasoning_candidates=reasoning_candidates,
				output_candidates=output_candidates,
				action_candidates=action_candidates,
			)

			# Should generate expected number of documents
			assert len(formatted_documents) == len(expected_documents), (
				f"Expected {len(expected_documents)} documents, got {len(formatted_documents)}"
			)

			# Each document should exactly match the expected string
			for i, (formatted_doc, expected_doc) in enumerate(
				zip(formatted_documents, expected_documents, strict=True)
			):
				assert formatted_doc == expected_doc, (
					f"Document {i} mismatch:\n"
					f"Expected:\n{expected_doc}\n\n"
					f"Got:\n{formatted_doc}"
				)

if __name__ == "__main__":
	pytest.main([__file__, "-vv"])

