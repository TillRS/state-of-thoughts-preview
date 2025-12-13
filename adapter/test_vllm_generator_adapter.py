# Standard library imports
from typing import Any

# Third-party imports
import pytest
from dspy.utils.exceptions import AdapterParseError

from adapter.adapter_constants import XMLTag

# Local imports
from adapter.constraints import GranularityType, ResponseLength
from adapter.vllm_generator_adapter import (
	VLLMGeneratorAdapter,
)
from lm.lm_constants import MessageKey, MessageRole
from predict.controller_constants import (
	ControllerContinueReasoningChoice,
	ControllerOutputParameters,
)
from signatures import (
	ArgumentStance,
	GenerateArgumentWithReasoning,
	QuestionAnsweringWithReasoning,
	ReasoningSignature,
	SolveMathProblemWithReasoning,
)
from signatures.example_signatures import ArgumentField, MathField
from tree.tree_constants import ReasoningState
from utilities_for_tests import MockGenerativeLocalVLLM


@pytest.fixture
def vllm_generator_adapter():
	return VLLMGeneratorAdapter()


# Test cases for create_system_prompt
@pytest.mark.parametrize(
	# Parameter names
	[
		"signature",
		"thought_length",
		"response_length",
		"has_internal_reasoning",
		"expected_system_prompt",
	],
	# Parameter values
	[
		pytest.param(
			SolveMathProblemWithReasoning,  				# signature
			ResponseLength(					# thought_length
				granularity=GranularityType.WORD,
				bounds=(20, 100),
			),
			ResponseLength(					# response_length
				granularity=GranularityType.WORD,
				bounds=(None, 20),
			),
			True,  							# has_internal_reasoning
			(  								# expected_system_prompt
				f"""
# Instructions

Solve the provided math problem and return its answer.

Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The answer to the math problem

When solving this problem, you must break down your solution into a series of reasoning steps, followed by a final answer.
Each step towards the answer should be encased within {XMLTag.STEP_START}...{XMLTag.STEP_END} tags, and contain a `math_operation` that advances the solution towards producing `answer`.
Your final answer must remain highly faithful to the reasoning steps and their underlying ideas.

- Preserve the full set of reasoning steps and their original order.
- You may lightly rephrase for clarity and readability, but the meaning must remain unchanged.
- Structure and sequence should closely follow the original reasoning steps.
- Do NOT introduce new ideas, arguments, facts, or examples.
- Do NOT remove or significantly alter any existing reasoning.

Your goal is to produce a clear synthesis that respects both the content and structure of the original reasoning, while allowing minimal refinement.

Your reasoning process should follow the rules below:
- Each `math_operation` (of type `str`) entails a math operation towards solving the math problem.
- Before writing a new `math_operation`, start with some internal reasoning which discusses and guides what to do with the next `math_operation`.
- Each `math_operation` should be between 20 and 100 words.
- Your final answer should be at most 20 words.

## Response Format

Once a user provides `math_problem`, your response must follow this exact template:

{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## internal_reasoning
Your internal reasoning about the first `math_operation`
## {MathField.MATH_OPERATION}
The first reasoning step towards producing `answer`
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## internal_reasoning
Your internal reasoning about the second `math_operation`
## {MathField.MATH_OPERATION}
The second reasoning step towards producing `answer`
{XMLTag.STEP_END}
...
{XMLTag.STEP_START}
## internal_reasoning
Your internal reasoning about the final `math_operation`
## {MathField.MATH_OPERATION}
The final reasoning step towards producing `answer`
{XMLTag.STEP_END}
{XMLTag.THINKING_END}
{XMLTag.ANSWER_START}
## {MathField.ANSWER}
Your response for `answer` here
{XMLTag.ANSWER_END}
""".strip()
			),
			id="solve_math_problem_word_bounds_with_internal_reasoning",
		),
		pytest.param(
			SolveMathProblemWithReasoning,	# signature
			ResponseLength(					# thought_length
				granularity=GranularityType.SENTENCE,
				bounds=(2, 5),
			),
			ResponseLength(					# response_length
				granularity=GranularityType.WORD,
				bounds=(None, 15),
			),
			False,  						# has_internal_reasoning
			(  								# expected_system_prompt
				f"""
# Instructions
Solve the provided math problem and return its answer.

Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The answer to the math problem

When solving this problem, you must break down your solution into a series of reasoning steps, followed by a final answer.
Each step towards the answer should be encased within {XMLTag.STEP_START}...{XMLTag.STEP_END} tags, and contain a `math_operation` that advances the solution towards producing `answer`.
Your final answer must remain highly faithful to the reasoning steps and their underlying ideas.

- Preserve the full set of reasoning steps and their original order.
- You may lightly rephrase for clarity and readability, but the meaning must remain unchanged.
- Structure and sequence should closely follow the original reasoning steps.
- Do NOT introduce new ideas, arguments, facts, or examples.
- Do NOT remove or significantly alter any existing reasoning.

Your goal is to produce a clear synthesis that respects both the content and structure of the original reasoning, while allowing minimal refinement.

Your reasoning process should follow the rules below:
- Each `math_operation` (of type `str`) entails a math operation towards solving the math problem.
- Each `math_operation` should be between 2 and 5 sentences.
- Your final answer should be at most 15 words.

## Response Format
Once a user provides `math_problem`, your response must follow this exact template:

{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
The first reasoning step towards producing `answer`
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
The second reasoning step towards producing `answer`
{XMLTag.STEP_END}
...
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
The final reasoning step towards producing `answer`
{XMLTag.STEP_END}
{XMLTag.THINKING_END}
{XMLTag.ANSWER_START}
## {MathField.ANSWER}
Your response for `answer` here
{XMLTag.ANSWER_END}
""".strip()
			),
			id="solve_math_problem_sentence_bounds_no_internal_reasoning",
		),
		pytest.param(
			GenerateArgumentWithReasoning,	# signature
			ResponseLength(					# thought_length
				granularity=GranularityType.WORD,
				bounds=(15, 80),
			),
			ResponseLength(					# response_length
				granularity=GranularityType.SENTENCE,
				bounds=(1, 3),
			),
			True,  							# has_internal_reasoning
			(  								# expected_system_prompt
				f"""
# Instructions

Generate an argument which takes the provided stance towards the provided topic.

Your inputs will be:
1. `topic` (str): The topic to generate an argument about
2. `stance` (ArgumentStance): The stance to take on the topic

Your goal is to produce the following output:
`argument` (str): The generated argument

When solving this problem, you must break down your solution into a series of reasoning steps, followed by a final answer.
Each step towards the answer should be encased within {XMLTag.STEP_START}...{XMLTag.STEP_END} tags, and contain a `claim` that advances the solution towards producing `argument`.
Your final answer must remain highly faithful to the reasoning steps and their underlying ideas.

- Preserve the full set of reasoning steps and their original order.
- You may lightly rephrase for clarity and readability, but the meaning must remain unchanged.
- Structure and sequence should closely follow the original reasoning steps.
- Do NOT introduce new ideas, arguments, facts, or examples.
- Do NOT remove or significantly alter any existing reasoning.

Your goal is to produce a clear synthesis that respects both the content and structure of the original reasoning, while allowing minimal refinement.

Your reasoning process should follow the rules below:
- Each `claim` (of type `str`) entails a component of the argument that advocates for the given stance towards the topic.
- Before writing a new `claim`, start with some internal reasoning which discusses and guides what to do with the next `claim`.
- Each `claim` should be between 15 and 80 words.
- Your final answer should be between 1 and 3 sentences.

## Response Format

Once a user provides `topic` and `stance`, your response must follow this exact template:

{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## internal_reasoning
Your internal reasoning about the first `claim`
## claim
The first reasoning step towards producing `argument`
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## internal_reasoning
Your internal reasoning about the second `claim`
## claim
The second reasoning step towards producing `argument`
{XMLTag.STEP_END}
...
{XMLTag.STEP_START}
## internal_reasoning
Your internal reasoning about the final `claim`
## claim
The final reasoning step towards producing `argument`
{XMLTag.STEP_END}
{XMLTag.THINKING_END}
{XMLTag.ANSWER_START}
## argument
Your response for `argument` here
{XMLTag.ANSWER_END}
""".strip()
			),
			id="generate_argument_mixed_granularity_with_internal_reasoning",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			ResponseLength(				# thought_length
				granularity=GranularityType.PARAGRAPH, bounds=(1, 2)
			),
			ResponseLength(				# response_length
				granularity=GranularityType.WORD,
				bounds=(25, None),
			),
			True,  						# has_internal_reasoning
			(  							# expected_system_prompt
				f"""
# Instructions

Solve the provided math problem and return its answer.

Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The answer to the math problem

When solving this problem, you must break down your solution into a series of reasoning steps, followed by a final answer.
Each step towards the answer should be encased within {XMLTag.STEP_START}...{XMLTag.STEP_END} tags, and contain a `math_operation` that advances the solution towards producing `answer`.
Your final answer must remain highly faithful to the reasoning steps and their underlying ideas.

- Preserve the full set of reasoning steps and their original order.
- You may lightly rephrase for clarity and readability, but the meaning must remain unchanged.
- Structure and sequence should closely follow the original reasoning steps.
- Do NOT introduce new ideas, arguments, facts, or examples.
- Do NOT remove or significantly alter any existing reasoning.

Your goal is to produce a clear synthesis that respects both the content and structure of the original reasoning, while allowing minimal refinement.

Your reasoning process should follow the rules below:
- Each `math_operation` (of type `str`) entails a math operation towards solving the math problem.
- Before writing a new `math_operation`, start with some internal reasoning which discusses and guides what to do with the next `math_operation`.
- Each `math_operation` should be between 1 and 2 paragraphs.
- Your final answer should be at least 25 words.

## Response Format

Once a user provides `math_problem`, your response must follow this exact template:

{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## internal_reasoning
Your internal reasoning about the first `math_operation`
## {MathField.MATH_OPERATION}
The first reasoning step towards producing `answer`
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## internal_reasoning
Your internal reasoning about the second `math_operation`
## {MathField.MATH_OPERATION}
The second reasoning step towards producing `answer`
{XMLTag.STEP_END}
...
{XMLTag.STEP_START}
## internal_reasoning
Your internal reasoning about the final `math_operation`
## {MathField.MATH_OPERATION}
The final reasoning step towards producing `answer`
{XMLTag.STEP_END}
{XMLTag.THINKING_END}
{XMLTag.ANSWER_START}
## {MathField.ANSWER}
Your response for `answer` here
{XMLTag.ANSWER_END}
""".strip()
			),
			id="solve_math_problem_paragraph_bounds_with_internal_reasoning",
		),
	],
)
def test_create_system_prompt(
	vllm_generator_adapter: VLLMGeneratorAdapter,
	signature: type[ReasoningSignature],
	thought_length: ResponseLength,
	response_length: ResponseLength,
	has_internal_reasoning: bool,
	expected_system_prompt: str,
) -> None:
	"""
	Test the system prompt generation for VLLMGeneratorAdapter with various parameter combinations.

	This test checks that the system prompt is correctly generated based on different
	signatures, length constraints, and chain-of-thought configurations.

	Args:
		vllm_generator_adapter: The VLLMGeneratorAdapter instance to test.
		signature: The reasoning signature to use for formatting.
		thought_length: Response length constraints for reasoning steps.
		response_length: Response length constraints for final outputs.
		has_internal_reasoning: Whether internal reasoning guidance is provided.
		expected_system_prompt: The expected system prompt string.
	"""
	system_prompt = vllm_generator_adapter.create_system_prompt(
		signature=signature,
		thought_length=thought_length,
		response_length=response_length,
		has_internal_reasoning=has_internal_reasoning,
	)
	# Compare the generated prompt with the expected prompt
	assert system_prompt.strip() == expected_system_prompt.strip()


# Test cases for format_continued_assistant_message
@pytest.mark.parametrize(
	# Parameter names
	[
		"previous_content",
		"internal_reasoning",
		"prefix",
		"continue_reasoning",
		"signature",
		"expected",
	],
	# Parameter values
	[
		pytest.param(
			"",  					# previous_content
			(  						# internal_reasoning
				"I need to start by identifying the key variables in this problem."
			),
			"", 					# prefix
			True, 					# continue_reasoning
			SolveMathProblemWithReasoning,		# signature
			(  						# expected message
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I need to start by identifying the key variables in this problem.
## {MathField.MATH_OPERATION}
""".strip()
			),
			id="start_new_reasoning_step",
		),
		pytest.param(
			(  						# previous_content
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
First step reasoning
## {MathField.MATH_OPERATION}
First operation
{XMLTag.STEP_END}
""".strip()
			),
			(  						# internal_reasoning_for_output
				"Now I'll solve for x."
			),
			"", 					# prefix
			True, 					# continue_reasoning
			SolveMathProblemWithReasoning,		# signature
			(  						# expected message
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
First step reasoning
## {MathField.MATH_OPERATION}
First operation
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
Now I'll solve for x.
## {MathField.MATH_OPERATION}
""".strip()
			),
			id="continue_reasoning_step",
		),
		pytest.param(
			(  						# previous_content
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
First step internal reasoning
## {MathField.MATH_OPERATION}
First step operation
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
Second step internal reasoning
## {MathField.MATH_OPERATION}
Second step operation
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
Final step reasoning
## {MathField.MATH_OPERATION}
Final operation
{XMLTag.STEP_END}
{XMLTag.THINKING_END}
""".strip()
			),
			(  						# internal_reasoning_for_output
				"The answer is clearly 42."
			),
			"The answer is", 		# prefix
			False, 					# continue_reasoning
			SolveMathProblemWithReasoning,		# signature
			(  						# expected message
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
First step internal reasoning
## {MathField.MATH_OPERATION}
First step operation
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
Second step internal reasoning
## {MathField.MATH_OPERATION}
Second step operation
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
Final step reasoning
## {MathField.MATH_OPERATION}
Final operation
{XMLTag.STEP_END}
{XMLTag.THINKING_END}
{XMLTag.ANSWER_START}
## {MathField.ANSWER}
The answer is
""".strip()
			),
			id="switch_to_answer_section",
		),
		pytest.param(
			(  						# previous_content
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
I need to subtract 5 from 12
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
12 - 5 = 12 - 2 - 3 = 10 - 3
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
10 - 3 = 7
{XMLTag.STEP_END}
""".strip()
			),
			(  						# internal_reasoning_for_output
				"Based on the steps above, I can conclude the answer."
			),
			"", 					# prefix
			False, 					# continue_reasoning
			SolveMathProblemWithReasoning,		# signature
		(  							# expected message
			f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
I need to subtract 5 from 12
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
12 - 5 = 12 - 2 - 3 = 10 - 3
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
10 - 3 = 7
{XMLTag.STEP_END}
{XMLTag.THINKING_END}
{XMLTag.ANSWER_START}
## {MathField.ANSWER}
""".strip()
		),
			id="transition_to_answer_without_closing_tag",
		),
	pytest.param(
		(  						# previous_content
			f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
First step
{XMLTag.STEP_END}
{XMLTag.THINKING_END}
""".strip()
		),
		(  						# internal_reasoning_for_output
			"Now let's continue with the next step."
		),
		"", 					# prefix
		True, 					# continue_reasoning
		SolveMathProblemWithReasoning,		# signature
		(  						# expected message
			f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
First step
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
Now let's continue with the next step.
## {MathField.MATH_OPERATION}
""".strip()
		),
		id="continue_reasoning_with_closed_thinking",
	),
	],
)
def test_format_continued_assistant_message(
	vllm_generator_adapter: VLLMGeneratorAdapter,
	previous_content: str,
	internal_reasoning: str,
	prefix: str,
	continue_reasoning: bool,
	signature: type[ReasoningSignature],
	expected: str,
):
	"""
	Test the formatting of continued assistant messages.

	This test checks that the assistant message is formatted correctly based on the
	previous content, internal reasoning, and other parameters.

	Args:
	    vllm_generator_adapter: The VLLMGeneratorAdapter instance to test.
	    previous_content: The content of the previous assistant message.
	    internal_reasoning: The internal reasoning for the next step.
	    prefix: The prefix for the output section.
	    continue_reasoning: Whether to continue reasoning or switch to answer section.
	    signature: The signature class to use for formatting.
	    expected: The expected formatted message string.
	"""
	result = vllm_generator_adapter.format_continued_assistant_message(
		previous_content=previous_content,
		internal_reasoning_for_output=internal_reasoning,
		prefix_for_output=prefix,
		continue_reasoning=continue_reasoning,
		signature=signature,
	)
	assert result == expected


# Test cases for user_message_output_requirements
@pytest.mark.parametrize(
	# Parameter names
	[
		"signature",
		"has_internal_reasoning",
		"expected_user_message_output_requirements",
	],
	# Parameter values
	[
	pytest.param(
		SolveMathProblemWithReasoning,  			# signature
		True,  						# has_internal_reasoning
		(  							# expected_user_message_output_requirements
			f"""
Structure your response as follows:

1. Begin with `{XMLTag.THINKING_START}...{XMLTag.THINKING_END}` tags. Inside these tags, include multiple `{XMLTag.STEP_START}...{XMLTag.STEP_END}` sections for your math_operations.
	Each `{XMLTag.STEP_START}` section should include a `## {ControllerOutputParameters.INTERNAL_REASONING}` section (guidance provided to help your thinking), followed by a `## {MathField.MATH_OPERATION}` section.
2. After the `{XMLTag.THINKING_END}` tag, include your final answer within `{XMLTag.ANSWER_START}...{XMLTag.ANSWER_END}` tags.
	The `{XMLTag.ANSWER_START}` section should include sections for `answer`.
""".strip()
		),
		id="solve_math_problem_with_internal_reasoning",
	),
	pytest.param(
		GenerateArgumentWithReasoning,  			# signature
		True,  						# has_internal_reasoning
		(  							# expected_user_message_output_requirements
			f"""
Structure your response as follows:

1. Begin with `{XMLTag.THINKING_START}...{XMLTag.THINKING_END}` tags. Inside these tags, include multiple `{XMLTag.STEP_START}...{XMLTag.STEP_END}` sections for your claims.
	Each `{XMLTag.STEP_START}` section should include a `## {ControllerOutputParameters.INTERNAL_REASONING}` section (guidance provided to help your thinking), followed by a `## claim` section.
2. After the `{XMLTag.THINKING_END}` tag, include your final answer within `{XMLTag.ANSWER_START}...{XMLTag.ANSWER_END}` tags.
	The `{XMLTag.ANSWER_START}` section should include sections for `argument`.
""".strip()
		),
		id="generate_argument_with_internal_reasoning",
	),
	pytest.param(
		SolveMathProblemWithReasoning,  			# signature
		False,  					# has_internal_reasoning
		(  							# expected_user_message_output_requirements
			f"""
Structure your response as follows:

1. Begin with `{XMLTag.THINKING_START}...{XMLTag.THINKING_END}` tags. Inside these tags, include multiple `{XMLTag.STEP_START}...{XMLTag.STEP_END}` sections for your math_operations.
	Each `{XMLTag.STEP_START}` section should contain a `## {MathField.MATH_OPERATION}` section.
2. After the `{XMLTag.THINKING_END}` tag, include your final answer within `{XMLTag.ANSWER_START}...{XMLTag.ANSWER_END}` tags.
	The `<answer>` section should include sections for `answer`.
""".strip()
		),
		id="solve_math_problem_no_internal_reasoning",
	),
	],
)
def test_user_message_output_requirements(
	vllm_generator_adapter: VLLMGeneratorAdapter,
	signature: type[ReasoningSignature],
	has_internal_reasoning: bool,
	expected_user_message_output_requirements: str,
):
	"""
	Test the generation of output format requirements for the user message.

	This test checks that the method correctly generates user message output requirements
	based on the signature and chain-of-thought configuration.

	Args:
	    vllm_generator_adapter: The VLLMGeneratorAdapter instance to test.
	    signature: The reasoning signature to use for formatting.
	    has_internal_reasoning: Whether internal reasoning guidance is provided.
	    expected_user_message_output_requirements: The expected output requirements string.
	"""
	result = vllm_generator_adapter.user_message_output_requirements(
		signature=signature,
		has_internal_reasoning=has_internal_reasoning,
	)
	assert result == expected_user_message_output_requirements


# Test cases for format_demo_assistant_message
@pytest.mark.parametrize(
	# Parameter names
	[
		"signature",
		"demo",
		"has_internal_reasoning",
		"expected",
	],
	# Parameter values
	[
	pytest.param(
		SolveMathProblemWithReasoning,  			# signature
		(  							# demo
			{
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 1+1?"},
				ReasoningState.REASONING: [
					{
						ControllerOutputParameters.INTERNAL_REASONING: "I need to add 1 and 1.",
						MathField.MATH_OPERATION: "1+1=2",
					}
				],
				ReasoningState.OUTPUT: {
					ControllerOutputParameters.INTERNAL_REASONING: "The answer is clearly 2.",
					MathField.ANSWER: "2",
				},
			}
		),
		True,  						# has_internal_reasoning
		(  							# expected
			f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I need to add 1 and 1.
## {MathField.MATH_OPERATION}
1+1=2
{XMLTag.STEP_END}
{XMLTag.THINKING_END}
{XMLTag.ANSWER_START}
## {MathField.ANSWER}
2
{XMLTag.ANSWER_END}
""".strip()
		),
		id="solve_math_problem_with_internal_reasoning",
	),
	pytest.param(
		SolveMathProblemWithReasoning,  			# signature
		(  							# demo
			{
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 1+1?"},
				ReasoningState.REASONING: [
					{
						ControllerOutputParameters.INTERNAL_REASONING: "I need to add 1 and 1.",
						MathField.MATH_OPERATION: "1+1=2",
					}
				],
				ReasoningState.OUTPUT: {
					ControllerOutputParameters.INTERNAL_REASONING: "The answer is clearly 2.",
					MathField.ANSWER: "2",
				},
			}
		),
		False,  					# has_internal_reasoning
		(  							# expected
			f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
1+1=2
{XMLTag.STEP_END}
{XMLTag.THINKING_END}
{XMLTag.ANSWER_START}
## {MathField.ANSWER}
2
{XMLTag.ANSWER_END}
""".strip()
		),
		id="solve_math_problem_no_internal_reasoning",
	),
	],
)
def test_format_demo_assistant_message(
	vllm_generator_adapter: VLLMGeneratorAdapter,
	signature: type[ReasoningSignature],
	demo: dict[str, Any],
	has_internal_reasoning: bool,
	expected: str,
) -> None:
	"""
	Test formatting of demo assistant messages with reasoning steps and outputs.

	Args:
	    vllm_generator_adapter: The VLLMGeneratorAdapter instance to test.
	    signature: The signature class to use for formatting.
	    demo: The demo to format.
	    has_internal_reasoning: Whether internal reasoning guidance is provided.
	    expected: The expected formatted message.
	"""
	result = vllm_generator_adapter.format_demo_assistant_message(
		signature=signature,
		demo=demo,
		has_internal_reasoning=has_internal_reasoning,
	)
	assert result == expected


# Test cases for format_user_message_content
@pytest.mark.parametrize(
	# Parameter names
	[
		"signature",
		"inputs",
		"main_request",
		"expected_message",
		"expected_error",
	],
	# Parameter values
	[
	pytest.param(
		SolveMathProblemWithReasoning,  			# signature
		{							# inputs
			MathField.MATH_PROBLEM: "What is 1+1?"
		},
		False,  					# main_request
		(  							# expected_message
			"""
Solve the provided math problem and return its answer.

## math_problem
What is 1+1?
""".strip()
		),
		None,  						# expected_error
		id="solve_math_problem_basic_input_formatting",
	),
		pytest.param(
			SolveMathProblemWithReasoning,  		# signature
			{						# inputs
				MathField.MATH_PROBLEM: "What is 1+1?"
			},
			True,  					# main_request
			(  						# expected_message
			f"""
Solve the provided math problem and return its answer.

## math_problem
What is 1+1?

To produce `answer`, reason step-by-step by writing a sequence of math_operations.

Structure your response as follows:

1. Begin with `{XMLTag.THINKING_START}...{XMLTag.THINKING_END}` tags. Inside these tags, include multiple `{XMLTag.STEP_START}...{XMLTag.STEP_END}` sections for your math_operations.
	Each `{XMLTag.STEP_START}` section should contain a `## {MathField.MATH_OPERATION}` section.
2. After the `{XMLTag.THINKING_END}` tag, include your final answer within `{XMLTag.ANSWER_START}...{XMLTag.ANSWER_END}` tags.
	The `<answer>` section should include sections for `answer`.
""".strip()
		),
			None,  					# expected_error
			id="solve_math_problem_main_request_complete",
		),
		pytest.param(
			SolveMathProblemWithReasoning,  			# signature
			{							# inputs
				MathField.MATH_PROBLEM: "What is 1+1?"
			},
			True,  						# main_request
		(  								# expected_message
			f"""
Solve the provided math problem and return its answer.

## math_problem
What is 1+1?

To produce `answer`, reason step-by-step by writing a sequence of math_operations.

Structure your response as follows:

1. Begin with `{XMLTag.THINKING_START}...{XMLTag.THINKING_END}` tags. Inside these tags, include multiple `{XMLTag.STEP_START}...{XMLTag.STEP_END}` sections for your math_operations.
	Each `{XMLTag.STEP_START}` section should contain a `## {MathField.MATH_OPERATION}` section.
2. After the `{XMLTag.THINKING_END}` tag, include your final answer within `{XMLTag.ANSWER_START}...{XMLTag.ANSWER_END}` tags.
	The `<answer>` section should include sections for `answer`.
""".strip()
		),
		None,  						# expected_error
		id="solve_math_problem_main_request_no_internal_reasoning",
	),
	pytest.param(
		GenerateArgumentWithReasoning,  			# signature
		(  							# inputs
			{ArgumentField.TOPIC: "renewable energy", ArgumentField.STANCE: ArgumentStance.PRO.value}
		),
		False,  					# main_request
		(  							# expected_message
			"""
Generate an argument which takes the provided stance towards the provided topic.

## topic
renewable energy

## stance
PRO
""".strip()
		),
		None,  						# expected_error
		id="multiple_input_fields_formatting",
	),
	pytest.param(
		SolveMathProblemWithReasoning,  			# signature
		{},  						# inputs
		False,  					# main_request
		(  							# expected_message
			f"""
Solve the provided math problem and return its answer.

## math_problem
What is 1+1?

To produce `answer`, reason step-by-step by writing a sequence of math_operations.

Structure your response as follows:

1. Begin with `{XMLTag.THINKING_START}...{XMLTag.THINKING_END}` tags. Inside these tags, include multiple `{XMLTag.STEP_START}...{XMLTag.STEP_END}` sections for your math_operations.
	Each `{XMLTag.STEP_START}` section should contain a `## {MathField.MATH_OPERATION}` section.
2. After the `{XMLTag.THINKING_END}` tag, include your final answer within `{XMLTag.ANSWER_START}...{XMLTag.ANSWER_END}` tags.
	The `<answer>` section should include sections for `answer`.
""".strip()
		),
		AssertionError,  			# expected_error
		id="empty_inputs_only_instructions",
	),
	],
)
def test_format_user_message_content(
	vllm_generator_adapter: VLLMGeneratorAdapter,
	signature: type[ReasoningSignature],
	inputs: dict[str, Any],
	main_request: bool,
	expected_message: str | None,
	expected_error: type[Exception] | None,
) -> None:
	"""
	Test formatting of user message content with different inputs and main request flags.

	This test checks that the formatted content exactly matches the expected output
	based on the signature and inputs, or raises the expected error. It verifies that
	the content includes the input fields and, if applicable, the main request guidance.

	Args:
	    vllm_generator_adapter: The VLLMGeneratorAdapter instance to test.
	    signature: The signature class to use for formatting.
	    inputs: The inputs to format in the user message.
	    main_request: True if the user request is the final one (containing the input),
	        and False if it is part of an in-context example (pair of user-assistant messages).
	    expected_message: The exact expected output string (None if error expected).
	    expected_error: The expected error type (None if success expected).
	"""
	if expected_error is not None:
		with pytest.raises(expected_error):
			vllm_generator_adapter.format_user_message_content(
				signature=signature,
				inputs=inputs,
				main_request=main_request,
			)
	else:
		result = vllm_generator_adapter.format_user_message_content(
			signature=signature,
			inputs=inputs,
			main_request=main_request,
		)
		assert result == expected_message


@pytest.mark.parametrize(
	# Parameter names
	[
		"demos",
		"has_internal_reasoning",
		"expected_messages",
		"expected_error",
	],
	# Parameter values
	[
		pytest.param(
			[{  			# demos
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 1+1?"},
				ReasoningState.REASONING: [
					{
						ControllerOutputParameters.INTERNAL_REASONING: "I need to add 1 and 1.",
						MathField.MATH_OPERATION: "1+1=2",
					}
				],
				ReasoningState.OUTPUT: {
					ControllerOutputParameters.INTERNAL_REASONING: "The answer is clearly 2.",
					MathField.ANSWER: "2",
				},
			}],
			True,  			# has_internal_reasoning
			[  				# expected_messages
				{  			# User demo message
					MessageKey.ROLE: MessageRole.USER,
					MessageKey.CONTENT: """
Solve the provided math problem and return its answer.

## math_problem
What is 1+1?
""".strip(),
				},
				{  			# Assistant demo message
					MessageKey.ROLE: MessageRole.ASSISTANT,
					MessageKey.CONTENT: f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I need to add 1 and 1.
## {MathField.MATH_OPERATION}
1+1=2
{XMLTag.STEP_END}
{XMLTag.THINKING_END}
{XMLTag.ANSWER_START}
## {MathField.ANSWER}
2
{XMLTag.ANSWER_END}
""".strip(),
				},
			],
			None,  			# expected_error
			id="math_demo_1_with_internal_reasoning",
		),
		pytest.param(
			[{  			# demos
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 3*3+4?"},
				ReasoningState.REASONING: [
					{
						ControllerOutputParameters.INTERNAL_REASONING: "Multiply 3 by itself.",
						MathField.MATH_OPERATION: "3*3=9",
					},
					{
						ControllerOutputParameters.INTERNAL_REASONING: "Add 9 and 4.",
						MathField.MATH_OPERATION: "9+4=13",
					}
				],
				ReasoningState.OUTPUT: {
					ControllerOutputParameters.INTERNAL_REASONING: "The answer is 13.",
					MathField.ANSWER: "13",
				},
			}],
			True,  			# has_internal_reasoning
			[  				# expected_messages
				{  			# User demo message
					MessageKey.ROLE: MessageRole.USER,
					MessageKey.CONTENT: """
Solve the provided math problem and return its answer.

## math_problem
What is 3*3+4?
""".strip(),
				},
				{  			# Assistant demo message
					MessageKey.ROLE: MessageRole.ASSISTANT,
					MessageKey.CONTENT: f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
Multiply 3 by itself.
## {MathField.MATH_OPERATION}
3*3=9
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
Add 9 and 4.
## {MathField.MATH_OPERATION}
9+4=13
{XMLTag.STEP_END}
{XMLTag.THINKING_END}
{XMLTag.ANSWER_START}
## {MathField.ANSWER}
13
{XMLTag.ANSWER_END}
""".strip(),
				},
			],
			None,  				# expected_error
			id="math_demo_2_with_internal_reasoning",
		),
		pytest.param(
			[{  				# demos
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 1+1?"},
				ReasoningState.REASONING: [
					{
						ControllerOutputParameters.INTERNAL_REASONING: "I need to add 1 and 1.",
						MathField.MATH_OPERATION: "1+1=2",
					}
				],
				ReasoningState.OUTPUT: {
					ControllerOutputParameters.INTERNAL_REASONING: "The answer is clearly 2.",
					MathField.ANSWER: "2",
				},
			},
			{  					# demos
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 3*3+4?"},
				ReasoningState.REASONING: [
					{
						ControllerOutputParameters.INTERNAL_REASONING: "Multiply 3 by itself.",
						MathField.MATH_OPERATION: "3*3=9",
					},
					{
						ControllerOutputParameters.INTERNAL_REASONING: "Add 9 and 4.",
						MathField.MATH_OPERATION: "9+4=13",
					}
				],
				ReasoningState.OUTPUT: {
					ControllerOutputParameters.INTERNAL_REASONING: "The answer is 13.",
					MathField.ANSWER: "13",
				},
			}],
			True,  				# has_internal_reasoning
			[  					# expected_messages
				{  				# User demo message
					MessageKey.ROLE: MessageRole.USER,
					MessageKey.CONTENT: """
Solve the provided math problem and return its answer.

## math_problem
What is 1+1?
""".strip(),
				},
				{  				# Assistant demo message
					MessageKey.ROLE: MessageRole.ASSISTANT,
					MessageKey.CONTENT: f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I need to add 1 and 1.
## {MathField.MATH_OPERATION}
1+1=2
{XMLTag.STEP_END}
{XMLTag.THINKING_END}
{XMLTag.ANSWER_START}
## {MathField.ANSWER}
2
{XMLTag.ANSWER_END}
""".strip(),
				},
				{  				# User demo message
					MessageKey.ROLE: MessageRole.USER,
					MessageKey.CONTENT: """
Solve the provided math problem and return its answer.

## math_problem
What is 3*3+4?
""".strip(),
				},
				{  				# Assistant demo message
					MessageKey.ROLE: MessageRole.ASSISTANT,
					MessageKey.CONTENT: f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
Multiply 3 by itself.
## {MathField.MATH_OPERATION}
3*3=9
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
Add 9 and 4.
## {MathField.MATH_OPERATION}
9+4=13
{XMLTag.STEP_END}
{XMLTag.THINKING_END}
{XMLTag.ANSWER_START}
## {MathField.ANSWER}
13
{XMLTag.ANSWER_END}
""".strip(),
				},
			],					# expected_messages
			None,  				# expected_error
			id="math_demos_1_2_with_internal_reasoning",
		),
		pytest.param(
			[{					# demos (wrong output field)
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "2+2"},
				ReasoningState.REASONING: [{
					ControllerOutputParameters.INTERNAL_REASONING: "Add the numbers",
					MathField.MATH_OPERATION: "2 + 2 = 4",
				}],
				ReasoningState.OUTPUT: {"wrong_field": "4"},
			}],
			True,  				# has_internal_reasoning
			None,				# expected_messages
			AssertionError,		# expected_error
			id="error_demo_with_missing_output_field",
		),
		pytest.param(
			[{					# demos (non-list reasoning)
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "2+2"},
				ReasoningState.REASONING: "not a list",
				ReasoningState.OUTPUT: {MathField.ANSWER: "4"},
			}],
			True,  				# has_internal_reasoning
			None,				# expected_messages
			AssertionError,		# expected_error
			id="error_demo_with_non_list_reasoning",
		),
		pytest.param(
			[{					# demos (reasoning step missing required field)
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "2+2"},
				ReasoningState.REASONING: [{
					ControllerOutputParameters.INTERNAL_REASONING: "Some reasoning",
					"wrong_field": "value",  # Should be MathField.MATH_OPERATION
				}],
				ReasoningState.OUTPUT: {MathField.ANSWER: "4"},
			}],
			True,  				# has_internal_reasoning
			None,				# expected_messages
			AssertionError,		# expected_error
			id="error_demo_with_reasoning_step_missing_required_field",
		),
	],
)
def test_format_demos(
	vllm_generator_adapter: VLLMGeneratorAdapter,
	demos: list[dict[str, Any]],
	has_internal_reasoning: bool,
	expected_messages: list[dict[str, str]] | None,
	expected_error: type[Exception] | None,
) -> None:
	"""
	Test formatting of in-context examples into messages with parameterized demos.

	This test checks that demos are correctly formatted into user-assistant message pairs,
	or raises the expected error. It verifies message roles and content formatting.

	Args:
	    vllm_generator_adapter: The VLLMGeneratorAdapter instance to test.
	    demos: List of demos to format.
	    has_internal_reasoning: Whether internal reasoning guidance is provided.
	    expected_messages: List of expected messages (None if error expected).
	    expected_error: The expected error type (None if success expected).
	"""
	if expected_error is not None:
		with pytest.raises(expected_error):
			vllm_generator_adapter.format_demos(
				signature=SolveMathProblemWithReasoning,
				demos=demos,
				has_internal_reasoning=has_internal_reasoning,
			)
	else:
		result = vllm_generator_adapter.format_demos(
			signature=SolveMathProblemWithReasoning,
			demos=demos,
			has_internal_reasoning=has_internal_reasoning,
		)
		assert expected_messages is not None
		assert len(result) == len(expected_messages)
		for msg, expected in zip(result, expected_messages, strict=True):
			assert msg[MessageKey.ROLE] == expected[MessageKey.ROLE], \
				f"Expected {expected[MessageKey.ROLE]} but got {msg[MessageKey.ROLE]}. Message: {msg[MessageKey.CONTENT]}"
			assert msg[MessageKey.CONTENT].strip() == expected[MessageKey.CONTENT], \
				f"Expected {expected[MessageKey.CONTENT]} but got {msg[MessageKey.CONTENT]}. Message: {msg[MessageKey.CONTENT]}"


@pytest.mark.parametrize(
	# Parameter names
	[
		"signature",
		"inputs",
		"demos",
		"response_length",
		"has_internal_reasoning",
		"previous_content",
		"internal_reasoning_for_output",
		"prefix_for_output",
		"continue_reasoning",
		"expected_message_count",
		"expected_roles",
		"expected_final_message_content",
		"expected_error",
	],
	# Parameter values
	[
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				MathField.MATH_PROBLEM: "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			None,						# previous_content (empty)
			None,						# internal_reasoning_for_output
			None,						# prefix_for_output
			[True],						# continue_reasoning
			3,							# expected_message_count
			[							# expected_roles
				MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT
			],
			[							# xpected_final_message_content
				XMLTag.THINKING_START + "\n" + XMLTag.STEP_START + f"\n## {MathField.MATH_OPERATION}"
			],
			None,						# expected_error
			id="no_demos_no_previous_content_no_interventions",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				MathField.MATH_PROBLEM: "What is (10 / 2) + 6 / 3?"
			},
			[{  						# demos
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 1+1?"},
				ReasoningState.REASONING: [
					{ControllerOutputParameters.INTERNAL_REASONING: "I need to add 1 and 1.", MathField.MATH_OPERATION: "1+1=2"}
				],
				ReasoningState.OUTPUT: {
					ControllerOutputParameters.INTERNAL_REASONING: "The answer is clearly 2.", MathField.ANSWER: "2",
				},
			}],
			None,						# response_length
			True,  						# has_internal_reasoning
			"",							# previous_content (empty)
			None,						# internal_reasoning_for_output
			None,						# prefix_for_output
			[True],						# continue_reasoning
			5,							# expected_message_count
			[							# expected_roles
				MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT, MessageRole.USER, MessageRole.ASSISTANT
			],
			[							# expected_final_message_content
				XMLTag.THINKING_START + "\n" + XMLTag.STEP_START + f"\n## {MathField.MATH_OPERATION}"
			],
			None,						# expected_error
			id="with_demos_no_previous_content_no_interventions",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				MathField.MATH_PROBLEM: "What is (10 / 2) + 6 / 3?"
			},
			[{  						# demos
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 1+1?"},
				ReasoningState.REASONING: [
					{
						ControllerOutputParameters.INTERNAL_REASONING: "I need to add 1 and 1.",
						MathField.MATH_OPERATION: "1+1=2",
					}
				],
				ReasoningState.OUTPUT: {
					ControllerOutputParameters.INTERNAL_REASONING: "The answer is clearly 2.",
					MathField.ANSWER: "2",
				},
			}],
			None,						# response_length
			True,  						# has_internal_reasoning
			"",							# previous_content
			None,						# internal_reasoning_for_output
			None,						# prefix_for_output
			[True],						# continue_reasoning
			5,							# expected_message_count
			[
				MessageRole.SYSTEM, 					# system prompt
				MessageRole.USER, MessageRole.ASSISTANT, 		# in-context example #1
				MessageRole.USER, 						# main request
				MessageRole.ASSISTANT					# assistant response to main request
			],									# expected_roles
			[							# expected_final_message_content
				XMLTag.THINKING_START + "\n" + XMLTag.STEP_START + f"\n## {MathField.MATH_OPERATION}"
			],
			None,								# expected_error
			id="with_demos_no_previous_content_with_continue_reasoning_true",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				MathField.MATH_PROBLEM: "What is (10 / 2) + 6 / 3?"
			},
			[{  						# demos
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 1+1?"},
				ReasoningState.REASONING: [
					{
						ControllerOutputParameters.INTERNAL_REASONING: "I need to add 1 and 1.",
						MathField.MATH_OPERATION: "1+1=2",
					}
				],
				ReasoningState.OUTPUT: {
					ControllerOutputParameters.INTERNAL_REASONING: "The answer is clearly 2.",
					MathField.ANSWER: "2",
				},
			}],
			None,						# response_length
			True,  						# has_internal_reasoning
			"",							# previous_content
			None,						# internal_reasoning_for_output
			None,						# prefix_for_output
			[False],					# continue_reasoning
			5,							# expected_message_count
			[							# expected_roles
				MessageRole.SYSTEM, 					# system prompt
				MessageRole.USER, MessageRole.ASSISTANT, 		# in-context example #1
				MessageRole.USER, 						# main request
				MessageRole.ASSISTANT					# assistant response to main request
			],
			[							# expected_final_message_content
				XMLTag.ANSWER_START + "\n" + f"## {MathField.ANSWER}"
			],
			None,						# expected_error
			id="with_demos_no_previous_content_with_continue_reasoning_false",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				MathField.MATH_PROBLEM: "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
""".strip(),							# previous_content
			None,						# internal_reasoning_for_output
			None,						# prefix_for_output
			[True],						# continue_reasoning
			3,							# expected_message_count
			[							# expected_roles
				MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT
			],
			[							# expected_final_message_content
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
""".strip()
			],
			None,						# expected_error
			id="no_demos_with_previous_content_no_interventions",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				MathField.MATH_PROBLEM: "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
""".strip(),							# previous_content
			[							# internal_reasoning_for_output
				"Now I need to divide 6 by 3"
			],
			None,						# prefix_for_output
			[True],						# continue_reasoning
			3,							# expected_message_count
			[							# expected_roles
				MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT
			],
			[							# expected_final_message_content
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
Now I need to divide 6 by 3
## {MathField.MATH_OPERATION}
""".strip()
			],
			None,						# expected_error
			id="no_demos_with_previous_content_internal_reasoning_intervention",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				MathField.MATH_PROBLEM: "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
""".strip(),							# previous_content
			[							# internal_reasoning_for_output
				"Now I need to divide 6 by 3"
			],
			["6 / 3 ="],				# prefix_for_output
			[True],						# continue_reasoning
			3,							# expected_message_count
			[							# expected_roles
				MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT
			],
			[							# expected_final_message_content
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
Now I need to divide 6 by 3
## {MathField.MATH_OPERATION}
6 / 3 =
""".strip()
			],
			None,						# expected_error
			id="no_demos_with_previous_content_full_intervention",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				MathField.MATH_PROBLEM: "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
""".strip(),							# previous_content
			[							# internal_reasoning_for_output
				"6 / 3 = 2, so I should compute 5 + 2"
			],
			["The answer is"],			# prefix_for_output
			[False],					# continue_reasoning
			3,							# expected_message_count
			[							# expected_roles
				MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT
			],
			[							# expected_final_message_content (with final answer)
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
{XMLTag.THINKING_END}
{XMLTag.ANSWER_START}
## {MathField.ANSWER}
The answer is
""".strip()
			],
			None,						# expected_error
			id="no_demos_with_previous_content_with_interventions_final_answer",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				MathField.MATH_PROBLEM: "What is (10 / 2) + 6 / 3?"
			},
			[{  						# demos
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 1+1?"},
				ReasoningState.REASONING: [
					{
						ControllerOutputParameters.INTERNAL_REASONING: "I need to add 1 and 1.",
						MathField.MATH_OPERATION: "1+1=2",
					}
				],
				ReasoningState.OUTPUT: {
					ControllerOutputParameters.INTERNAL_REASONING: "The answer is clearly 2.",
					MathField.ANSWER: "2",
				},
			},
			{
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 3*3+4?"},
				ReasoningState.REASONING: [
					{
						ControllerOutputParameters.INTERNAL_REASONING: "Multiply 3 by itself.",
						MathField.MATH_OPERATION: "3*3=9",
					},
					{
						ControllerOutputParameters.INTERNAL_REASONING: "Add 9 and 4.",
						MathField.MATH_OPERATION: "9+4=13",
					}
				],
				ReasoningState.OUTPUT: {
					ControllerOutputParameters.INTERNAL_REASONING: "The answer is 13.",
					MathField.ANSWER: "13",
				},
			},
			{
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is (3*3+4)*2?"},
				ReasoningState.REASONING: [
					{
						ControllerOutputParameters.INTERNAL_REASONING: "Multiply 3 by itself.",
						MathField.MATH_OPERATION: "3*3=9",
					},
					{
						ControllerOutputParameters.INTERNAL_REASONING: "Add 9 and 4.",
						MathField.MATH_OPERATION: "9+4=13",
					},
					{
						ControllerOutputParameters.INTERNAL_REASONING: "Multiply 13 by 2.",
						MathField.MATH_OPERATION: "13*2=26",
					}
				],
				ReasoningState.OUTPUT: {
					ControllerOutputParameters.INTERNAL_REASONING: "The answer is 26.",
					MathField.ANSWER: "26",
				},
			}],
			None,						# response_length
			True,  						# has_internal_reasoning
			f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
""".strip(),							# previous_content
			None,						# internal_reasoning_for_output
			None,						# prefix_for_output
			[True],						# continue_reasoning
			9,							# expected_message_count
			[							# expected_roles
				MessageRole.SYSTEM, 					# system prompt
				MessageRole.USER, MessageRole.ASSISTANT, 	    # in-context example 1
				MessageRole.USER, MessageRole.ASSISTANT, 	    # in-context example 2
				MessageRole.USER, MessageRole.ASSISTANT, 	    # in-context example 3
				MessageRole.USER, 						# main request
				MessageRole.ASSISTANT					# assistant response to main request
			],
			[							# expected_final_message_content
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
""".strip()
			],
			None,						# expected_error
			id="multiple_demos_with_previous_content_no_interventions_no_internal_reasoning",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				MathField.MATH_PROBLEM: "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			(  							# previous_content
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
""".strip()
			),
			[							# internal_reasoning_for_output
				"I should divide 6 by 3",
				"I should handle the division outside of the parentheses",
			],
			[							# prefix_for_output
				"6 / 3 =", "6 divided by 3 is"
			],
			[True, True],				# continue_reasoning (both true)
			3,							# expected_message_count
			[							# expected_roles
				MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT
			],
			(  							# expected_final_message_content
				[
					f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
""".strip() + "\n" + f"""
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should divide 6 by 3
## {MathField.MATH_OPERATION}
6 / 3 =
""".strip(),
					f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should handle the division outside of the parentheses
## {MathField.MATH_OPERATION}
6 divided by 3 is
""".strip(),
				]
			),
			None,						# expected_error
			id="no_demos_with_previous_content_multiple_interventions_continue_reasoning_true",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				MathField.MATH_PROBLEM: "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
""".strip(),							# previous_content
			[							# internal_reasoning_for_output
				"I should divide 6 by 3 and add that to 5",
				"Now I will add the term after the parentheses",
			],
			[							# prefix_for_output
				"5 + 6 / 3 =", "The answer is"
			],
			[False, False],				# continue_reasoning (both false)
			3,							# expected_message_count
			[							# expected_roles
				MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT
			],
			[							# expected_final_message_content
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
{XMLTag.THINKING_END}
{XMLTag.ANSWER_START}
## {MathField.ANSWER}
5 + 6 / 3 =
""".strip(),
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
{XMLTag.THINKING_END}
{XMLTag.ANSWER_START}
## {MathField.ANSWER}
The answer is
""".strip(),
			],
			None,						# expected_error
			id="no_demos_with_previous_content_multiple_interventions_continue_reasoning_false",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				MathField.MATH_PROBLEM: "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			(  							# previous_content
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
""".strip()
			),
			[							# internal_reasoning_for_output
				"I should divide 6 by 3",
				"I should handle the division outside of the parentheses",
			],
			[							# prefix_for_output
				"6 / 3 =", "The answer is"
			],
			[True, False],				# continue_reasoning (mixed)
			3,							# expected_message_count
			[							# expected_roles
				MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT
			],
			(  							# expected_final_message_content
				[
					f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should divide 6 by 3
## {MathField.MATH_OPERATION}
6 / 3 =
""".strip(),
					f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should start by simplifying the term in the parentheses.
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
{XMLTag.THINKING_END}
{XMLTag.ANSWER_START}
## {MathField.ANSWER}
The answer is
""".strip()
				]
			),
			None,						# expected_error
			id="no_demos_with_previous_content_multiple_interventions_continue_reasoning_mixed",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				MathField.MATH_PROBLEM: "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			(  							# previous_content
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
""".strip()
			),
			[							# internal_reasoning_for_output
				"I should divide 6 by 3",
				"Now I will add the term after the parentheses",
			],
			[							# prefix_for_output
				"6 / 3 =", "Six divided by three is"
			],
			[True, True],				# continue_reasoning (both true)
			3,							# expected_message_count
			[							# expected_roles
				MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT
			],
			(  							# expected_final_message_content
				[						# Two reasoning steps with internal reasoning provided
					f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I should divide 6 by 3
## {MathField.MATH_OPERATION}
6 / 3 =
""".strip(),
					f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
10 / 2 = 5
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
Now I will add the term after the parentheses
## {MathField.MATH_OPERATION}
Six divided by three is
""".strip()
				]
			),
			None,						# expected_error
			id="no_demos_with_previous_content_multiple_interventions_continue_reasoning_with_internal_reasoning",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				MathField.MATH_PROBLEM: "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			"",							# previous_content (empty)
			[							# internal_reasoning_for_output (both empty)
				"",
				"",
			],
			[							# prefix_for_output
				"6 / 3 =", "Now I will add"
			],
			[True, True],				# continue_reasoning (both true)
			3,							# expected_message_count
			[							# expected_roles
				MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT
			],
			[							# expected_final_message_content
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
6 / 3 =
""".strip(),
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
Now I will add
""".strip(),
			],
			None,						# expected_error
			id="no_demos_no_previous_content_multiple_interventions_continue_reasoning_both_true_no_internal_reasoning",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				MathField.MATH_PROBLEM: "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			"",							# previous_content (empty)
			[							# internal_reasoning_for_output (both empty)
				"",
				"",
			],
			[							# prefix_for_output
				"6 / 3 =", "The answer is"
			],
			[True, False],				# continue_reasoning (mixed: True, False)
			3,							# expected_message_count
			[							# expected_roles
				MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT
			],
			[							# expected_final_message_content
				# First generation is a reasoning step without previous content or internal reasoning
				f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {MathField.MATH_OPERATION}
6 / 3 =
""".strip(),
				# Second generation is a final answer without previous content or internal reasoning
				f"""
{XMLTag.ANSWER_START}
## {MathField.ANSWER}
The answer is
""".strip(),
			],
			None,						# expected_error
			id="no_demos_no_previous_content_multiple_interventions_continue_reasoning_mixed_no_internal_reasoning",
		),
	pytest.param(
		SolveMathProblemWithReasoning,	# signature
		{								# inputs
			"equation": "What is 2+2?"
		},
		None,							# demos
		None,							# response_length
		True,  							# has_internal_reasoning
		"",								# previous_content
		None,							# internal_reasoning_for_output
		None,							# prefix_for_output
		[[True]],						# continue_reasoning
		2,								# expected_message_count
		[MessageRole.SYSTEM, MessageRole.USER],		# expected_roles
		"",								# expected_final_message_content
		AssertionError,					# expected_error
		id="error_wrong_input_field_name",
	),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs (too many fields)
				MathField.MATH_PROBLEM: "What is 2+2?",
				"subject": "arithmetic",
				"difficulty": 2,
			},							# inputs
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			"",							# previous_content
			None,						# internal_reasoning_for_output
			None,						# prefix_for_output
			[True],						# continue_reasoning
			2,							# expected_message_count
			[MessageRole.SYSTEM, MessageRole.USER],	# expected_roles
			"",							# expected_final_message_content
			AssertionError,				# expected_error
			id="error_too_many_input_fields",
		),
	],
)
def test_format_single_trajectory_with_interventions(
	vllm_generator_adapter: VLLMGeneratorAdapter,
	signature: Any,
	inputs: dict[str, Any],
	demos: list[dict[str, Any]] | None,
	response_length: ResponseLength | None,
	has_internal_reasoning: bool,
	previous_content: str,
	internal_reasoning_for_output: str | list[str],
	prefix_for_output: str | list[str],
	continue_reasoning: bool | list[bool],
	expected_message_count: int,
	expected_roles: list[str],
	expected_final_message_content: list[str],
	expected_error: type[Exception] | None,
) -> None:
	"""
	Verify that the `format_single_trajectory_with_interventions` method produces the correct output structure.

	Args:
		vllm_generator_adapter: The VLLMGeneratorAdapter instance to test.
		signature: The signature class to use for formatting.
		inputs: The inputs to format in the user message.
		demos: Optional list of demo examples.
		response_length: Optional response length constraints.
		has_internal_reasoning: Whether internal reasoning guidance is provided.
		previous_content: Previous content to continue reasoning from.
		internal_reasoning_for_output: Internal reasoning for the output section.
		prefix_for_output: Prefix for the output section.
		continue_reasoning: Whether to continue reasoning or switch to answer section.
		expected_message_count: Expected number of messages in the output.
		expected_roles: Expected roles in the output.
		expected_final_message_content: The expected content of the final (assistant) message for
			each resulting trajectory.
		expected_error: Expected error message.
	"""
	if expected_error is not None:
		# TODO[P3]: Check against an expected error message rather than simply the type of exception.
		with pytest.raises(expected_error):
			vllm_generator_adapter.format_single_trajectory_with_interventions(
				signature=signature,
				inputs=inputs,
				demos=demos,
				response_length=response_length,
				has_internal_reasoning=has_internal_reasoning,
				previous_content=previous_content,
				internal_reasoning_for_output=internal_reasoning_for_output,
				prefix_for_output=prefix_for_output,
				continue_reasoning=continue_reasoning,
			)
	else:
		result: list[list[dict[str, Any]]] = (
			vllm_generator_adapter.format_single_trajectory_with_interventions(
				signature=signature,
				inputs=inputs,
				demos=demos,
				response_length=response_length,
				has_internal_reasoning=has_internal_reasoning,
				previous_content=previous_content,
				internal_reasoning_for_output=internal_reasoning_for_output,
				prefix_for_output=prefix_for_output,
				continue_reasoning=continue_reasoning,
			)
		)
		num_interventions = len(expected_final_message_content)
		assert len(result) == num_interventions, \
			f"Expected {num_interventions} trajectories, but got {len(result)}"
		for trajectory_index in range(num_interventions):
			trajectory = result[trajectory_index]
			expected_content = expected_final_message_content[trajectory_index]
			assert len(trajectory) == expected_message_count, \
				f"Trajectory #{trajectory_index} should have {expected_message_count} messages, but got {len(trajectory)}."
			assert [msg[MessageKey.ROLE] for msg in trajectory] == expected_roles
			final_message = trajectory[-1]
			# We are only interested in verifying the final (assistant) message.
			if expected_roles[-1] == MessageRole.ASSISTANT:
				assert final_message[MessageKey.CONTENT].strip() == expected_content.strip()

			# Check user message content based on whether demos are provided
			# Find the user message (should be the last USER message before the assistant message)
			user_message = None
			if final_message[MessageKey.ROLE] == MessageRole.ASSISTANT and len(trajectory) > 1:
				user_message = trajectory[-2]
			elif final_message[MessageKey.ROLE] == MessageRole.USER:
				user_message = final_message

			if user_message is not None and user_message[MessageKey.ROLE] == MessageRole.USER:
				user_message_content = user_message[MessageKey.CONTENT]
				substrings_in_main_user_message = ["To produce", "Structure your response"]
				# When demos are None or empty list, main_request=False, so should NOT include task guidance
				if demos is None or (isinstance(demos, list) and len(demos) == 0):
					for substring in substrings_in_main_user_message:
						assert substring not in user_message_content, (
							f"User message should not include {substring} when no demos are provided"
						)
				else:
					# When demos are provided, main_request=True, so should include task guidance
					for substring in substrings_in_main_user_message:
						assert substring in user_message_content, (
							f"User message should include {substring} when demos are provided"
						)


class TestFormatMainMethod:
	"""
	Test the main format method.

	This test checks that the main format method correctly formats the batch with
	the specified parameters.
	"""

	@pytest.fixture
	def adapter(self):
		return VLLMGeneratorAdapter()

	@pytest.mark.parametrize(
		# Parameters:
		[
			"signature",
			"inputs",
			"demos",
			"continue_reasoning",
			"internal_reasoning_for_output",
			"prefix_for_output",
			"previous_content",
			"expected_roles_per_trajectory",
			"expected_tag_counts",
		],
		[
			# Basic cases without interventions (no internal_reasoning_for_output or prefix_for_output)
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					MathField.MATH_PROBLEM: "What is 2+2?"
				},
				[],						# demos (empty)
				[[True]],				# continue_reasoning
				None,					# internal_reasoning_for_output (empty)
				None,					# prefix_for_output (empty)
				None,					# previous_content
				[						# expected_roles_per_trajectory
					MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT
				],
				[{						# expected_tag_counts
				# There is no intervention, so the assistant message is empty.
					XMLTag.THINKING_START: 1,			# started reasoning section
					XMLTag.THINKING_END: 0,			# did not end reasoning section
					XMLTag.STEP_START: 1,				# started the first reasoning step
					XMLTag.STEP_END: 0,				# did not complete the first reasoning step
					XMLTag.ANSWER_START: 0,			# did not start the answer section
					XMLTag.ANSWER_END: 0,				# did not complete the answer section
					ControllerOutputParameters.INTERNAL_REASONING: 0,			# no internal reasoning provided
				}],
				id="solve_math_empty_demos_bool_true_no_interventions_continue_reasoning_true",
			),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					MathField.MATH_PROBLEM: "What is 5*3?"
				},
			[{							# demos (single demo)
				ReasoningState.INPUT: {
					MathField.MATH_PROBLEM: "What is 1+1?"
				},
				ReasoningState.REASONING: [
					{
						ControllerOutputParameters.INTERNAL_REASONING: "I need to add 1 and 1.",
						MathField.MATH_OPERATION: "1+1=2",
					},
				],
				ReasoningState.OUTPUT: {
					ControllerOutputParameters.INTERNAL_REASONING: "The answer is clearly 2.",
					MathField.ANSWER: "2",
				},
			}],
				[[False]],  			# continue_reasoning
				None,					# internal_reasoning_for_output (empty)
				None,					# prefix_for_output (empty)
				None,					# previous_content
				[						# expected_roles_per_trajectory
					MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT, MessageRole.USER, MessageRole.ASSISTANT
				],
				[{						# expected_tag_counts
				# When continue_reasoning is False, but there are no interventions,
				# we prepend the <answer> tag to the assistant so that it immediately
				# starts the answer section.
					XMLTag.THINKING_START: 0,			# started thinking section
					XMLTag.THINKING_END: 0,			# ended thinking section
					XMLTag.STEP_START: 0,				# no reasoning
					XMLTag.STEP_END: 0,				# no reasoning
					XMLTag.ANSWER_START: 1,			# answer section started
					XMLTag.ANSWER_END: 0,				# answer section not completed
					ControllerOutputParameters.INTERNAL_REASONING: 0,			# no internal reasoning provided
				}],
				id="solve_math_single_demo_bool_false_no_interventions_continue_reasoning_false",
			),
			# Basic cases with no interventions, but with previous content
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					MathField.MATH_PROBLEM: "What is 7 * (12 / 4 + 3)?"
				},
				[],						# demos (empty)
				[[True]],  				# continue_reasoning
				None,					# internal_reasoning_for_output (empty)
				None,					# prefix_for_output (empty)
				[						# previous_content
					f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I need to divide 12 by 4
## {MathField.MATH_OPERATION}
12 / 4 = 3
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
Now I should add 3 to the result
## {MathField.MATH_OPERATION}
3 + 3 = 6
{XMLTag.STEP_END}
""".strip(),
				],
				[						# expected_roles_per_trajectory
					MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT
				],
				[{						# expected_tag_counts
					XMLTag.THINKING_START: 1,			# reasoning section started
					XMLTag.THINKING_END: 0,			# reasoning section not ended
					XMLTag.STEP_START: 3,				# two existing steps, one new step
					XMLTag.STEP_END: 2,				# two existing steps completed, one new step not completed
					XMLTag.ANSWER_START: 0,			# answer section not started
					XMLTag.ANSWER_END: 0,				# answer section not completed
					ControllerOutputParameters.INTERNAL_REASONING: 2,			# existing internal reasoning provided
				}],
				id="solve_math_single_intervention_continue_reasoning",
			),
			# Multiple trajectories with no intervention, but with previous content.
			# Mix of continue_reasoning and no continue_reasoning.
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					MathField.MATH_PROBLEM: "What is 7 * (12 / 4 + 3)?"
				},
				[{  					# demos (multiple demos)
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 1+1?"},
				ReasoningState.REASONING: [
					{
						ControllerOutputParameters.INTERNAL_REASONING: "I need to add 1 and 1.",
						MathField.MATH_OPERATION: "1+1=2",
					}
				],
				ReasoningState.OUTPUT: {
					ControllerOutputParameters.INTERNAL_REASONING: "The answer is clearly 2.",
					MathField.ANSWER: "2",
				},
			},
			{
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 3*3+4?"},
				ReasoningState.REASONING: [
					{
						ControllerOutputParameters.INTERNAL_REASONING: "Multiply 3 by itself.",
						MathField.MATH_OPERATION: "3*3=9",
					},
					{
						ControllerOutputParameters.INTERNAL_REASONING: "Add 9 and 4.",
						MathField.MATH_OPERATION: "9+4=13",
					}
				],
				ReasoningState.OUTPUT: {
					ControllerOutputParameters.INTERNAL_REASONING: "The answer is 13.",
					MathField.ANSWER: "13",
				},
			}],
				[[True], [False]],  	# continue_reasoning (list)
				[						# internal_reasoning_for_output (list)
					["Next, I need to add 3 to the result"],
					["Now I need to compute 3 + 3, and then multiply the result by 7"]
				],
				[						# prefix_for_output (list)
					["3 + 3 ="],
					["7 * (3 + 3) ="],
				],
				[
					f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
The first step is to divide 12 by 4
## {MathField.MATH_OPERATION}
12 / 4 = 3
{XMLTag.STEP_END}
""".strip(),							# previous_content for trajectory 1
					f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
First, I need to divide 12 by 4.
## {MathField.MATH_OPERATION}
12 / 4 = 3
{XMLTag.STEP_END}
""".strip(),							# previous_content for trajectory 2
				],
				[						# expected_roles_per_trajectory
					MessageRole.SYSTEM, 					# system prompt
					MessageRole.USER, 						# demo #1 input
					MessageRole.ASSISTANT, 				# demo #1 output
					MessageRole.USER, 						# demo #2 input
					MessageRole.ASSISTANT, 				# demo #2 output
					MessageRole.USER, 						# user request (example input)
					MessageRole.ASSISTANT, 				# assistant response (example output)
				],
				[
					{								# expected_tag_counts for trajectory 1
						XMLTag.THINKING_START: 1,		# reasoning section started
						XMLTag.THINKING_END: 0,		# reasoning section not ended
						XMLTag.STEP_START: 2,			# one existing step, one new step
						XMLTag.STEP_END: 1,			# one existing step completed, one new step not completed
						XMLTag.ANSWER_START: 0,		# answer section not started
						XMLTag.ANSWER_END: 0,			# answer section not completed
						ControllerOutputParameters.INTERNAL_REASONING: 2,		# existing internal reasoning + intervention provided
					},
					{								# expected_tag_counts for trajectory 2
						XMLTag.THINKING_START: 1,		# reasoning section started
						XMLTag.THINKING_END: 1,		# reasoning section ended
						XMLTag.STEP_START: 1,			# one new step started
						XMLTag.STEP_END: 1,			# one new step completed
						XMLTag.ANSWER_START: 1,		# answer section started
						XMLTag.ANSWER_END: 0,			# answer section not completed
						ControllerOutputParameters.INTERNAL_REASONING: 1,		# existing internal reasoning (intervention for answer section not included)
					}
				],
				id="solve_math_multiple_trajectories_mixed_continue_reasoning",
			),
			# No past context, one intervention
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					MathField.MATH_PROBLEM: "What is 12/4?"
				},
				[],						# demos (empty)
				[[True]],  				# continue_reasoning
				[						# internal_reasoning_for_output
					["I need to perform division"]
				],
				[["12 / 4 ="]],			# prefix_for_output
				None,					# previous_content (empty)
				[						# expected_roles_per_trajectory
					MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT
				],
				[{									# expected_tag_counts
					XMLTag.THINKING_START: 1,			# reasoning section started
					XMLTag.THINKING_END: 0,			# reasoning section not ended
					XMLTag.STEP_START: 1,				# one new step being started
					XMLTag.STEP_END: 0,				# step not completed
					XMLTag.ANSWER_START: 0,			# no answer section
					XMLTag.ANSWER_END: 0,				# no answer section
					ControllerOutputParameters.INTERNAL_REASONING: 1,			# internal reasoning provided
				}],
				id="solve_math_no_past_context_single_intervention_continue",
			),
			# One trajectory (without past context), one intervention
			pytest.param(
				GenerateArgumentWithReasoning,		# signature
				{						# inputs
					ArgumentField.TOPIC: "renewable energy", ArgumentField.STANCE: ArgumentStance.PRO.value
				},
				[{  					# demos (single demo)
					ReasoningState.INPUT: {ArgumentField.TOPIC: "renewable energy", ArgumentField.STANCE: ArgumentStance.PRO.value},
					ReasoningState.REASONING: [
						{
							ControllerOutputParameters.INTERNAL_REASONING: "I need to highlight the benefits of renewable energy.",
							ArgumentField.CLAIM: "Renewable energy reduces carbon emissions",
						}
					],
					ReasoningState.OUTPUT: {
						ControllerOutputParameters.INTERNAL_REASONING: "I need to synthesize my reasoning into a strong argument.",
						ArgumentField.ARGUMENT: "Renewable energy is crucial for combating climate change by significantly reducing carbon emissions.",
					},
				}],
				[[False]],  			# continue_reasoning
				[						# internal_reasoning_for_output
					["Good for the environment"]
				],
				[						# prefix_for_output
					["Renewable energy is"]
				],
				None,					# previous_content (empty)
				[						# expected_roles_per_trajectory
					MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT, MessageRole.USER, MessageRole.ASSISTANT
				],
				[{									# expected_tag_counts
					XMLTag.THINKING_START: 0,			# started reasoning section
					XMLTag.THINKING_END: 0,			# ended reasoning section
					XMLTag.STEP_START: 0,				# no new steps (going to answer)
					XMLTag.STEP_END: 0,				# no step completion
					XMLTag.ANSWER_START: 1,			# answer section started
					XMLTag.ANSWER_END: 0,				# answer not completed
					ControllerOutputParameters.INTERNAL_REASONING: 0,			# internal reasoning for answer section not included
				}],
				id="generate_argument_no_past_context_single_intervention_final_answer",
			),

			# One trajectory (with past context), one intervention
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					MathField.MATH_PROBLEM: "What is 15 + 8?"
				},
				[{  					# demos (two demos)
					ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 1+1?"},
					ReasoningState.REASONING: [
						{
							ControllerOutputParameters.INTERNAL_REASONING: "I need to add 1 and 1.",
							MathField.MATH_OPERATION: "1+1=2",
						}
					],
					ReasoningState.OUTPUT: {
						ControllerOutputParameters.INTERNAL_REASONING: "The answer is clearly 2.",
						MathField.ANSWER: "2",
					},
				},
				{
					ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 3*3+4?"},
					ReasoningState.REASONING: [
						{
							ControllerOutputParameters.INTERNAL_REASONING: "Multiply 3 by itself.",
							MathField.MATH_OPERATION: "3*3=9",
						},
						{
							ControllerOutputParameters.INTERNAL_REASONING: "Add 9 and 4.",
							MathField.MATH_OPERATION: "9+4=13",
						}
					],
					ReasoningState.OUTPUT: {
						ControllerOutputParameters.INTERNAL_REASONING: "The answer is 13.",
						MathField.ANSWER: "13",
					},
				}],
				[[True]],  				# continue_reasoning
				[						# internal_reasoning_for_output
					["Now I'll add these numbers"]
				],
				[["15 + 8 ="]],			# prefix_for_output
				[						# previous_content
					f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I need to add 15 and 8
## {MathField.MATH_OPERATION}
Let me identify the numbers first
{XMLTag.STEP_END}
""".strip(),
				],
				[						# expected_roles_per_trajectory
					MessageRole.SYSTEM, 					# system prompt
					MessageRole.USER, 						# demo #1 input
					MessageRole.ASSISTANT, 				# demo #1 output
					MessageRole.USER, 						# demo #2 input
					MessageRole.ASSISTANT, 				# demo #2 output
					MessageRole.USER, 						# user request (example input)
					MessageRole.ASSISTANT, 				# assistant response (example output)
				],
				[{									# expected_tag_counts
					XMLTag.THINKING_START: 1,			# reasoning section started
					XMLTag.THINKING_END: 0,			# reasoning section not ended
					XMLTag.STEP_START: 2,				# one existing step, one new step
					XMLTag.STEP_END: 1,				# one existing step completed, new step not completed
					XMLTag.ANSWER_START: 0,			# no answer section
					XMLTag.ANSWER_END: 0,				# no answer section
					ControllerOutputParameters.INTERNAL_REASONING: 2,			# existing internal reasoning + intervention provided
				}],
				id="solve_math_single_past_context_single_intervention_continue",
			),
			# One trajectory (with past context), one intervention (final answer)
			pytest.param(
				GenerateArgumentWithReasoning,		# signature
				{						# inputs
					ArgumentField.TOPIC: "space exploration", ArgumentField.STANCE: ArgumentStance.ANTI.value
				},
				[],						# demos (empty)
				[[False]],  			# continue_reasoning
				[						# internal_reasoning_for_output
					["I should talk about cost and risk"]
				],
				[						# prefix_for_output
					["Space exploration is"]
				],
				[						# previous_content
					f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ArgumentField.CLAIM}
Space exploration is too expensive and risky
{XMLTag.STEP_END}
{XMLTag.STEP_START}
## {ArgumentField.CLAIM}
The cost of a single mission is outrageous
{XMLTag.STEP_END}
""".strip(),
				],
				[						# expected_roles_per_trajectory
					MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT
				],
				[{						# expected_tag_counts
					XMLTag.THINKING_START: 1,			# reasoning section not started (going to answer)
					XMLTag.THINKING_END: 1,			# reasoning section not ended
					XMLTag.STEP_START: 2,				# two existing steps from previous content
					XMLTag.STEP_END: 2,				# two existing steps completed from previous content
					XMLTag.ANSWER_START: 1,			# answer section started
					XMLTag.ANSWER_END: 0,				# answer not completed
					ControllerOutputParameters.INTERNAL_REASONING: 0,			# intervention for answer section not included
				}],
				id="generate_argument_single_past_context_single_intervention_final_answer",
			),
			# One trajectory (with past context), multiple interventions
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					MathField.MATH_PROBLEM: "What is 15+8-3?"
				},
				[{  					# demos (single demo)
					ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 1+1?"},
					ReasoningState.REASONING: [
						{
							ControllerOutputParameters.INTERNAL_REASONING: "I need to add 1 and 1.",
							MathField.MATH_OPERATION: "1+1=2",
						}
					],
					ReasoningState.OUTPUT: {
						ControllerOutputParameters.INTERNAL_REASONING: "The answer is clearly 2.",
						MathField.ANSWER: "2",
					},
				}],
				[[True, True, False]],  # continue_reasoning (list[list]) -> single trajectory with multiple interventions
				[						# internal_reasoning_for_output (list[list]) -> single trajectory with multiple interventions
					[
						"Next, I will subtract 3 from the result",
						"I have to subtract 3 from the result",
						"I can compute the final result"
					]
				],
				[						# prefix_for_output (list[list]) -> single trajectory with multiple interventions
					[
						"23 - 3 =",
						"Subtracting 3 from 23 yields",
						"The answer is",
					]
				],
				[						# previous_content
					f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
I will start with the first addition
## {MathField.MATH_OPERATION}
15 + 8 = 23
{XMLTag.STEP_END}
""".strip()
				],
				[						# expected_roles_per_trajectory
					MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT, MessageRole.USER, MessageRole.ASSISTANT
				],
				[
					{					# expected_tag_counts for intervention 1
						XMLTag.THINKING_START: 1,		# reasoning section started
						XMLTag.THINKING_END: 0,		# reasoning section has not ended
						XMLTag.STEP_START: 2,			# one existing step, one new step
						XMLTag.STEP_END: 1,			# one existing step completed, new step not completed
						XMLTag.ANSWER_START: 0,		# no answer section
						XMLTag.ANSWER_END: 0,			# no answer section
						ControllerOutputParameters.INTERNAL_REASONING: 2,		# existing internal reasoning + intervention provided
					},
					{								# expected_tag_counts for intervention 2
						XMLTag.THINKING_START: 1,		# reasoning section started
						XMLTag.THINKING_END: 0,		# reasoning section has not ended
						XMLTag.STEP_START: 2,			# one existing step, one new step
						XMLTag.STEP_END: 1,			# one existing step completed, new step not completed
						XMLTag.ANSWER_START: 0,		# no answer section
						XMLTag.ANSWER_END: 0,			# no answer section
						ControllerOutputParameters.INTERNAL_REASONING: 2,		# existing internal reasoning + intervention provided
					},
					{								# expected_tag_counts for intervention 3
						XMLTag.THINKING_START: 1,		# reasoning section started
						XMLTag.THINKING_END: 1,		# reasoning section ended
						XMLTag.STEP_START: 1,			# one existing step (thinking ends)
						XMLTag.STEP_END: 1,			# one existing step completed
						XMLTag.ANSWER_START: 1,		# answer section started
						XMLTag.ANSWER_END: 0,			# answer not completed
						ControllerOutputParameters.INTERNAL_REASONING: 1,		# existing internal reasoning (intervention for answer section not included)
					}
				],
				id="solve_math_single_past_context_multiple_interventions",
			),
			# Multiple trajectories (with past context), single intervention per trajectory
			pytest.param(
				GenerateArgumentWithReasoning,		# signature
				{						# inputs
					ArgumentField.TOPIC: "artificial intelligence", ArgumentField.STANCE: ArgumentStance.PRO.value
				},
				[{  					# demos (multiple demos)
					ReasoningState.INPUT: {ArgumentField.TOPIC: "renewable energy", ArgumentField.STANCE: ArgumentStance.PRO.value},
					ReasoningState.REASONING: [
						{
							ControllerOutputParameters.INTERNAL_REASONING: "I need to highlight the benefits of renewable energy.",
							ArgumentField.CLAIM: "Renewable energy reduces carbon emissions",
						}
					],
					ReasoningState.OUTPUT: {
						ControllerOutputParameters.INTERNAL_REASONING: "I need to synthesize my reasoning into a strong argument.",
						ArgumentField.ARGUMENT: "Renewable energy is crucial for combating climate change by significantly reducing carbon emissions.",
					},
				},
				{
					ReasoningState.INPUT: {ArgumentField.TOPIC: "nuclear power", ArgumentField.STANCE: ArgumentStance.ANTI.value},
					ReasoningState.REASONING: [
						{
							ControllerOutputParameters.INTERNAL_REASONING: "I need to argue against nuclear power.",
							ArgumentField.CLAIM: "Nuclear waste poses long-term risks",
						}
					],
					ReasoningState.OUTPUT: {
						ControllerOutputParameters.INTERNAL_REASONING: "I need to synthesize my reasoning into a strong argument.",
						"argument": "Nuclear power creates radioactive waste that remains dangerous for thousands of years.",
					},
				}],
				[[True], [False]],  	# continue_reasoning
				None,					# internal_reasoning_for_output (empty because `use_internal_reasoning_for_thought_generation` is False)
				[						# prefix_for_output (list)
					["For example, "],
					["Generally, "]
				],
				[						# previous_content (list)
					f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ArgumentField.CLAIM}
AI advances medical research significantly
{XMLTag.STEP_END}
""".strip(),
					f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ArgumentField.CLAIM}
AI improves efficiency across industries
{XMLTag.STEP_END}
""".strip(),
				],
				[						# expected_roles_per_trajectory
					MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT, MessageRole.USER, MessageRole.ASSISTANT, MessageRole.USER, MessageRole.ASSISTANT
				],
				[						# expected tag counts
					{								# expected_tag_counts for trajectory 1
						XMLTag.THINKING_START: 1,		# reasoning section started
						XMLTag.THINKING_END: 0,		# reasoning section has not ended
						XMLTag.STEP_START: 2,			# one existing step, one new step
						XMLTag.STEP_END: 1,			# one existing step completed, new step not completed
						XMLTag.ANSWER_START: 0,		# no answer section
						XMLTag.ANSWER_END: 0,			# no answer section
						ControllerOutputParameters.INTERNAL_REASONING: 0,		# no internal reasoning provided
					},
					{								# expected_tag_counts for trajectory 2
						XMLTag.THINKING_START: 1,		# reasoning section started
						XMLTag.THINKING_END: 1,		# reasoning section has ended
						XMLTag.STEP_START: 1,			# one existing step (thinking ends)
						XMLTag.STEP_END: 1,			# one existing step completed
						XMLTag.ANSWER_START: 1,		# answer section started
						XMLTag.ANSWER_END: 0,			# answer not completed
						ControllerOutputParameters.INTERNAL_REASONING: 0,		# no internal reasoning provided
					}
				],
				id="generate_argument_multiple_past_contexts_single_intervention_each",
			),
			# Multiple trajectories (with past context), multiple interventions per trajectory
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					MathField.MATH_PROBLEM: "What is (4*5)+(3*2)?"
				},
				[],						# demos (empty)
				[						# continue_reasoning (list of lists - 2 interventions for traj1, 3 for traj2)
					[True, False], [True, True, False]
				],
				[						# internal_reasoning_for_output (list of lists)
					[
						"Now let me calculate 3 * 2",
						"I just need to add the result of the second parenthesis to get the final answer"
					],
					[
						"Now let me calculate 4 * 5",
						"Now let me calculate the first parenthesis",
						"I just need to add the result of the first parenthesis to get the final answer"
					]
				],
				[						# prefix_for_output (list of lists)
					[
						"3 * 2 =",
						"20 + (3 * 2) ="
					],
					[
						"4 * 5 =",
						"(4 * 5) =",
						"(4 * 5) + 6 =",
					]
				],
				[						# previous_content (list)
					f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
Let me start with the first parenthesis
## {MathField.MATH_OPERATION}
4*5 = 20
{XMLTag.STEP_END}
""".strip(),
					f"""
{XMLTag.THINKING_START}
{XMLTag.STEP_START}
## {ControllerOutputParameters.INTERNAL_REASONING}
Let me start with the second parenthesis
## {MathField.MATH_OPERATION}
3*2 = 6
{XMLTag.STEP_END}
""".strip(),
				],
				[						# expected_roles_per_trajectory
					MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT
				],
				[						# expected tag counts
					{								# expected_tag_counts for trajectory 1, intervention 1
						XMLTag.THINKING_START: 1,		# reasoning section started
						XMLTag.THINKING_END: 0,		# reasoning section has not ended
						XMLTag.STEP_START: 2,			# one existing step, one new step
						XMLTag.STEP_END: 1,			# one existing step completed, new step not completed
						XMLTag.ANSWER_START: 0,		# no answer section
						XMLTag.ANSWER_END: 0,			# no answer section
						ControllerOutputParameters.INTERNAL_REASONING: 2,		# existing internal reasoning + intervention provided
					},
					{								# expected_tag_counts for trajectory 1, intervention 2
						XMLTag.THINKING_START: 1,		# reasoning section started
						XMLTag.THINKING_END: 1,		# reasoning section has ended
						XMLTag.STEP_START: 1,			# one existing step (thinking ends)
						XMLTag.STEP_END: 1,			# one existing step completed
						XMLTag.ANSWER_START: 1,		# answer section started
						XMLTag.ANSWER_END: 0,			# answer not completed
						ControllerOutputParameters.INTERNAL_REASONING: 1,		# existing internal reasoning (intervention for answer section not included)
					},
					{								# expected_tag_counts for trajectory 2, intervention 1
						XMLTag.THINKING_START: 1,		# reasoning section started
						XMLTag.THINKING_END: 0,		# reasoning section has not ended
						XMLTag.STEP_START: 2,			# one existing step, one new step
						XMLTag.STEP_END: 1,			# one existing step completed, new step not completed
						XMLTag.ANSWER_START: 0,		# no answer section
						XMLTag.ANSWER_END: 0,			# no answer section
						ControllerOutputParameters.INTERNAL_REASONING: 2,		# existing internal reasoning + intervention provided
					},
					{								# expected_tag_counts for trajectory 2, intervention 2
						XMLTag.THINKING_START: 1,		# reasoning section started
						XMLTag.THINKING_END: 0,		# reasoning section has not ended
						XMLTag.STEP_START: 2,			# one existing step, one new step
						XMLTag.STEP_END: 1,			# one existing step completed, new step not completed
						XMLTag.ANSWER_START: 0,		# no answer section
						XMLTag.ANSWER_END: 0,			# no answer section
						ControllerOutputParameters.INTERNAL_REASONING: 2,		# existing internal reasoning + intervention provided
					},
					{								# expected_tag_counts for trajectory 2, intervention 3
						XMLTag.THINKING_START: 1,		# reasoning section started
						XMLTag.THINKING_END: 1,		# reasoning section has ended
						XMLTag.STEP_START: 1,			# one existing step (thinking ends)
						XMLTag.STEP_END: 1,			# one existing step completed
						XMLTag.ANSWER_START: 1,		# answer section started
						XMLTag.ANSWER_END: 0,			# answer not completed
						ControllerOutputParameters.INTERNAL_REASONING: 1,		# existing internal reasoning (intervention for answer section not included)
					}
				],
				id="solve_math_multiple_past_contexts_multiple_interventions_each",
			),

		],
	)
	def test_format(
		self,
		adapter: VLLMGeneratorAdapter,
		signature: type[ReasoningSignature],
		inputs: dict[str, Any],
		demos: list[dict[str, Any]],
		continue_reasoning: list[list[bool]],
		internal_reasoning_for_output: list[list[str]],
		prefix_for_output: list[list[str]],
		previous_content: list[str],
		expected_roles_per_trajectory: list[str],
		expected_tag_counts: list[dict[str, int]],
	) -> None:
		"""
		Test formatting with single input and no interventions - parameterized.

		This test covers multiple combinations of:
		- Two different signatures (SolveMathProblemWithReasoning, GenerateArgumentWithReasoning)
		- Three demo configurations (empty, single demo, multiple demos)
		- Two different inputs per signature
		- Various has_internal_reasoning and continue_reasoning parameter combinations

		Args:
		    adapter: The VLLMGeneratorAdapter instance to test.
		    signature: The signature to use for formatting.
		    inputs: Input dictionary for the signature.
		    demos: Demo list to use for in-context examples.
		    continue_reasoning: Whether to continue reasoning (boolean or list).
		    previous_content: Previous content to continue from.
		    expected_roles_per_trajectory: Expected message roles (same for all trajectories).
		    expected_tag_counts: Expected counts of tags in final assistant message.
		"""
		result = adapter.format(
			signature=signature,
			inputs=inputs,
			demos=demos,
			previous_content=previous_content,
			continue_reasoning=continue_reasoning,
			internal_reasoning_for_output=internal_reasoning_for_output,
			prefix_for_output=prefix_for_output,
		)
		assert len(result) == len(expected_tag_counts)
		for trajectory_index, expected_tag_counts_for_trajectory in enumerate(expected_tag_counts):
			formatted_trajectory: list[dict[str, Any]] = result[trajectory_index]
			actual_roles: list[str] = [msg[MessageKey.ROLE] for msg in formatted_trajectory]
			assert actual_roles == expected_roles_per_trajectory
			if expected_roles_per_trajectory[-1] == MessageRole.ASSISTANT: 	# We expect the last message to be the assistant message
				for tag, expected_count in expected_tag_counts_for_trajectory.items():
					assert expected_count == formatted_trajectory[-1][MessageKey.CONTENT].count(tag)

	@pytest.mark.parametrize(
		"error_scenario,expected_error",
		[
			pytest.param(
				{
					"inputs": {MathField.MATH_PROBLEM: "test"},
					"demos": [[]],
					ControllerContinueReasoningChoice.CONTINUE_REASONING: [[True]],
				},
				AttributeError,
				id="list_of_lists_demos_unsupported",
			),
			pytest.param(
				{
					"inputs": {MathField.MATH_PROBLEM: "test"},
					"previous_content": [["a", "b"]],
					ControllerContinueReasoningChoice.CONTINUE_REASONING: [[True]],
				},
				NotImplementedError,
				id="previous_content_list_of_lists_unsupported",
			),
			# New: inputs as list is not supported
			pytest.param(
				{
					"inputs": [{MathField.MATH_PROBLEM: "test"}],
					ControllerContinueReasoningChoice.CONTINUE_REASONING: [[True]],
				},
				NotImplementedError,
				id="inputs_list_unsupported",
			),
			# New: continue_reasoning empty list invalid when used in parsing context
			pytest.param(
				{
					"inputs": {MathField.MATH_PROBLEM: "test"},
					ControllerContinueReasoningChoice.CONTINUE_REASONING: [],
				},
				ValueError,
				id="empty_continue_reasoning_list_invalid",
			),
			# New: mismatched keys between signature.input_fields and inputs should assert
			pytest.param(
				{
					"inputs": {},
					ControllerContinueReasoningChoice.CONTINUE_REASONING: [[True]],
				},
				AssertionError,
				id="inputs_missing_required_field",
			),
		],
	)
	# TODO[P3]: Make sure to also test against the expected error message.
	def test_format_error_scenarios(
		self,
		adapter: VLLMGeneratorAdapter,
		error_scenario: dict,
		expected_error: type[Exception],
	) -> None:
		"""Test that format method raises appropriate errors for invalid inputs.

		Args:
			adapter: VLLMGeneratorAdapter under test.
			error_scenario: Keyword arguments passed to adapter.format that trigger errors.
			expected_error: Exception type expected to be raised.
		"""
		with pytest.raises(expected_error):
			adapter.format(signature=SolveMathProblemWithReasoning, **error_scenario)


class TestBatchFormatterErrorCases:
	"""Test error cases and edge conditions."""

	@pytest.fixture
	def adapter(self):
		return VLLMGeneratorAdapter()

	@pytest.fixture
	def signature(self):
		return QuestionAnsweringWithReasoning

class TestCallMethod:
	"""Tests functionality of VLLMGeneratorAdapter.__call__ using a mock LocalVLLM."""

	@pytest.fixture
	def adapter(self) -> VLLMGeneratorAdapter:
		return VLLMGeneratorAdapter()

	@pytest.mark.parametrize(
		[
			"signature",
			"inputs",
			"previous_content",
			"continue_reasoning",
			"internal_reasoning_for_output",
			"prefix_for_output",
			"mock_responses",
			"expected_outputs",
			"expected_error",
			"lm_kwargs",
		],
		[
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					MathField.MATH_PROBLEM: "Compute (8/2) + 3"
				},
				None,					# previous_content: two trajectories starting fresh
				[[True], [True]],		# continue_reasoning (two trajectories, each continues)
				None,					# internal_reasoning_for_output (two trajectories, one intervention each)
				None,					# prefix_for_output (two trajectories, one intervention each)
			[						# mock_responses: 1 layer, 2 requests (batched trajectories), n=2
				[
					[
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\ninternal reasoning 1\n## {MathField.MATH_OPERATION}\nmath operation 1",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\ninternal reasoning 2\n## {MathField.MATH_OPERATION}\nmath operation 2",
					],
					[
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\ninternal reasoning 3\n## {MathField.MATH_OPERATION}\nmath operation 3",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\ninternal reasoning 4\n## {MathField.MATH_OPERATION}\nmath operation 4",
					],
				],
			],
			[						# expected_outputs
				[
					{MathField.MATH_OPERATION: "math operation 1"},
					{MathField.MATH_OPERATION: "math operation 2"},
				],
				[
					{MathField.MATH_OPERATION: "math operation 3"},
					{MathField.MATH_OPERATION: "math operation 4"},
				],
			],
			None,					# expected_error
			[						# lm_kwargs as list of dicts (high-temp sampling, n=2)
				{"temperature": 0.1, "n": 2},
				{"temperature": 0.8, "n": 2}
			],
			id="two_traj_no_prev_both_continue_with_internal_reasoning",
			),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					MathField.MATH_PROBLEM: "Compute (8/2) + 3"
				},
				[						# previous_content
					f"{XMLTag.THINKING_START}\n{XMLTag.STEP_START}\n## {MathField.MATH_OPERATION}\nprevious step 1\n{XMLTag.STEP_END}",
					f"{XMLTag.THINKING_START}\n{XMLTag.STEP_START}\n## {MathField.MATH_OPERATION}\nprevious step 2\n{XMLTag.STEP_END}",
				],
				[[True], [True]],		# continue_reasoning (two trajectories, each continues)
				None,					# internal_reasoning_for_output (two trajectories, one intervention each)
				None,					# prefix_for_output (two trajectories, one intervention each)
			[						# mock_responses: 1 layer, 2 requests (batched trajectories), n=2
				[
					[
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\ninternal reasoning 5\n## {MathField.MATH_OPERATION}\nmath operation 5",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\ninternal reasoning 6\n## {MathField.MATH_OPERATION}\nmath operation 6",
					],
					[
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\ninternal reasoning 7\n## {MathField.MATH_OPERATION}\nmath operation 7",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\ninternal reasoning 8\n## {MathField.MATH_OPERATION}\nmath operation 8",
					],
				],
			],
			[						# expected_outputs
				[
					{MathField.MATH_OPERATION: "math operation 5"},
					{MathField.MATH_OPERATION: "math operation 6"},
				],
				[
					{MathField.MATH_OPERATION: "math operation 7"},
					{MathField.MATH_OPERATION: "math operation 8"},
				],
		],
		None,						# expected_error
		[							# lm_kwargs as list of dicts (medium-temp sampling, n=2)
			{"temperature": 0.8, "n": 2},
			{"temperature": 0.8, "n": 2},
		],
		id="two_traj_with_prev_both_continue_with_internal_reasoning",
		),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					MathField.MATH_PROBLEM: "Compute (8/2) + 3"
				},
				None,					# previous_content: two trajectories starting fresh
				[[True], [False]],		# continue_reasoning: first continues, second finishes
				[						# internal_reasoning_for_output
					["reasoning 1"],
					["reasoning 2"]
				],
				[						# prefix_for_output
					["prefix 1"],
					["prefix 2"]
				],
				[						# mock_responses: 1 layer, 2 requests (batched trajectories)
				[
					[
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 1\n## {MathField.MATH_OPERATION}\nprefix 1 continuation",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 1\n## {MathField.MATH_OPERATION}\nprefix 1 continuation",
					],
					[
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 2\n## {MathField.ANSWER}\nprefix 2 continuation",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 2\n## {MathField.ANSWER}\nprefix 2 continuation",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 2\n## {MathField.ANSWER}\nprefix 2 continuation",
					],
				],
			],
			[						# expected_outputs
				[
					{MathField.MATH_OPERATION: "prefix 1 continuation"},
					{MathField.MATH_OPERATION: "prefix 1 continuation"},
				],
				[
					{MathField.ANSWER: "prefix 2 continuation"},
					{MathField.ANSWER: "prefix 2 continuation"},
					{MathField.ANSWER: "prefix 2 continuation"},
				],
			],
			None,					# expected_error
			[						# lm_kwargs as list of list of dicts (medium-temp sampling, n=2 for first trajectory, n=3 for second trajectory)
				[{"temperature": 0.8, "n": 2}],
				[{"temperature": 0.8, "n": 3}],
			],
			id="two_traj_no_prev_mixed_continue_false_with_internal_reasoning",
			),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					MathField.MATH_PROBLEM: "Compute (8/2) + 3"
				},
				None,					# previous_content: two trajectories starting fresh
				[[True], [False]],		# continue_reasoning: first continues, second finishes
				None,					# internal_reasoning_for_output: no interventions
				None,					# prefix_for_output: no interventions
			[						# mock_responses: 1 layer, 2 requests (batched trajectories)
				[
					[
						" 9",
						" 10",
					],
					[
						"\nanswer 1",
						"\nanswer 2",
						"\nanswer 3",
					],
				],
			],
			[						# expected_outputs
				[
					{MathField.MATH_OPERATION: "9"},
					{MathField.MATH_OPERATION: "10"},
				],
				[
					{MathField.ANSWER: "answer 1"},
					{MathField.ANSWER: "answer 2"},
					{MathField.ANSWER: "answer 3"},
				],
			],
			None,					# expected_error
			[						# lm_kwargs as list of list of dicts (medium-temp sampling, n=2 for first trajectory, n=3 for second trajectory)
				[{"temperature": 0.8, "n": 2}],
				[{"temperature": 0.8, "n": 3}],
			],
			id="two_traj_no_prev_mixed_continue_false_no_interventions",
			),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					MathField.MATH_PROBLEM: "Compute (8/2) + 3"
				},
				[						# previous_content
					f"{XMLTag.THINKING_START}\n{XMLTag.STEP_START}\n## {MathField.MATH_OPERATION}\nprevious step 3\n{XMLTag.STEP_END}",
					f"{XMLTag.THINKING_START}\n{XMLTag.STEP_START}\n## {MathField.MATH_OPERATION}\nprevious step 4\n{XMLTag.STEP_END}",
				],
				[[False], [False]],		# continue_reasoning: both finish
				[						# internal_reasoning_for_output
					["reasoning 3"], ["reasoning 4"]
				],
				[						# prefix_for_output
					["prefix 3"],
					["prefix 4"]
				],
			[						# mock_responses: 1 layer, 2 requests (batched trajectories)
				[
					[
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 3\n## {MathField.ANSWER}\nprefix 3 continuation",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 3\n## {MathField.ANSWER}\nprefix 3 continuation",
					],
					[
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 4\n## {MathField.ANSWER}\nprefix 4 continuation",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 4\n## {MathField.ANSWER}\nprefix 4 continuation",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 4\n## {MathField.ANSWER}\nprefix 4 continuation",
					],
				],
			],
				[						# expected_outputs
					[
						{MathField.ANSWER: "prefix 3 continuation"},
						{MathField.ANSWER: "prefix 3 continuation"},
					],
					[
						{MathField.ANSWER: "prefix 4 continuation"},
						{MathField.ANSWER: "prefix 4 continuation"},
						{MathField.ANSWER: "prefix 4 continuation"},
					],
				],
				None,					# expected_error
				[						# lm_kwargs as list of list of dicts (medium-temp sampling, n=2 for first trajectory, n=3 for second trajectory)
					[{"temperature": 0.8, "n": 2}],
					[{"temperature": 0.8, "n": 3}],
				],
				id="two_traj_with_prev_both_finish_with_internal_reasoning",
			),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					MathField.MATH_PROBLEM: "Compute (8/2) + 3"
				},
				[						# previous_content
					f"{XMLTag.THINKING_START}\n{XMLTag.STEP_START}\n## {MathField.MATH_OPERATION}\nprevious step 5\n{XMLTag.STEP_END}",
					f"{XMLTag.THINKING_START}\n{XMLTag.STEP_START}\n## {MathField.MATH_OPERATION}\nprevious step 6\n{XMLTag.STEP_END}",
				],
				[[False], [False]],		# continue_reasoning: both finish
				None,					# internal_reasoning_for_output: no interventions
				None,					# prefix_for_output: no interventions
			[						# mock_responses: 1 layer, 2 requests (batched trajectories)
				[
					[
						"\nanswer 4",
						"\nanswer 5",
					],
					[
						"\nanswer 6",
						"\nanswer 7",
						"\nanswer 8",
					],
				],
			],
				[						# expected_outputs
					[
						{MathField.ANSWER: "answer 4"},
						{MathField.ANSWER: "answer 5"},
					],
					[
						{MathField.ANSWER: "answer 6"},
						{MathField.ANSWER: "answer 7"},
						{MathField.ANSWER: "answer 8"},
					],
				],
				None,					# expected_error
				[						# lm_kwargs as list of list of dicts (medium-temp sampling, n=2 for first trajectory, n=3 for second trajectory)
					[{"temperature": 0.8, "n": 2}],
					[{"temperature": 0.8, "n": 3}],
				],
				id="two_traj_with_prev_both_finish_no_interventions",
			),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					MathField.MATH_PROBLEM: "Compute (8/2) + 3"
				},
				None,					# previous_content: single trajectory starting fresh
				[[True, False, True]],	# continue_reasoning: 3 interventions (continue, finish, continue)
				[						# internal_reasoning_for_output
					["reasoning 5", "reasoning 6", "reasoning 7"]
				],
				[						# prefix_for_output
					["prefix 5", "prefix 6", "prefix 7"]
				],
			[						# mock_responses: 1 layer, 3 requests (all interventions batched), n=2
				[
					[
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 5\n## {MathField.MATH_OPERATION}\nprefix 5 continuation",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 5\n## {MathField.MATH_OPERATION}\nprefix 5 continuation",
					],
					[
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 6\n## {MathField.ANSWER}\nprefix 6 continuation",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 6\n## {MathField.ANSWER}\nprefix 6 continuation",
					],
					[
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 7\n## {MathField.MATH_OPERATION}\nprefix 7 continuation",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 7\n## {MathField.MATH_OPERATION}\nprefix 7 continuation",
					],
				],
			],
				[						# expected_outputs
					[
						{MathField.MATH_OPERATION: "prefix 5 continuation"},
						{MathField.MATH_OPERATION: "prefix 5 continuation"},
					],
					[
						{MathField.ANSWER: "prefix 6 continuation"},
						{MathField.ANSWER: "prefix 6 continuation"},
					],
					[
						{MathField.MATH_OPERATION: "prefix 7 continuation"},
						{MathField.MATH_OPERATION: "prefix 7 continuation"},
					],
				],
				None,					# expected_error
				[						# lm_kwargs as list of dicts (medium-temp sampling, n=2)
					{"temperature": 0.8, "n": 2},
				],
				id="single_traj_three_interventions_mixed_with_internal_reasoning",
			),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					MathField.MATH_PROBLEM: "Compute (8/2) + 3"
				},
				None,					# previous_content: single trajectory starting fresh
				[[True, False, True]],	# continue_reasoning: 3 interventions
				None,					# internal_reasoning_for_output: no interventions
				None,					# prefix_for_output: no interventions
			[						# mock_responses: 1 layer, 3 requests (all interventions batched), n=2
				[
					[
						"\nmath operation 11",
						"\nmath operation 12",
					],
					[
						f"\n{MathField.ANSWER} 9",
						f"\n{MathField.ANSWER} 10",
					],
					[
						"\nmath operation 13",
						"\nmath operation 14",
					],
				],
			],
				[						# expected_outputs
					[
						{MathField.MATH_OPERATION: "math operation 11"},
						{MathField.MATH_OPERATION: "math operation 12"},
					],
					[
						{MathField.ANSWER: "answer 9"},
						{MathField.ANSWER: "answer 10"},
					],
					[
						{MathField.MATH_OPERATION: "math operation 13"},
						{MathField.MATH_OPERATION: "math operation 14"},
					],
				],
				None,					# expected_error
				[						# lm_kwargs as list of dicts (medium-temp sampling, n=2)
					{"temperature": 0.8, "n": 2},
				],
				id="single_traj_three_interventions_mixed_no_interventions",
			),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				MathField.MATH_PROBLEM: "Compute (8/2) + 3"
			},
			None,						# previous_content: two trajectories starting fresh
				[						# continue_reasoning: different patterns per trajectory
					[True, False], [False, True]
				],
				[						# internal_reasoning_for_output
					["reasoning 8", "reasoning 9"],
					["reasoning 10", "reasoning 11"]
				],
				[						# prefix_for_output
					["prefix 8", "prefix 9"],
					["prefix 10", "prefix 11"]
				],
			[						# mock_responses: 1 layer, 4 requests (2 traj x 2 interventions batched)
				[
					[
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 8\n## {MathField.MATH_OPERATION}\nprefix 8 continuation",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 8\n## {MathField.MATH_OPERATION}\nprefix 8 continuation",
					],
					[
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 9\n## {MathField.ANSWER}\nprefix 9 continuation",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 9\n## {MathField.ANSWER}\nprefix 9 continuation",
					],
					[
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 10\n## {MathField.ANSWER}\nprefix 10 continuation",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 10\n## {MathField.ANSWER}\nprefix 10 continuation",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 10\n## {MathField.ANSWER}\nprefix 10 continuation",
					],
					[
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 11\n## {MathField.MATH_OPERATION}\nprefix 11 continuation",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 11\n## {MathField.MATH_OPERATION}\nprefix 11 continuation",
						f"## {ControllerOutputParameters.INTERNAL_REASONING}\nreasoning 11\n## {MathField.MATH_OPERATION}\nprefix 11 continuation",
					],
				],
			],
				[						# expected_outputs
					[
						{MathField.MATH_OPERATION: "prefix 8 continuation"},
						{MathField.MATH_OPERATION: "prefix 8 continuation"},
					],
					[
						{MathField.ANSWER: "prefix 9 continuation"},
						{MathField.ANSWER: "prefix 9 continuation"},
					],
					[
						{MathField.ANSWER: "prefix 10 continuation"},
						{MathField.ANSWER: "prefix 10 continuation"},
						{MathField.ANSWER: "prefix 10 continuation"},
					],
					[
						{MathField.MATH_OPERATION: "prefix 11 continuation"},
						{MathField.MATH_OPERATION: "prefix 11 continuation"},
						{MathField.MATH_OPERATION: "prefix 11 continuation"},
					],
				],
				None,					# expected_error
				[						# lm_kwargs as list of list of dicts (medium-temp sampling, n=2 for first trajectory, n=3 for second trajectory)
					[
						{"temperature": 0.8, "n": 2},		# lm_kwargs for traj 1, intervention 1
						{"temperature": 0.8, "n": 2},		# lm_kwargs for traj 1, intervention 2
					],
					[
						{"temperature": 0.8, "n": 3},		# lm_kwargs for traj 2, intervention 1
						{"temperature": 0.8, "n": 3},		# lm_kwargs for traj 2, intervention 2
					],
				],
				id="two_traj_two_interventions_each_mixed_with_internal_reasoning",
			),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					MathField.MATH_PROBLEM: "Compute (8/2) + 3"
				},
				None,					# previous_content: two trajectories starting fresh
				[						# continue_reasoning: different patterns per trajectory
					[True, False], [False, True]
				],
				None,					# internal_reasoning_for_output: no interventions
				None,					# prefix_for_output: no interventions
			[						# mock_responses: 1 layer, 4 requests (2 traj x 2 interventions batched)
				[
					[
						"\nmath operation 15",
						"\nmath operation 16",
					],
					[
						f"\n{MathField.ANSWER} 11",
						f"\n{MathField.ANSWER} 12",
					],
					[
						f"\n{MathField.ANSWER} 13",
						f"\n{MathField.ANSWER} 14",
						f"\n{MathField.ANSWER} 15",
					],
					[
						"\nmath operation 17",
						"\nmath operation 18",
						"\nmath operation 19",
					],
				],
			],
				[
					[
						{MathField.MATH_OPERATION: "math operation 15"},
						{MathField.MATH_OPERATION: "math operation 16"},
					],
					[
						{MathField.ANSWER: "answer 11"},
						{MathField.ANSWER: "answer 12"},
					],
					[
						{MathField.ANSWER: "answer 13"},
						{MathField.ANSWER: "answer 14"},
						{MathField.ANSWER: "answer 15"},
					],
					[
						{MathField.MATH_OPERATION: "math operation 17"},
						{MathField.MATH_OPERATION: "math operation 18"},
						{MathField.MATH_OPERATION: "math operation 19"},
					],
				],
				None,					# expected_error
				[						# lm_kwargs as list of list of dicts (medium-temp sampling, n=2 for first trajectory, n=3 for second trajectory)
					[
						{"temperature": 0.8, "n": 2},		# lm_kwargs for traj 1, intervention 1
						{"temperature": 0.8, "n": 2},		# lm_kwargs for traj 1, intervention 2
					],
					[
						{"temperature": 0.8, "n": 3},		# lm_kwargs for traj 2, intervention 1
						{"temperature": 0.8, "n": 3},		# lm_kwargs for traj 2, intervention 2
					],
				],
				id="two_traj_two_interventions_each_mixed_no_interventions",
			),
		],
	)
	def test_call(
		self,
		adapter: VLLMGeneratorAdapter,
		signature: type[ReasoningSignature],
		inputs: dict[str, Any],
		previous_content: str | list[str],
		continue_reasoning: bool | list[bool] | list[list[bool]],
		internal_reasoning_for_output: str | list[str] | list[list[str]],
		prefix_for_output: str | list[str] | list[list[str]],
		mock_responses: list[str] | list[list[str]],
		expected_outputs: list[list[dict[str, Any]]] | None,
		expected_error: type[Exception] | None,
		lm_kwargs: dict[str, Any],
	) -> None:
		"""Comprehensive parameterized test for VLLMGeneratorAdapter.__call__.

		This test asserts exact parsed outputs (per conversation and completion) when using
		a mock LLM, and verifies expected exceptions when inputs or outputs are invalid.
		"""
		lm = MockGenerativeLocalVLLM(mock_responses)
		if expected_error is not None:
			with pytest.raises(expected_error):
				adapter(
					signature=signature,
					lm=lm,
					inputs=inputs,
					lm_kwargs=lm_kwargs,
					previous_content=previous_content,
					continue_reasoning=continue_reasoning,
					internal_reasoning_for_output=internal_reasoning_for_output,
					prefix_for_output=prefix_for_output,
				)
		else:
			result = adapter(
				signature=signature,
				lm=lm,
				inputs=inputs,
				lm_kwargs=lm_kwargs,
				previous_content=previous_content,
				continue_reasoning=continue_reasoning,
				internal_reasoning_for_output=internal_reasoning_for_output,
				prefix_for_output=prefix_for_output,
			)
			assert expected_outputs is not None
			assert len(result) == len(expected_outputs)
			for i, expected_candidates in enumerate(expected_outputs):
				assert len(result[i]) == len(expected_candidates)
				for j, expected_parsed in enumerate(expected_candidates):
					assert result[i][j] == expected_parsed

class TestVLLMGeneratorAdapterParsing:
	"""Integration tests for parse() method of VLLMGeneratorAdapter."""

	@pytest.fixture
	def adapter(self) -> VLLMGeneratorAdapter:
		return VLLMGeneratorAdapter()

	@pytest.mark.parametrize([
		"signature", "mock_response", "continue_reasoning",
		"expected_fields", "expected_error"
	], [
		# JSON reasoning response scenarios
		pytest.param(
			SolveMathProblemWithReasoning,
			f'{{"{ControllerOutputParameters.INTERNAL_REASONING}": "I need to solve this step by step", "{MathField.MATH_OPERATION}": "First, I will compute 2 + 2 = 4"}}',
			True,
			{
						ControllerOutputParameters.INTERNAL_REASONING: "I need to solve this step by step",
				MathField.MATH_OPERATION: "First, I will compute 2 + 2 = 4",
			},
			None,
			id="parse_json_reasoning"
		),
		pytest.param(
			SolveMathProblemWithReasoning,
			f'{{"{MathField.ANSWER}": "The answer is 4"}}',
			False,
			{MathField.ANSWER: "The answer is 4"},
			None,
			id="parse_json_answer"
		),
		# Field header response scenarios
		pytest.param(
			SolveMathProblemWithReasoning,
			f"""## {ControllerOutputParameters.INTERNAL_REASONING}
I need to think about this problem carefully
## {MathField.MATH_OPERATION}
Let me compute 5 * 6 = 30""",
			True,
			{MathField.MATH_OPERATION: "Let me compute 5 * 6 = 30"},
			None,
			id="parse_headers_reasoning"
		),
		pytest.param(
			SolveMathProblemWithReasoning,
			f"""## {MathField.ANSWER}
The final solution is 30""",
			False,
			{MathField.ANSWER: "The final solution is 30"},
			None,
			id="parse_headers_final_answer"
		),
		# Mixed format scenarios
		pytest.param(
			SolveMathProblemWithReasoning,
			f"""## {ControllerOutputParameters.INTERNAL_REASONING}
This is my reasoning process
## {MathField.MATH_OPERATION}
{{"result": "Computing the result: 8 / 2 = 4", "value": 4}}""",
			True,
			{MathField.MATH_OPERATION: '{"result": "Computing the result: 8 / 2 = 4", "value": 4}'},
			None,
			id="parse_mixed_header_with_json_content"
		),
		# Error scenarios
		pytest.param(
			SolveMathProblemWithReasoning,
			'{"INTERNAL_REASONING": "Missing closing brace", "MATH_OPERATION": "2+2=4"}',
			True,
			None,
			AdapterParseError,
			id="parse_malformed_json_error"
		),
		pytest.param(
			SolveMathProblemWithReasoning,
			'{"wrong_field": "This field is not required", "another_wrong": "Also not needed"}',
			True,
			None,
			AdapterParseError,
			id="parse_missing_required_fields"
		),
		pytest.param(
			SolveMathProblemWithReasoning,
			"",
			True,
			None,
			AdapterParseError,
			id="parse_empty_response_error"
		),
		pytest.param(
			SolveMathProblemWithReasoning,
			f"""## {MathField.MATH_OPERATION}
Only one field provided""",
			False,  # Expecting answer fields but only got math_operation
			None,
			AdapterParseError,
			id="parse_incomplete_headers_error"
		),
	])
	def test_parse_method_scenarios(
		self,
		adapter: VLLMGeneratorAdapter,
		signature: ReasoningSignature,
		mock_response,
		continue_reasoning,
		expected_fields,
		expected_error
	) -> None:
		"""Test parse() method with various response formats and reasoning contexts."""
		# Set up adapter context to match the scenario
		adapter._current_continue_reasoning = continue_reasoning

		if expected_error is not None:
			with pytest.raises(expected_error):
				adapter.parse(signature, mock_response, parse_reasoning=continue_reasoning)
		else:
			result = adapter.parse(signature, mock_response, parse_reasoning=continue_reasoning)
			assert result == expected_fields

class TestVLLMGeneratorAdapterCallMethod:
	"""Integration tests for __call__ method end-to-end."""
	# TODO[P3]: Make these tests less opaque by using fewer constants or more explicit objects.

	@pytest.fixture
	def adapter(self) -> VLLMGeneratorAdapter:
		return VLLMGeneratorAdapter()

	@pytest.mark.parametrize([
		"signature", "inputs", "continue_reasoning", "mock_vllm_responses",
		"expected_completions_structure", "lm_kwargs"
	], [
		# Single input, reasoning step - 1 layer, 1 request, 1 completion
		pytest.param(
			SolveMathProblemWithReasoning,
			{MathField.MATH_PROBLEM: "What is 2+2?"},
			[[True]],
			[[[
				f"""## {ControllerOutputParameters.INTERNAL_REASONING}
I need to add these numbers together
## {MathField.MATH_OPERATION}
2 + 2 = 4"""
			]]],
			[[{MathField.MATH_OPERATION: "2 + 2 = 4"}]],
			{"temperature": 0.1, "n": 1},
			id="call_single_reasoning_step"
		),
		# Single input, final answer - 1 layer, 1 request, 1 completion
		pytest.param(
			SolveMathProblemWithReasoning,
			{MathField.MATH_PROBLEM: "What is 2+2?"},
			[[False]],
			[[[
				"""\nThe answer is 4"""
			]]],
			[[{"answer": "The answer is 4"}]],
			{"temperature": 0.1, "n": 1},
			id="call_single_final_answer"
		),
		# Multiple reasoning steps - 1 layer, 2 requests (batched), 1 completion each
		pytest.param(
			SolveMathProblemWithReasoning,
			{MathField.MATH_PROBLEM: "Solve 5*6"},
			[[True], [True]],  # Both continue reasoning
			[
				[
					[f"""## {ControllerOutputParameters.INTERNAL_REASONING}
I need to add these numbers together
## {MathField.MATH_OPERATION}
2 + 2 = 4"""],
					[f"""## {ControllerOutputParameters.INTERNAL_REASONING}
I need to add these numbers together
## {MathField.MATH_OPERATION}
2 + 2 = 4"""]
				]
			],
			[[{MathField.MATH_OPERATION: "2 + 2 = 4"}], [{MathField.MATH_OPERATION: "2 + 2 = 4"}]],
			{"temperature": 0.3, "n": 1},
			id="call_multiple_reasoning_steps"
		),
	])
	def test_call_method_integration(self, adapter: VLLMGeneratorAdapter, signature, inputs, continue_reasoning,
	                               mock_vllm_responses, expected_completions_structure, lm_kwargs):
		"""Test __call__ method end-to-end with various parameter combinations."""
		lm = MockGenerativeLocalVLLM(mock_vllm_responses)
		result = adapter(
			signature=signature,
			lm=lm,
			inputs=inputs,
			lm_kwargs=lm_kwargs,
			previous_content=None,
			continue_reasoning=continue_reasoning,
			internal_reasoning_for_output=None,
			prefix_for_output=None,
		)

		assert result == expected_completions_structure


class TestVLLMGeneratorAdapterContextLogic:
	"""Integration tests for context-dependent behavior and stop token determination."""

	@pytest.fixture
	def adapter(self) -> VLLMGeneratorAdapter:
		return VLLMGeneratorAdapter()

	@pytest.mark.parametrize([
		"continue_reasoning_input", "expected_stop_tokens"
	], [
		(True, ["</step>"]),
		(False, ["</answer>"]),
	])
	def test_determine_stop_tokens(self, adapter: VLLMGeneratorAdapter, continue_reasoning_input,
	                               expected_stop_tokens):
		"""Test _determine_stop_tokens for individual boolean inputs."""
		stop_tokens = adapter._determine_stop_tokens(continue_reasoning_input)
		assert stop_tokens == expected_stop_tokens

	@pytest.mark.parametrize(
		["continue_reasoning_input", "expected_stop_tokens_per_message"],
		[
			pytest.param(
				[[True]],
				[[XMLTag.STEP_END]],
				id="single_traj_single_intervention_continue"
			),
			pytest.param(
				[[False]],
				[[XMLTag.ANSWER_END]],
				id="single_traj_single_intervention_stop"
			),
			pytest.param(
				[[True, False]],
				[[XMLTag.STEP_END], [XMLTag.ANSWER_END]],
				id="single_traj_multi_intervention_continue_then_stop"
			),
			pytest.param(
				[[False, True]],
				[[XMLTag.ANSWER_END], [XMLTag.STEP_END]],
				id="single_traj_multi_intervention_stop_then_continue"
			),
			pytest.param(
				[[True], [False]],
				[[XMLTag.STEP_END], [XMLTag.ANSWER_END]],
				id="multi_traj_single_intervention_continue_and_stop"
			),
			pytest.param(
				[[False], [True]],
				[[XMLTag.ANSWER_END], [XMLTag.STEP_END]],
				id="multi_traj_single_intervention_stop_and_continue"
			),
			pytest.param(
				[[True, False], [False, True]],
				[[XMLTag.STEP_END], [XMLTag.ANSWER_END], [XMLTag.ANSWER_END], [XMLTag.STEP_END]],
				id="multi_traj_multi_intervention_mixed"
			),
		]
	)
	def test_batch_stop_tokens_through_call(
		self,
		adapter: VLLMGeneratorAdapter,
		continue_reasoning_input: list[list[bool]],
		expected_stop_tokens_per_message: list[list[str]]
	) -> None:
		"""Test that batch processing correctly determines stop tokens for each message through __call__."""
		# Create mock responses that match whether we're generating reasoning or answer
		# For reasoning steps (True), use math_operation field; for answer steps (False), use answer field
		# Format: list[list[list[str]]] = [num_layers, num_requests, num_completions]
		# All messages are batched in 1 layer
		requests: list[list[str]] = []
		for traj in continue_reasoning_input:
			for continue_val in traj:
				if continue_val:
					# Reasoning step - return reasoning field
					requests.append([f"## {MathField.MATH_OPERATION}\ntest"])
				else:
					# Final answer - return output field
					requests.append([f"## {MathField.ANSWER}\n42"])
		mock_responses: list[list[list[str]]] = [requests]  # 1 layer, N requests

		lm = MockGenerativeLocalVLLM(mock_responses)

		# Call adapter to populate the trajectory mapping
		adapter(
			signature=SolveMathProblemWithReasoning,
			lm=lm,
			inputs={MathField.MATH_PROBLEM: "test"},
			lm_kwargs={"temperature": 0.1, "n": 1},
			continue_reasoning=continue_reasoning_input,
		)

		# Verify that each message index in the trajectory mapping has the correct continue_reasoning value
		total_messages = sum(len(traj) for traj in continue_reasoning_input)
		assert len(adapter._trajectory_continue_reasoning) == total_messages

		# Verify stop tokens for each message
		message_idx = 0
		for traj_idx, traj_continue_reasoning in enumerate(continue_reasoning_input):
			for intervention_idx, _ in enumerate(traj_continue_reasoning):
				expected_stop = expected_stop_tokens_per_message[message_idx][0]
				actual_stop = adapter._determine_stop_tokens(adapter._trajectory_continue_reasoning[message_idx])[0]
				assert actual_stop == expected_stop, \
					f"Message {message_idx} (traj {traj_idx}, intervention {intervention_idx}): " \
					f"expected {expected_stop}, got {actual_stop}"
				message_idx += 1


class TestFormatNormalization:
	"""
	Tests for format method input normalization and tag handling.

	These tests verify that the format method correctly:
	1. Normalizes different input shapes (str, list[str], list[list[str]])
	2. Handles tag placement (especially closing </step> before </thinking>)
	3. Processes single/multiple trajectories with single/multiple interventions
	"""

	@pytest.fixture
	def adapter(self) -> VLLMGeneratorAdapter:
		return VLLMGeneratorAdapter()

	def test_single_intervention_single_trajectory(self, adapter: VLLMGeneratorAdapter):
		"""Simplest case: single trajectory, single intervention."""
		msgs = adapter.format(
			signature=SolveMathProblemWithReasoning,
			inputs={MathField.MATH_PROBLEM: "What is 7+7?"},
			demos=[],
			previous_content=None,
			internal_reasoning_for_output=None,
			prefix_for_output=None,
			continue_reasoning=[[True]],
		)
		assert len(msgs) == 1
		assert isinstance(msgs[0], list)
		assert msgs[0][2]["content"] == f"<thinking>\n<step>\n## {MathField.MATH_OPERATION}"

	def test_format_normalization_list_previous_content_as_trajectories(
		self, adapter: VLLMGeneratorAdapter
	):
		"""previous_content as list[str] -> treated as multiple trajectories (one per string)"""
		prevs = [f"<thinking>\n<step>\n## {MathField.MATH_OPERATION}\n first", f"<thinking>\n<step>\n## {MathField.MATH_OPERATION}\n second"]
		msgs = adapter.format(
			signature=SolveMathProblemWithReasoning,
			inputs={MathField.MATH_PROBLEM: "What is 2+2?"},
			demos=[{
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 1+1?"},
				ReasoningState.REASONING: [
					{
						ControllerOutputParameters.INTERNAL_REASONING: "I need to add 1 and 1.",
						MathField.MATH_OPERATION: "1+1=2",
					}
				],
				ReasoningState.OUTPUT: {
					ControllerOutputParameters.INTERNAL_REASONING: "The answer is clearly 2.",
					MathField.ANSWER: "2",
				},
			}],
			previous_content=prevs,
			internal_reasoning_for_output=None,
			prefix_for_output=None,
			continue_reasoning=[[True], [False]],
		)
		assert isinstance(msgs, list) and len(msgs) == 2
		assert all(isinstance(conv, list) for conv in msgs)

		assert (
			msgs[0][4][MessageKey.CONTENT] == f"{XMLTag.THINKING_START}\n{XMLTag.STEP_START}\n## {MathField.MATH_OPERATION}\n first\n{XMLTag.STEP_END}\n{XMLTag.STEP_START}\n## {MathField.MATH_OPERATION}"
		)  # 4th message as this includes a demo
		assert (
			msgs[1][4][MessageKey.CONTENT]
			== f"{XMLTag.THINKING_START}\n{XMLTag.STEP_START}\n## {MathField.MATH_OPERATION}\n second\n{XMLTag.STEP_END}\n{XMLTag.THINKING_END}\n{XMLTag.ANSWER_START}\n## {MathField.ANSWER}"
		)

	def test_single_trajectory_multiple_interventions(self, adapter: VLLMGeneratorAdapter):
		"""Single trajectory with multiple interventions."""
		internal = [["r1", "r2", "r3"]]
		prefixes = [["p1", "p2", "p3"]]
		continue_flags = [[True, True, False]]
		prev = [f"{XMLTag.THINKING_START}\n{XMLTag.STEP_START}\nprevious context"]

		msgs = adapter.format(
			signature=SolveMathProblemWithReasoning,
			inputs={MathField.MATH_PROBLEM: "What is 5+5?"},
			demos=[],
			previous_content=prev,
			internal_reasoning_for_output=internal,
			prefix_for_output=prefixes,
			continue_reasoning=continue_flags,
		)
		# Expect one conversation per intervention
		assert len(msgs) == 3
		assert all(
			isinstance(conv, list) and conv[0]["role"] == MessageRole.SYSTEM for conv in msgs
		)

		assert (
			msgs[0][2][MessageKey.CONTENT]
			== f"{XMLTag.THINKING_START}\n{XMLTag.STEP_START}\nprevious context\n{XMLTag.STEP_END}\n{XMLTag.STEP_START}\n## {ControllerOutputParameters.INTERNAL_REASONING}\nr1\n## {MathField.MATH_OPERATION}\np1"
		)
		assert (
			msgs[1][2][MessageKey.CONTENT]
			== f"{XMLTag.THINKING_START}\n{XMLTag.STEP_START}\nprevious context\n{XMLTag.STEP_END}\n{XMLTag.STEP_START}\n## {ControllerOutputParameters.INTERNAL_REASONING}\nr2\n## {MathField.MATH_OPERATION}\np2"
		)
		assert (
			msgs[2][2][MessageKey.CONTENT]
			== f"{XMLTag.THINKING_START}\n{XMLTag.STEP_START}\nprevious context\n{XMLTag.STEP_END}\n{XMLTag.THINKING_END}\n{XMLTag.ANSWER_START}\n## {MathField.ANSWER}\np3"
		)

	def test_multiple_trajectories_single_intervention(self, adapter: VLLMGeneratorAdapter):
		"""Multiple trajectories, single intervention each."""
		prevs = [
			f"{XMLTag.THINKING_START}\n{XMLTag.STEP_START}\n## {MathField.MATH_OPERATION}\na",
			f"{XMLTag.THINKING_START}\n{XMLTag.STEP_START}\n## {MathField.MATH_OPERATION}\nb"
		]
		internal = None
		prefixes = None
		continue_flags = [[True], [False]]

		msgs = adapter.format(
			signature=SolveMathProblemWithReasoning,
			inputs={MathField.MATH_PROBLEM: "What is 6+6?"},
			demos=[{
				ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 1+1?"},
				ReasoningState.REASONING: [
					{
						ControllerOutputParameters.INTERNAL_REASONING: "I need to add 1 and 1.",
						MathField.MATH_OPERATION: "1+1=2",
					}
				],
				ReasoningState.OUTPUT: {
					ControllerOutputParameters.INTERNAL_REASONING: "The answer is clearly 2.",
					MathField.ANSWER: "2",
				},
			}],
			previous_content=prevs,
			internal_reasoning_for_output=internal,
			prefix_for_output=prefixes,
			continue_reasoning=continue_flags,
		)

		assert len(msgs) == 2
		assert all(isinstance(conv, list) for conv in msgs)

		assert msgs[0][4][MessageKey.CONTENT] == f"{XMLTag.THINKING_START}\n{XMLTag.STEP_START}\n## {MathField.MATH_OPERATION}\na\n{XMLTag.STEP_END}\n{XMLTag.STEP_START}\n## {MathField.MATH_OPERATION}"
		assert (
			msgs[1][4][MessageKey.CONTENT] == f"{XMLTag.THINKING_START}\n{XMLTag.STEP_START}\n## {MathField.MATH_OPERATION}\nb\n{XMLTag.STEP_END}\n{XMLTag.THINKING_END}\n{XMLTag.ANSWER_START}\n## {MathField.ANSWER}"
		)


if __name__ == "__main__":
	pytest.main([__file__, "-vv"])
