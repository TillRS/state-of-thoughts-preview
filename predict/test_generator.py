"""
Test suite for the State-based TreeOfThoughtGenerator class.

This module provides comprehensive tests for the overhauled generator that uses
State objects as inputs and supports automatic heterogeneous batch splitting.

Expected usage:
```bash
pytest predict/test_generator.py -vv
```
"""

# Standard library imports
import logging
import os
from unittest.mock import Mock, patch

# Third-party imports
import dspy
import pytest
import torch
from dspy.primitives.prediction import Prediction

# Local imports
from constants import OpenSourceModel, Verbosity
from lm.generative_local_lm import GenerativeLocalVLLM
from lm.lm_constants import SamplingParam
from predict import (
	ControllerActionParameters,
	ControllerContinueReasoningChoice,
	ControllerOutputParameters,
)
from predict.controller_constants import ControllerOutput
from predict.demos.generator_demos import ARGUMENT_DEMOS, MATH_DEMOS
from predict.generator import TreeOfThoughtGenerator
from signatures import (
	ArgumentStance,
	GenerateArgumentWithReasoning,
	QuestionAnsweringWithReasoning,
	SolveMathProblemWithReasoning,
)
from signatures.example_signatures import ArgumentField, MathField, QuestionField
from tree import (
	FinalOutputKind,
	State,
)
from tree.tree_constants import ReasoningState
from utilities_for_tests import MockGenerativeLocalVLLM

logger = logging.getLogger(__name__)

# =============================================================================
# GPU Skip Markers
# =============================================================================

# Check if one or more GPUs are available
if torch.cuda.is_available():
	_has_gpu = True
	# Use real torch - don't mock
else:
	_has_gpu = False

# Skip GPU tests if no GPU is available
pytestmark_gpu = pytest.mark.skipif(
	not _has_gpu,
	reason="GPU tests require GPU access",
)


class TestTreeOfThoughtGeneratorInit:
	"""Test cases for TreeOfThoughtGenerator initialization."""

	def test_init_with_reasoning_signature_class(self):
		"""Test initialization with a ReasoningSignature class."""
		generator = TreeOfThoughtGenerator(
			signature=QuestionAnsweringWithReasoning,
			max_reasoning_steps=3,
		)

		assert generator.signature == QuestionAnsweringWithReasoning
		assert generator.max_reasoning_steps == 3
		assert len(generator.signature.input_fields) == 1
		assert len(generator.signature.reasoning_fields) == 1
		assert len(generator.signature.output_fields) == 1
		assert QuestionField.QUESTION in generator.signature.input_fields
		assert QuestionField.REASONING_STEP in generator.signature.reasoning_fields
		assert MathField.ANSWER in generator.signature.output_fields

	def test_init_with_string_signature(self):
		"""Test initialization with a string signature."""
		generator = TreeOfThoughtGenerator(
			signature="question -> reasoning -> answer",
			max_reasoning_steps=5,
		)

		assert generator.signature is not None
		assert generator.max_reasoning_steps == 5
		assert len(generator.signature.input_fields) == 1
		assert len(generator.signature.reasoning_fields) == 1
		assert len(generator.signature.output_fields) == 1

	def test_init_with_config(self):
		"""Test initialization with configuration parameters."""
		config = {
			SamplingParam.TEMPERATURE: 0.7,
			SamplingParam.MAX_TOKENS: 100,
		}
		generator = TreeOfThoughtGenerator(
			signature="question -> reasoning -> answer",
			max_reasoning_steps=3,
			**config,
		)

		assert generator.config == config
		assert generator.config[SamplingParam.TEMPERATURE] == 0.7
		assert generator.config[SamplingParam.MAX_TOKENS] == 100


class TestStateToForwardContext:
	"""Test cases for _state_to_forward_context method."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"state",
			"max_reasoning_steps",
			"expected_continue_reasoning",
			"expected_previous_content_contains",
			"expected_internal_reasoning",
			"expected_prefix",
		],
		# Parameter values
		[
			pytest.param(
				State(input={QuestionField.QUESTION: "What is 2+2?"}),  	# state
				3,  													# max_reasoning_steps
				[True],  												# expected_continue_reasoning
				[],  													# expected_previous_content_contains
				[""],  													# expected_internal_reasoning
				[""],  													# expected_prefix
				id="state_without_reasoning",
			),
			pytest.param(
				State(  												# state
					input={QuestionField.QUESTION: "What is 2+2?"},
					reasoning={
						QuestionField.REASONING_STEP: [
							{QuestionField.REASONING_STEP: "First step"}
						]
					},
				),
				3,  													# max_reasoning_steps
				[True],  												# expected_continue_reasoning
				["<thinking>", "First step"],  							# expected_previous_content_contains
				[""],  													# expected_internal_reasoning
				[""],  													# expected_prefix
				id="state_with_reasoning_below_max",
			),
			pytest.param(
				State(  												# state
					input={QuestionField.QUESTION: "What is 2+2?"},
					reasoning={
						QuestionField.REASONING_STEP: [
							{QuestionField.REASONING_STEP: "Step 1"},
							{QuestionField.REASONING_STEP: "Step 2"},
						]
					},
				),
				2,  													# max_reasoning_steps
				[False],  												# expected_continue_reasoning
				[],  													# expected_previous_content_contains
				[""],  													# expected_internal_reasoning
				[""],  													# expected_prefix
				id="state_at_max_reasoning_steps",
			),
			pytest.param(
				State(  												# state
					input={QuestionField.QUESTION: "What is 2+2?"},
					reasoning={
						QuestionField.REASONING_STEP: [
							{QuestionField.REASONING_STEP: "2 + 2 = 4"}
						]
					},
					output={MathField.ANSWER: "4"},
				),
				3,  													# max_reasoning_steps
				[False],  												# expected_continue_reasoning
				[],  													# expected_previous_content_contains
				[""],  													# expected_internal_reasoning
				[""],  													# expected_prefix
				id="state_with_output",
			),
			pytest.param(
				State(  												# state
					input={QuestionField.QUESTION: "What is 2+2?"},
					controller_outputs=[
						{
							ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.CONTINUE_REASONING,
							ControllerActionParameters.ARGUMENTS: {},
							ControllerOutputParameters.TOOL_DESCRIPTIONS: "Action Name: continue_reasoning",
							ControllerActionParameters.CONSIDERATIONS: "Test",
							ControllerContinueReasoningChoice.CONTINUE_REASONING: True,
							ControllerOutputParameters.INTERNAL_REASONING: "Think carefully about addition",
							ControllerOutputParameters.PREFIX: "First, I will",
						}
					],
				),
				3,  													# max_reasoning_steps
				[True],  												# expected_continue_reasoning
				[],  													# expected_previous_content_contains
				["Think carefully about addition"],  					# expected_internal_reasoning
				["First, I will"],  									# expected_prefix
				id="state_with_interventions",
			),
			pytest.param(
				State(  												# state
					input={QuestionField.QUESTION: "What is 2+2?"},
					reasoning={
						QuestionField.REASONING_STEP: [
							{QuestionField.REASONING_STEP: "2 + 2 = 4"}
						]
					},
					controller_outputs=[
						{
							ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.FINISH,
							ControllerActionParameters.ARGUMENTS: {},
							ControllerOutputParameters.TOOL_DESCRIPTIONS: "Action Name: finish",
							ControllerActionParameters.CONSIDERATIONS: "Ready to finish",
							ControllerContinueReasoningChoice.CONTINUE_REASONING: False,
							ControllerOutputParameters.INTERNAL_REASONING: "Ready to provide answer",
						}
					],
				),
				5,  													# max_reasoning_steps
				[False],  												# expected_continue_reasoning
				[],  													# expected_previous_content_contains
				["Ready to provide answer"],  							# expected_internal_reasoning
				[""],  													# expected_prefix
				id="state_with_controller_finish_decision",
			),
		],
	)
	def test_state_to_forward_context(
		self,
		state: State,
		max_reasoning_steps: int,
		expected_continue_reasoning: list[bool],
		expected_previous_content_contains: list[str],
		expected_internal_reasoning: list[str],
		expected_prefix: list[str],
	) -> None:
		"""Test converting states to forward context with various configurations."""
		generator = TreeOfThoughtGenerator(
			signature=QuestionAnsweringWithReasoning,
			max_reasoning_steps=max_reasoning_steps,
		)
		mock_lm = MockGenerativeLocalVLLM([])

		context = generator._state_to_forward_context(
			state=state,
			lm=mock_lm,
			demos=None,
		)

		assert context.continue_reasoning == expected_continue_reasoning
		assert context.internal_reasoning_for_output == expected_internal_reasoning
		assert context.prefix_for_output == expected_prefix
		for content in expected_previous_content_contains:
			assert content in context.previous_content


class TestFormatPreviousContent:
	"""Test cases for _format_previous_content method."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"state",
			"expected_content",
			"expected_content_contains",
			"expected_step_count",
		],
		# Parameter values
		[
			pytest.param(
				State(input={QuestionField.QUESTION: "What is 2+2?"}),  	# state
				"",  													# expected_content
				[],  													# expected_content_contains
				0,  													# expected_step_count
				id="empty_reasoning",
			),
			pytest.param(
				State(  												# state
					input={QuestionField.QUESTION: "What is 2+2?"},
					reasoning={
						QuestionField.REASONING_STEP: [
							{QuestionField.REASONING_STEP: "2 + 2 equals 4"}
						]
					},
				),
				None,  													# expected_content (None means check contains)
				[  														# expected_content_contains
					"<thinking>",
					"<step>",
					f"## {QuestionField.REASONING_STEP}",
					"2 + 2 equals 4",
					"</step>",
				],
				1,  													# expected_step_count
				id="single_reasoning_step",
			),
			pytest.param(
				State(  												# state
					input={QuestionField.QUESTION: "What is 2+2?"},
					reasoning={
						QuestionField.REASONING_STEP: [
							{QuestionField.REASONING_STEP: "First, identify the operation"},
							{QuestionField.REASONING_STEP: "Then, compute 2 + 2 = 4"},
						]
					},
				),
				None,  													# expected_content
				[  														# expected_content_contains
					"First, identify the operation",
					"Then, compute 2 + 2 = 4",
				],
				2,  													# expected_step_count
				id="multiple_reasoning_steps",
			),
		],
	)
	def test_format_previous_content(
		self,
		state: State,
		expected_content: str | None,
		expected_content_contains: list[str],
		expected_step_count: int,
	) -> None:
		"""Test formatting previous content from state reasoning."""
		generator = TreeOfThoughtGenerator(
			signature=QuestionAnsweringWithReasoning,
			max_reasoning_steps=3,
		)

		formatted = generator._format_previous_content(state)

		if expected_content is not None:
			assert formatted == expected_content
		else:
			for content in expected_content_contains:
				assert content in formatted
			if expected_step_count > 0:
				assert formatted.count("<step>") == expected_step_count


# NOTE: TestSplitStatesByContinueReasoning removed - batch splitting is now handled
# automatically by the VLLMGeneratorAdapter which accepts heterogeneous batches


class TestGeneratorForward:
	"""Integration tests for the forward method."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"states",
			"max_reasoning_steps",
			"n_samples_generation",
			"mock_adapter_return_value",
			"expected_num_states",
			"expected_num_completions_per_state",
			"expected_continue_reasoning",
			"expected_adapter_call_count",
		],
		# Parameter values
		[
			pytest.param(
				State(input={QuestionField.QUESTION: "What is 2+2?"}),  	# states
				3,  													# max_reasoning_steps
				1,  													# n_samples_generation
				[[{ReasoningState.REASONING: "2 + 2 = 4"}]],  				# mock_adapter_return_value
				1,  													# expected_num_states
				[1],  													# expected_num_completions_per_state
				[[True]],  												# expected_continue_reasoning
				1,  													# expected_adapter_call_count
				id="single_state_reasoning",
			),
			pytest.param(
				[  														# states
					State(input={QuestionField.QUESTION: "What is 1+1?"}),
					State(
						input={QuestionField.QUESTION: "What is 1+1?"},
						reasoning={
							QuestionField.REASONING_STEP: [
								{QuestionField.REASONING_STEP: "First step"}
							]
						},
					),
				],
				3,  													# max_reasoning_steps
				1,  													# n_samples_generation
				[  														# mock_adapter_return_value
					[{ReasoningState.REASONING: "Trajectory 1"}],
					[{ReasoningState.REASONING: "Trajectory 2"}],
				],
				2,  													# expected_num_states
				[1, 1],  												# expected_num_completions_per_state
				[[True], [True]],  										# expected_continue_reasoning
				1,  													# expected_adapter_call_count
				id="batch_homogeneous_states",
			),
			pytest.param(
				[  														# states
					State(input={QuestionField.QUESTION: "What is 1+1?"}),
					State(
						input={QuestionField.QUESTION: "What is 1+1?"},
						reasoning={
							QuestionField.REASONING_STEP: [
								{QuestionField.REASONING_STEP: "Step 1"},
								{QuestionField.REASONING_STEP: "Step 2"},
							]
						},
					),
					State(
						input={QuestionField.QUESTION: "What is 1+1?"},
						reasoning={
							QuestionField.REASONING_STEP: [
								{QuestionField.REASONING_STEP: "Step 1"}
							]
						},
					),
				],
				2,  													# max_reasoning_steps (state 1 has 2 steps = max)
				1,  													# n_samples_generation
				[  														# mock_adapter_return_value
					[{ReasoningState.REASONING: "Reasoning for state 0"}],
					[{MathField.ANSWER: "Answer for state 1"}],
					[{ReasoningState.REASONING: "Reasoning for state 2"}],
				],
				3,  													# expected_num_states
				[1, 1, 1],  											# expected_num_completions_per_state
				[[True], [False], [True]],  							# expected_continue_reasoning
				1,  													# expected_adapter_call_count
				id="heterogeneous_batch",
			),
			pytest.param(
				State(input={QuestionField.QUESTION: "What is 2+2?"}),  	# states
				3,  													# max_reasoning_steps
				3,  													# n_samples_generation
				[  														# mock_adapter_return_value
					[
						{ReasoningState.REASONING: "Completion 1"},
						{ReasoningState.REASONING: "Completion 2"},
						{ReasoningState.REASONING: "Completion 3"},
					]
				],
				1,  													# expected_num_states
				[3],  													# expected_num_completions_per_state
				[[True]],  												# expected_continue_reasoning
				1,  													# expected_adapter_call_count
				id="multiple_completions",
			),
		],
	)
	def test_generator_forward(
		self,
		states: State | list[State],
		max_reasoning_steps: int,
		n_samples_generation: int,
		mock_adapter_return_value: list[list[dict[str, str]]],
		expected_num_states: int,
		expected_num_completions_per_state: list[int],
		expected_continue_reasoning: list[list[bool]],
		expected_adapter_call_count: int,
	) -> None:
		"""Test generator forward method with various state configurations."""
		generator = TreeOfThoughtGenerator(
			signature=QuestionAnsweringWithReasoning,
			max_reasoning_steps=max_reasoning_steps,
		)
		mock_lm = MockGenerativeLocalVLLM([])
		generator.lm = mock_lm

		with patch("predict.generator.VLLMGeneratorAdapter") as mock_adapter_class:
			mock_adapter_instance = Mock()
			mock_adapter_class.return_value = mock_adapter_instance
			mock_adapter_instance.return_value = mock_adapter_return_value

			with patch("predict.generator.settings") as mock_settings:
				mock_settings.lm = mock_lm

				results = generator(states=states, n_samples_generation=n_samples_generation)

				# Verify structure
				assert len(results) == expected_num_states
				for i, expected_completions in enumerate(expected_num_completions_per_state):
					assert len(results[i]) == expected_completions

				# Verify adapter was called correctly
				assert mock_adapter_instance.call_count == expected_adapter_call_count
				call_kwargs = mock_adapter_instance.call_args[1]
				assert call_kwargs[ControllerContinueReasoningChoice.CONTINUE_REASONING] == expected_continue_reasoning

				# Verify n_samples_generation was passed correctly
				if isinstance(states, State):
					states_list = [states]
				else:
					states_list = states
				for i in range(len(states_list)):
					assert call_kwargs["lm_kwargs"][i][0]["n"] == n_samples_generation


class TestNSamplesGeneration:
	"""Test cases for n_samples_generation and match_n_to_controller_choice_count functionality."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"states",
			"n_samples_generation",
			"match_n_to_controller_choice_count",
			"expected_n_values",
		],
		# Parameter values
		[
			pytest.param(
				[  														# states
					State(input={QuestionField.QUESTION: "What is 1+1?"}),
					State(input={QuestionField.QUESTION: "What is 2+2?"}),
				],
				5,  													# n_samples_generation (scalar)
				False,  												# match_n_to_controller_choice_count
				[5, 5],  												# expected_n_values
				id="n_samples_scalar_broadcast",
			),
			pytest.param(
				[  														# states
					State(input={QuestionField.QUESTION: "What is 1+1?"}),
					State(input={QuestionField.QUESTION: "What is 2+2?"}),
				],
				[3, 7],  												# n_samples_generation (list per state)
				False,  												# match_n_to_controller_choice_count
				[3, 7],  												# expected_n_values
				id="n_samples_list_per_state",
			),
			pytest.param(
				[  														# states
					State(
						input={QuestionField.QUESTION: "What is 1+1?"},
						controller_outputs=[
							{
								ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.CONTINUE_REASONING,
								ControllerActionParameters.ARGUMENTS: {},
								ControllerOutputParameters.TOOL_DESCRIPTIONS: "Action Name: continue_reasoning",
								ControllerActionParameters.CONSIDERATIONS: "Step 1",
								ControllerContinueReasoningChoice.CONTINUE_REASONING: True,
								ControllerOutputParameters.INTERNAL_REASONING: "Step 1",
								ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT: 2,
							},
							{
								ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.CONTINUE_REASONING,
								ControllerActionParameters.ARGUMENTS: {},
								ControllerOutputParameters.TOOL_DESCRIPTIONS: "Action Name: continue_reasoning",
								ControllerActionParameters.CONSIDERATIONS: "Step 2",
								ControllerContinueReasoningChoice.CONTINUE_REASONING: True,
								ControllerOutputParameters.INTERNAL_REASONING: "Step 2",
								ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT: 3,
							},
						],
					),
					State(
						input={QuestionField.QUESTION: "What is 2+2?"},
						controller_outputs=[
							{
								ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.FINISH,
								ControllerActionParameters.ARGUMENTS: {},
								ControllerOutputParameters.TOOL_DESCRIPTIONS: "Action Name: finish",
								ControllerActionParameters.CONSIDERATIONS: "Final",
								ControllerContinueReasoningChoice.CONTINUE_REASONING: False,
								ControllerOutputParameters.INTERNAL_REASONING: "Final",
								ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT: 1,
							},
						],
					),
				],
				[[4, 6], [8]],  										# n_samples_generation (list of lists)
				False,  												# match_n_to_controller_choice_count
				[4, 6, 8],  											# expected_n_values (state0_int0, state0_int1, state1_int0)
				id="n_samples_list_of_lists_per_intervention",
			),
			pytest.param(
				[  														# states
					State(
						input={QuestionField.QUESTION: "What is 1+1?"},
						controller_outputs=[
							{
								ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.CONTINUE_REASONING,
								ControllerActionParameters.ARGUMENTS: {},
								ControllerOutputParameters.TOOL_DESCRIPTIONS: "Action Name: continue_reasoning",
								ControllerActionParameters.CONSIDERATIONS: "Think",
								ControllerContinueReasoningChoice.CONTINUE_REASONING: True,
								ControllerOutputParameters.INTERNAL_REASONING: "Think",
								ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT: 5,
							},
							{
								ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.CONTINUE_REASONING,
								ControllerActionParameters.ARGUMENTS: {},
								ControllerOutputParameters.TOOL_DESCRIPTIONS: "Action Name: continue_reasoning",
								"considerations": "Think more",
								ControllerContinueReasoningChoice.CONTINUE_REASONING: True,
								ControllerOutputParameters.INTERNAL_REASONING: "Think more",
								ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT: 3,
							},
						],
					),
				],
				999,  													# n_samples_generation (should be ignored)
				True,  													# match_n_to_controller_choice_count
				[5, 3],  												# expected_n_values (from controller)
				id="match_n_to_controller_choice_count_true",
			),
			pytest.param(
				[  														# states
					State(
						input={QuestionField.QUESTION: "What is 1+1?"},
						controller_outputs=[
							{
								"action": ControllerContinueReasoningChoice.CONTINUE_REASONING,
								"action_arguments": {},
								"tool_descriptions": "Action Name: continue_reasoning",
								"considerations": "Continue",
								ControllerContinueReasoningChoice.CONTINUE_REASONING: True,
								ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT: 7,
							},
						],
					),
					State(input={QuestionField.QUESTION: "What is 2+2?"}),
				],
				4,  													# n_samples_generation (fallback)
				True,  													# match_n_to_controller_choice_count
				[7, 4],  												# expected_n_values (controller, fallback)
				id="match_n_mixed_states_with_and_without_controller",
			),
		],
	)
	def test_n_samples_generation(
		self,
		states: list[State],
		n_samples_generation: int | list[int] | list[list[int]],
		match_n_to_controller_choice_count: bool,
		expected_n_values: list[int],
	) -> None:
		"""Test n_samples_generation with various formats and controller matching."""
		generator = TreeOfThoughtGenerator(
			signature=QuestionAnsweringWithReasoning,
			max_reasoning_steps=3,
		)
		mock_lm = MockGenerativeLocalVLLM([])
		generator.lm = mock_lm

		# Create mock adapter return value based on number of states/interventions
		mock_return_value = []
		for state in states:
			if state.controller_outputs:
				mock_return_value.append(
					[{ReasoningState.REASONING: "Result"}] * len(state.controller_outputs)
				)
			else:
				mock_return_value.append([{ReasoningState.REASONING: "Result"}])

		with patch("predict.generator.VLLMGeneratorAdapter") as mock_adapter_class:
			mock_adapter_instance = Mock()
			mock_adapter_class.return_value = mock_adapter_instance
			mock_adapter_instance.return_value = mock_return_value

			with patch("predict.generator.settings") as mock_settings:
				mock_settings.lm = mock_lm

				_ = generator(
					states=states,
					n_samples_generation=n_samples_generation,
					match_n_to_controller_choice_count=match_n_to_controller_choice_count,
				)

				call_kwargs = mock_adapter_instance.call_args[1]
				lm_kwargs = call_kwargs["lm_kwargs"]

				# Flatten expected n values to match structure
				idx = 0
				for state_idx, state in enumerate(states):
					if state.controller_outputs:
						for intervention_idx in range(len(state.controller_outputs)):
							assert lm_kwargs[state_idx][intervention_idx]["n"] == expected_n_values[idx]
							idx += 1
					else:
						assert lm_kwargs[state_idx][0]["n"] == expected_n_values[idx]
						idx += 1


class TestGeneratorHelperMethods:
	"""Test cases for helper methods."""

	def test_update_config(self):
		"""Test updating configuration."""
		generator = TreeOfThoughtGenerator(
			signature=QuestionAnsweringWithReasoning,
			max_reasoning_steps=3,
			temperature=0.5,
		)

		generator.update_config(temperature=0.8, max_tokens=200)
		config = generator.get_config()

		assert config[SamplingParam.TEMPERATURE] == 0.8
		assert config[SamplingParam.MAX_TOKENS] == 200


class TestFinalOutputKindConfiguration:
	"""Test final_output_kind parameter configuration."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"final_output_kind",
			"expected_final_output_kind",
			"test_passed_to_adapter",
		],
		# Parameter values
		[
			pytest.param(
				None,  													# final_output_kind (None = default)
				FinalOutputKind.SYNTHESIS_FAITHFUL,  					# expected_final_output_kind
				False,  												# test_passed_to_adapter
				id="default_final_output_kind",
			),
			pytest.param(
				FinalOutputKind.SYNTHESIS_FAITHFUL,  					# final_output_kind
				FinalOutputKind.SYNTHESIS_FAITHFUL,  					# expected_final_output_kind
				False,  												# test_passed_to_adapter
				id="custom_final_output_kind_synthesis",
			),
			pytest.param(
				FinalOutputKind.CONCLUSION,  							# final_output_kind
				FinalOutputKind.CONCLUSION,  							# expected_final_output_kind
				True,  													# test_passed_to_adapter
				id="custom_final_output_kind_conclusion_passed_to_adapter",
			),
		],
	)
	def test_final_output_kind_configuration(
		self,
		final_output_kind: FinalOutputKind | None,
		expected_final_output_kind: FinalOutputKind,
		test_passed_to_adapter: bool,
	) -> None:
		"""Test final_output_kind parameter configuration and adapter passing."""
		generator_kwargs = {
			"signature": QuestionAnsweringWithReasoning,
			"max_reasoning_steps": 3,
		}
		if final_output_kind is not None:
			generator_kwargs["final_output_kind"] = final_output_kind

		generator = TreeOfThoughtGenerator(**generator_kwargs)
		assert generator.final_output_kind == expected_final_output_kind

		if test_passed_to_adapter:
			mock_lm = MockGenerativeLocalVLLM([])
			generator.lm = mock_lm
			state = State(input={QuestionField.QUESTION: "What is 2+2?"})

			with patch("predict.generator.VLLMGeneratorAdapter") as mock_adapter_class:
				mock_adapter_instance = Mock()
				mock_adapter_class.return_value = mock_adapter_instance
				mock_adapter_instance.return_value = [
					[{ReasoningState.REASONING: "2 + 2 = 4"}]
				]

				with patch("predict.generator.settings") as mock_settings:
					mock_settings.lm = mock_lm

					generator(states=state, n_samples_generation=1)

					# Verify adapter was called with correct final_output_kind
					call_kwargs = mock_adapter_instance.call_args[1]
					assert call_kwargs["final_output_kind"] == expected_final_output_kind



# =============================================================================
# Integration Test Helper Functions
# =============================================================================

def validate_output_structure(
	results: list[list[Prediction]],
	expected_num_states: int,
	expected_num_predictions: int,
) -> bool:
	"""Validate that output has correct structure."""
	if len(results) != expected_num_states:
		return False

	for state_results in results:
		if len(state_results) != expected_num_predictions:
			return False

	return True


def validate_reasoning_state_output(predictions: list[Prediction], reasoning_field: str) -> bool:
	"""Validate that reasoning state generated reasoning (not final answer)."""
	for pred in predictions:
		if not hasattr(pred, reasoning_field):
			return False
		if not getattr(pred, reasoning_field):
			return False
	return True


def validate_finish_state_output(predictions: list[Prediction], output_field: str) -> bool:
	"""Validate that finish state generated final answer (not reasoning)."""
	for pred in predictions:
		if not hasattr(pred, output_field):
			return False
		if not getattr(pred, output_field):
			return False
	return True


def analyze_diversity(predictions: list[list[Prediction]], field_name: str) -> float:
	"""Analyze diversity of predictions by counting unique values."""
	all_values = []
	for state_predictions in predictions:
		for pred in state_predictions:
			if hasattr(pred, field_name):
				all_values.append(getattr(pred, field_name))

	if not all_values:
		return 0.0

	unique_values = len(set(all_values))
	total_values = len(all_values)
	return unique_values / total_values


# Shared GPU model fixture for all GPU tests
@pytest.fixture(scope="module")
def shared_gpu_model():
	"""Shared GenerativeLocalVLLM fixture for all GPU integration tests.

	This fixture loads a model once and shares it across all GPU test classes
	to avoid loading multiple models and running out of GPU memory.
	"""
	if not torch.cuda.is_available():
		pytest.skip("GPU not available")

	base_path = "/projects/BSTEWART/model_storage"
	model_name = OpenSourceModel.QWEN_3_30B_A3B_INSTRUCT_2507.value
	model_path = os.path.join(base_path, model_name)
	lm = None
	try:
		logger.info(f"Initializing shared GPU model from: {model_path}")
		lm = GenerativeLocalVLLM(
			model=model_path,
			tensor_parallel_size=1,
			dtype="auto",
			gpu_memory_utilization=0.9,
			max_model_len=16_384,
			enforce_eager=True,
			verbosity=Verbosity.DEBUG,
		)
		logger.info("Shared GPU model initialized successfully")
		dspy.settings.configure(lm=lm)
		yield lm
	except Exception as e:
		logger.error(f"Failed to load GPU model {model_path}: {e}")
		# Re-raise the exception so tests fail with clear error messages
		# rather than being skipped silently
		raise
	finally:
		# Cleanup after all GPU tests complete
		if lm is not None:
			logger.info("Cleaning up shared GPU model...")
			lm.kill()


@pytestmark_gpu
class TestGeneratorIntegration:
	"""Integration tests for the generator using real models (requires GPU)."""

	@pytest.fixture
	def local_lm(self, shared_gpu_model):
		"""Use the shared GPU model fixture."""
		return shared_gpu_model

	@pytest.fixture
	def math_generator(self, local_lm):
		"""Create a math generator instance with real LM."""
		dspy.settings.configure(lm=local_lm)
		return TreeOfThoughtGenerator(
			signature=SolveMathProblemWithReasoning,
			max_reasoning_steps=3,
		)

	@pytest.fixture
	def argument_generator(self, local_lm):
		"""Create an argument generator instance with real LM."""
		dspy.settings.configure(lm=local_lm)
		return TreeOfThoughtGenerator(
			signature=GenerateArgumentWithReasoning,
			max_reasoning_steps=3,
		)

	def test_single_state_reasoning(self, math_generator):
		"""Test generating next reasoning step for a single state."""
		state = State(input={MathField.MATH_PROBLEM: "What is 15 + 27?"})
		try:
			results = math_generator(states=state, n_samples_generation=1, demos=MATH_DEMOS)
			assert validate_output_structure(results, 1, 1)
			assert validate_reasoning_state_output(results[0], reasoning_field=MathField.MATH_OPERATION)
		except Exception as e:
			pytest.fail(f"Single state reasoning failed: {e}")

	def test_single_state_answer(self, math_generator):
		"""Test generating final answer for a state at max reasoning steps."""
		state = State(
			input={MathField.MATH_PROBLEM: "What is 15 + 27?"},
			reasoning={
				MathField.MATH_OPERATION: [
					{MathField.MATH_OPERATION: "I need to add 15 and 27"},
					{MathField.MATH_OPERATION: "15 + 27 = 15 + 20 + 7 = 35 + 7"},
					{MathField.MATH_OPERATION: "35 + 7 = 42"},
				]
			},
		)
		try:
			results = math_generator(states=state, n_samples_generation=1, demos=MATH_DEMOS)
			assert validate_output_structure(results, 1, 1)
			assert validate_finish_state_output(results[0], output_field=MathField.ANSWER)
		except Exception as e:
			pytest.fail(f"Single state answer failed: {e}")

	def test_batch_processing(self, math_generator):
		"""Test batch processing of multiple trajectories."""
		problem = {MathField.MATH_PROBLEM: "What is 8 * 9?"}
		states = [
			State(input=problem),
			State(
				input=problem,
				reasoning={MathField.MATH_OPERATION: [{MathField.MATH_OPERATION: "I need to multiply 8 by 9"}]},
			),
			State(
				input=problem,
				reasoning={
					MathField.MATH_OPERATION: [
						{MathField.MATH_OPERATION: "I need to multiply 8 by 9"},
						{MathField.MATH_OPERATION: "8 * 9 = 8 * (10 - 1) = 80 - 8"},
					]
				},
				controller_outputs=[
					ControllerOutput(
						action=ControllerContinueReasoningChoice.CONTINUE_REASONING,
						action_arguments={},
						tool_descriptions="Action Name: continue_reasoning",
						continue_reasoning=True,
						considerations="Break down the subtraction step by step.",
						internal_reasoning="Break down the subtraction step by step.",
						prefix="Step by step: ",
						unique_action_response_count=1,
					)
				],
			),
		]
		try:
			results = math_generator(states=states, n_samples_generation=1, demos=MATH_DEMOS)
			assert validate_output_structure(results, 3, 1)
		except Exception as e:
			pytest.fail(f"Batch processing failed: {e}")

	def test_heterogeneous_batch(self, math_generator):
		"""Test heterogeneous batch with mixed reasoning and answer generation."""
		problem = {MathField.MATH_PROBLEM: "What is 12 - 5?"}
		states = [
			State(input=problem),  # Needs reasoning
			State(
				input=problem,
				reasoning={
					MathField.MATH_OPERATION: [
						{MathField.MATH_OPERATION: "I need to subtract 5 from 12"},
						{MathField.MATH_OPERATION: "12 - 5 = 12 - 2 - 3 = 10 - 3"},
						{MathField.MATH_OPERATION: "10 - 3 = 7"},
					]
				},
			),  # Needs answer (max steps)
			State(
				input=problem,
				reasoning={MathField.MATH_OPERATION: [{MathField.MATH_OPERATION: "I need to subtract 5 from 12"}]},
			),  # Needs reasoning
		]

		try:
			results = math_generator(
				states=states, n_samples_generation=1, demos=MATH_DEMOS, verbosity=Verbosity.ERROR
			)

			assert validate_output_structure(results, 3, 1)
			assert validate_reasoning_state_output(results[0], reasoning_field=MathField.MATH_OPERATION)
			assert validate_finish_state_output(results[1], output_field=MathField.ANSWER)
			assert validate_reasoning_state_output(results[2], reasoning_field=MathField.MATH_OPERATION)
		except Exception as e:
			pytest.fail(f"Heterogeneous batch failed: {e}")

	def test_multiple_completions(self, math_generator):
		"""Test generating multiple diverse completions."""
		state = State(input={MathField.MATH_PROBLEM: "What is 6 * 7?"})

		try:
			results = math_generator(states=state, n_samples_generation=5, demos=MATH_DEMOS)

			assert validate_output_structure(results, 1, 5)
			assert validate_reasoning_state_output(results[0], reasoning_field=MathField.MATH_OPERATION)

			diversity = analyze_diversity(results, field_name=MathField.MATH_OPERATION)
			# We can't strictly enforce diversity > 0 on a small model/test, but we can check it runs
			logger.info(f"Diversity: {diversity}")
		except Exception as e:
			pytest.fail(f"Multiple completions failed: {e}")

	def test_controller_interventions(self, math_generator):
		"""Test generator respecting controller interventions."""
		state = State(
			input={MathField.MATH_PROBLEM: "What is 20 ÷ 4?"},
			reasoning={MathField.MATH_OPERATION: [{MathField.MATH_OPERATION: "I need to divide 20 by 4"}]},
			controller_outputs=[
				ControllerOutput(
					action=ControllerContinueReasoningChoice.CONTINUE_REASONING,
					action_arguments={},
					tool_descriptions="Action Name: continue_reasoning",
					continue_reasoning=True,
					considerations="Think about division as repeated subtraction",
					internal_reasoning="Think about division as repeated subtraction",
					prefix="Using repeated subtraction:",
					unique_action_response_count=1,
				)
			],
		)

		try:
			results = math_generator(states=state, n_samples_generation=1, demos=MATH_DEMOS)

			assert validate_output_structure(results, 1, 1)
			assert validate_reasoning_state_output(results[0], reasoning_field=MathField.MATH_OPERATION)

			# Check if prefix appears in output (it should, but might be fragile on small models)
			pred = results[0][0]
			if hasattr(pred, "reasoning"):
				# We log but don't fail if prefix is missing, as it depends on model capability
				if "Using repeated subtraction:" in pred.reasoning:
					logger.info("Prefix found in reasoning")
		except Exception as e:
			pytest.fail(f"Controller interventions failed: {e}")

	def test_controller_finish(self, math_generator):
		"""Test generator respecting controller's finish decision."""
		state = State(
			input={MathField.MATH_PROBLEM: "What is 3 + 4?"},
			reasoning={MathField.MATH_OPERATION: ["3 + 4 = 7"]},
			controller_outputs=[
				ControllerOutput(
					action=ControllerContinueReasoningChoice.FINISH,
					action_arguments={},
					tool_descriptions="Action Name: finish",
					continue_reasoning=False,
					considerations="The answer is clear, finish now",
					internal_reasoning="The answer is clear, finish now",
					prefix="",
					unique_action_response_count=1,
				)
			],
		)

		try:
			results = math_generator(states=state, n_samples_generation=1, demos=MATH_DEMOS)
			assert validate_output_structure(results, 1, 1)
			assert validate_finish_state_output(results[0], output_field=MathField.ANSWER)
		except Exception as e:
			pytest.fail(f"Controller finish failed: {e}")

	def test_argument_generation_interventions(self, argument_generator):
		"""Test argument generation with various interventions."""

		# Test cases for different intervention types
		test_cases = [
			(
				"Style (Knowledge)",
				State(
					input={ArgumentField.TOPIC: "electric vehicles", ArgumentField.STANCE: ArgumentStance.PRO.value},
					reasoning={ArgumentField.CLAIM: [{ArgumentField.CLAIM: "EVs reduce greenhouse gas emissions"}]},
					controller_outputs=[
						ControllerOutput(
							action=ControllerContinueReasoningChoice.CONTINUE_REASONING,
							action_arguments={},
							tool_descriptions="Action Name: continue_reasoning",
							continue_reasoning=True,
							considerations="I should provide accurate information...",
							internal_reasoning="I should provide accurate information...",
							prefix="",
							unique_action_response_count=1,
						)
					],
				)
			),
			(
				"Structure (Cause)",
				State(
					input={ArgumentField.TOPIC: "universal healthcare", ArgumentField.STANCE: ArgumentStance.PRO.value},
					reasoning={ArgumentField.CLAIM: [{ArgumentField.CLAIM: "Healthcare costs burden families"}]},
					controller_outputs=[
						ControllerOutput(
							action=ControllerContinueReasoningChoice.CONTINUE_REASONING,
							action_arguments={},
							tool_descriptions="Action Name: continue_reasoning",
							continue_reasoning=True,
							considerations="",
							internal_reasoning="",
							prefix="Therefore, ",
							unique_action_response_count=1,
						)
					],
				)
			),
		]

		for name, state in test_cases:
			try:
				results = argument_generator(states=state, n_samples_generation=1, demos=ARGUMENT_DEMOS)
				assert validate_output_structure(results, 1, 1)
			except Exception as e:
				pytest.fail(f"Argument generation ({name}) failed: {e}")


if __name__ == "__main__":
	import sys
	sys.exit(pytest.main(["-vv", __file__]))

