"""
Tests for the TreeOfThoughtEvaluator module.

Expected usage:
```bash
pytest predict/test_evaluator.py -vv
```
"""

# Standard library imports
import logging
import os
from typing import Annotated, NamedTuple
from unittest.mock import Mock

# Third-party imports
import annotated_types
import dspy
import numpy as np
import pytest
import torch

# Local imports
from constants import OpenSourceModel, Verbosity
from lm.generative_local_lm import GenerativeLocalVLLM
from predict.demos.evaluator_demos import ORM_DEMOS, PRM_DEMOS
from predict.evaluator import (
	EvaluationType,
	TreeOfThoughtEvaluator,
)
from predict.local_predict import LocalPredict
from signatures import (
	ArgumentEvaluatorMultiDimensional,
	ArgumentStance,
	GenerateArgumentWithReasoning,
	InputField,
	OutputField,
	ReasoningField,
	ReasoningSignature,
	SolveMathProblemWithReasoning,
)
from signatures.example_signatures import (
	ArgumentField,
	MathField,
	QuestionField,
)
from signatures.field_constants import (
	DEFAULT_REASONING_FIELD_NAME,
	EvaluationMetric,
)
from tree import EvaluationResult, JudgeEvaluation, State
from tree.tree_constants import NodeField, ReasoningState, ToTField

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


# Test data structures
class EvaluatorTestCase:
	"""Test case for evaluator functionality."""

	def __init__(
		self,
		test_id: str,
		states: list[State],
		evaluation_type: str,
		expected_evaluation_count: int,
		description: str,
	):
		self.test_id = test_id
		self.states = states
		self.evaluation_type = evaluation_type
		self.expected_evaluation_count = expected_evaluation_count
		self.description = description


@pytest.fixture
def mock_prm_predict():
	"""Mock LocalPredict for PRM (Process Reward Model) to avoid actual LLM calls."""
	mock = Mock(spec=LocalPredict)

	# Mock __call__ method to return fake PRM predictions with soundness + promise
	def mock_call(config=None, demos=None, **kwargs):  # noqa: ARG001
		batch_size = len(list(kwargs.values())[0]) if kwargs else 1
		predictions = []

		for i in range(batch_size):
			mock_prediction = Mock()
			# PRM uses soundness and promise scores
			mock_prediction.completions = {
				EvaluationMetric.SOUNDNESS: 3.0 + (i * 0.1),  # Vary scores slightly
				EvaluationMetric.PROMISE: 2.0 + (i * 0.1),  # Vary scores slightly
				ReasoningState.FEEDBACK: f"Test evaluation reasoning for item {i}",
			}
			predictions.append(mock_prediction)

		return predictions

	mock.__call__ = mock_call
	mock.side_effect = mock_call  # In case it's called directly
	mock.return_value = mock
	return mock


@pytest.fixture
def mock_orm_predict():
	"""Mock LocalPredict for ORM (Outcome Reward Model) to avoid actual LLM calls."""
	mock = Mock(spec=LocalPredict)

	# Mock __call__ method to return fake ORM predictions with single quality score
	def mock_call(config=None, demos=None, **kwargs):  # noqa: ARG001
		batch_size = len(list(kwargs.values())[0]) if kwargs else 1
		predictions = []

		for i in range(batch_size):
			mock_prediction = Mock()
			# ORM uses single quality score
			mock_prediction.completions = {
				NodeField.QUALITY: 4.0 + (i * 0.1),  # Vary scores slightly
				ReasoningState.FEEDBACK: f"Test evaluation reasoning for item {i}",
			}
			predictions.append(mock_prediction)

		return predictions

	mock.__call__ = mock_call
	mock.side_effect = mock_call  # In case it's called directly
	mock.return_value = mock
	return mock


@pytest.fixture
def mock_local_predict(mock_prm_predict):
	"""Backwards compatibility: default to PRM mock."""
	return mock_prm_predict


@pytest.fixture
def simple_reasoning_signature():
	"""Create a simple reasoning signature for testing."""

	class TestReasoningSignature(ReasoningSignature):
		question: str = InputField(desc="The question to answer")
		reasoning_step: str = ReasoningField(desc="A reasoning step toward the answer")
		answer: str = OutputField(desc="The final answer")

	return TestReasoningSignature


@pytest.fixture
def sample_states():
	"""Create sample states for testing."""
	states = []

	# State with first reasoning step
	state1 = State(
		input={"question": "What is the capital of France?"},
		reasoning={
			QuestionField.REASONING_STEP: [
				"What is France, I don't remember.",
				"Let me review my geography knowledge.",
			]
		},
		output={},
	)
	states.append(state1)

	# State with multiple reasoning steps
	state2 = State(
		input={"question": "What is the capital of France?"},
		reasoning={
			QuestionField.REASONING_STEP: [
				"I need to recall my knowledge about European capitals.",
				"France is a country in Western Europe.",
			]
		},
		output={},
	)
	states.append(state2)

	# State with final output
	state3 = State(
		input={"question": "What is the capital of France?"},
		reasoning={
				QuestionField.REASONING_STEP: [
				"The capital of France is Paris.",
			]
		},
		output={"answer": "Paris"},
	)
	states.append(state3)

	return states


@pytest.fixture
def evaluator_with_mocked_predictors(
	simple_reasoning_signature, mock_prm_predict, mock_orm_predict
):
	"""Create evaluator with mocked LocalPredict instances."""
	evaluator = TreeOfThoughtEvaluator(
		generator_signature=simple_reasoning_signature, verbosity=Verbosity.INFO
	)

	# Replace the LocalPredict instances with appropriate mocks
	evaluator.process_evaluator = mock_prm_predict  # PRM uses soundness + promise
	evaluator.outcome_evaluator = mock_orm_predict  # ORM uses quality

	return evaluator


class TestTreeOfThoughtEvaluator:
	"""Test cases for TreeOfThoughtEvaluator class."""

	def test_initialization(self, simple_reasoning_signature):
		"""Test that evaluator initializes correctly with required parameters."""
		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature
		)

		assert evaluator.reasoning_field_name == DEFAULT_REASONING_FIELD_NAME
		assert isinstance(evaluator.process_evaluator, LocalPredict)
		assert isinstance(evaluator.outcome_evaluator, LocalPredict)

	def test_signature_creation_methods(self, simple_reasoning_signature):
		"""Test that PRM and ORM evaluation signatures are created correctly."""
		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature
		)

		# Test PRM (process) signature - should have soundness + promise
		process_sig = evaluator._create_process_evaluator_signature()
		assert ToTField.REASONING_STEPS in process_sig.input_fields
		assert EvaluationMetric.SOUNDNESS in process_sig.output_fields
		assert EvaluationMetric.PROMISE in process_sig.output_fields
		assert ReasoningState.FEEDBACK in process_sig.output_fields

		# Test ORM (outcome) signature - should have single quality score
		outcome_sig = evaluator._create_outcome_evaluator_signature()
		assert "answer" in outcome_sig.input_fields  # Generator output field
		assert NodeField.QUALITY in outcome_sig.output_fields
		assert ReasoningState.FEEDBACK in outcome_sig.output_fields
		# ORM should NOT have soundness/promise
		assert EvaluationMetric.SOUNDNESS not in outcome_sig.output_fields
		assert EvaluationMetric.PROMISE not in outcome_sig.output_fields

	@pytest.mark.parametrize(
		"test_case",
		[
			EvaluatorTestCase(
				test_id="process_single_state",
				states=[
					State(
						input={"question": "Test question"},
						reasoning={"reasoning_step": ["First reasoning step"]},
						output={},
					)
				],
				evaluation_type=EvaluationType.PROCESS,
				expected_evaluation_count=1,
				description="Evaluate reasoning step quality for single state",
			),
			EvaluatorTestCase(
				test_id="process_multiple_states",
				states=[
					State(
						input={"question": "Q1"},
						reasoning={"reasoning_step": ["Step 1", "Step 2"]},
						output={},
					),
					State(
						input={"question": "Q2"},
						reasoning={"reasoning_step": ["Step A", "Step B", "Step C"]},
						output={},
					),
				],
				evaluation_type=EvaluationType.PROCESS,
				expected_evaluation_count=2,
				description="Evaluate reasoning step quality for multiple states",
			),
			EvaluatorTestCase(
				test_id="outcome_with_outputs",
				states=[
					State(
						input={"question": "Final test"},
						reasoning={"reasoning_step": ["Reasoning 1", "Reasoning 2"]},
						output={"answer": "Final answer"},
					)
				],
				evaluation_type=EvaluationType.OUTCOME,
				expected_evaluation_count=1,
				description="Evaluate final solution quality with reasoning chain",
			),
		],
	)
	def test_forward_evaluation_types(
		self, test_case, evaluator_with_mocked_predictors
	):
		"""Test forward method for different evaluation types and state configurations."""
		evaluator = evaluator_with_mocked_predictors

		# Call evaluator directly (evaluation type is auto-detected)
		evaluation_results = evaluator(
			states=test_case.states, n_samples_evaluator=3, evaluator_temperature=0.0
		)
		# Validate evaluation results structure
		assert len(evaluation_results) == test_case.expected_evaluation_count
		for result_list in evaluation_results:
			assert isinstance(result_list, list)
			assert len(result_list) == 1
			assert isinstance(result_list[0], EvaluationResult)
			assert isinstance(result_list[0].score, int | float)
			assert isinstance(result_list[0].judge_evaluations, list)

	def test_state_to_evaluator_input_conversion(
		self, evaluator_with_mocked_predictors, sample_states
	):
		"""Test state to evaluator input conversion for different evaluation types."""
		evaluator = evaluator_with_mocked_predictors

		# Test process evaluation conversion - evaluates all reasoning steps at once
		process_input = evaluator._state_to_evaluator_input(
			sample_states[0], EvaluationType.PROCESS
		)
		assert "question" in process_input
		assert ToTField.REASONING_STEPS in process_input
		assert process_input[ToTField.REASONING_STEPS] == [
			"What is France, I don't remember.",
			"Let me review my geography knowledge.",
		]

		# Test process evaluation with multiple steps
		process_input_multi = evaluator._state_to_evaluator_input(
			sample_states[1], EvaluationType.PROCESS
		)
		assert QuestionField.QUESTION in process_input_multi
		assert ToTField.REASONING_STEPS in process_input_multi
		assert process_input_multi[ToTField.REASONING_STEPS] == [
			"I need to recall my knowledge about European capitals.",
			"France is a country in Western Europe.",
		]

		# Test outcome evaluation conversion
		outcome_input = evaluator._state_to_evaluator_input(
			sample_states[2], EvaluationType.OUTCOME
		)
		assert QuestionField.QUESTION in outcome_input
		assert MathField.ANSWER in outcome_input
		assert ToTField.REASONING_STEPS in outcome_input  # includes reasoning for context
		assert len(outcome_input[ToTField.REASONING_STEPS]) == 1
		assert outcome_input[MathField.ANSWER] == "Paris"

	def test_consolidate_scores_conversion(
		self, simple_reasoning_signature
	):
		"""Test conversion of predictions to EvaluationResult with multi-dimension scores."""
		# Create custom evaluator with quality and clarity dimensions
		signature_fields = {
			"quality": (
				Annotated[float, annotated_types.Ge(1.0), annotated_types.Le(7.0)],
				dspy.OutputField(desc="Quality 1-7"),
			),
			EvaluationMetric.CLARITY: (
				Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(7.0)],
				dspy.OutputField(desc="Clarity 1-7"),
			),
			"feedback": (str, dspy.OutputField(desc="Feedback")),
		}
		custom_sig = dspy.Signature(signature_fields, "Multi-dimension evaluation")

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature=custom_sig,
		)

		# Create mock predictions with multiple judges and dimensions
		# Using scores within valid 1-7 range
		mock_predictions = []
		test_scores = [
			# (quality_j1, quality_j2, clarity_j1, clarity_j2)
			(3.0, 5.0, 4.0, 6.0),
			(4.0, 6.0, 2.0, 4.0),
			(2.0, 7.0, 5.0, 3.0),
		]


		for i, (quality1, quality2, clarity1, clarity2) in enumerate(test_scores):
			mock_pred = Mock()
			# Simulate 2 judges per prediction with 2 dimensions each
			mock_pred.completions = {
				NodeField.QUALITY: [quality1, quality2],  # Two judges
				EvaluationMetric.CLARITY: [clarity1, clarity2],  # Two judges
				ReasoningState.FEEDBACK: [f"Test reasoning {i} judge 1", f"Test reasoning {i} judge 2"],
			}
			mock_predictions.append(mock_pred)

		# Test conversion (using PRM rubric for this test)
		output_dicts = evaluator._consolidate_scores(mock_predictions, evaluator.prm_rubric)

		# Should return one EvaluationResult per prediction
		assert len(output_dicts) == 3
		for i, eval_result in enumerate(output_dicts):
			quality1, quality2, clarity1, clarity2 = test_scores[i]

			# Check structure
			assert isinstance(eval_result, EvaluationResult)
			assert hasattr(eval_result, 'score')
			assert hasattr(eval_result, 'judge_evaluations')

			# Verify total score is weighted average across dimensions, then averaged across judges
			# With equal weights (0.5 for quality, 0.5 for clarity):
			# For each judge: weighted_score = 0.5 * quality_norm + 0.5 * clarity_norm
			# Total score = mean(weighted_score_j1, weighted_score_j2)
			expected_quality_j1_norm = (quality1 - 1.0) / (7.0 - 1.0)
			expected_quality_j2_norm = (quality2 - 1.0) / (7.0 - 1.0)
			expected_clarity_j1_norm = (clarity1 - 0.0) / (7.0 - 0.0)
			expected_clarity_j2_norm = (clarity2 - 0.0) / (7.0 - 0.0)
			# Each dimension gets equal weight (0.5)
			weighted_score_j1 = 0.5 * expected_quality_j1_norm + 0.5 * expected_clarity_j1_norm
			weighted_score_j2 = 0.5 * expected_quality_j2_norm + 0.5 * expected_clarity_j2_norm
			expected_total = np.mean([weighted_score_j1, weighted_score_j2])
			assert eval_result.score == expected_total

			# Check judge_evaluations structure
			assert len(eval_result.judge_evaluations) == 2
			for judge_idx, judge_eval in enumerate(eval_result.judge_evaluations):
				assert isinstance(judge_eval, JudgeEvaluation)
				assert "quality" in judge_eval.normalized_scores
				assert "clarity" in judge_eval.normalized_scores
				assert "quality" in judge_eval.raw_scores
				assert "clarity" in judge_eval.raw_scores

				# Verify individual judge raw scores
				judge_quality = quality1 if judge_idx == 0 else quality2
				judge_clarity = clarity1 if judge_idx == 0 else clarity2
				assert judge_eval.raw_scores["quality"] == judge_quality
				assert judge_eval.raw_scores["clarity"] == judge_clarity

				# Verify individual judge normalized scores
				expected_judge_quality = (judge_quality - 1.0) / (7.0 - 1.0)
				expected_judge_clarity = (judge_clarity - 0.0) / (7.0 - 0.0)
				assert judge_eval.normalized_scores["quality"] == expected_judge_quality
				assert judge_eval.normalized_scores["clarity"] == expected_judge_clarity
				assert judge_eval.feedback == f"Test reasoning {i} judge {judge_idx + 1}"

	def test_evaluator_selection_by_type(self, evaluator_with_mocked_predictors):
		"""Test that correct evaluator is selected for each evaluation type."""
		evaluator = evaluator_with_mocked_predictors

		# Mock different evaluators for identification
		process_evaluator = Mock()
		outcome_evaluator = Mock()

		evaluator.process_evaluator = process_evaluator
		evaluator.outcome_evaluator = outcome_evaluator

		# Test evaluator selection
		assert (
			evaluator._get_evaluator_for_type(EvaluationType.PROCESS)
			== process_evaluator
		)
		assert (
			evaluator._get_evaluator_for_type(EvaluationType.OUTCOME)
			== outcome_evaluator
		)

	def test_batch_processing_efficiency(
		self, evaluator_with_mocked_predictors, sample_states
	):
		"""Test that batch processing works correctly with multiple states."""
		evaluator = evaluator_with_mocked_predictors

		# Process multiple states at once (evaluation type is auto-detected)
		evaluation_results = evaluator(
			states=sample_states[:2]  # Use first 2 states without outputs
		)

		# Verify batch processing occurred
		assert len(evaluation_results) == 2

		# Verify each result contains proper evaluation data
		for result_list in evaluation_results:
			assert isinstance(result_list, list)
			assert len(result_list) > 0
			assert isinstance(result_list[0], EvaluationResult)
			assert hasattr(result_list[0], 'score')
			assert hasattr(result_list[0], 'judge_evaluations')

	def test_fixed_likert_scale(self, simple_reasoning_signature):
		"""Test that evaluator uses fixed 1-7 Likert scale for both PRM and ORM."""
		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature
		)

		# Check PRM signature has soundness and promise fields with proper bounds (1-7 Likert scale)
		process_sig = evaluator._create_process_evaluator_signature()
		# Verify fields exist with 1-7 bounds
		assert EvaluationMetric.SOUNDNESS in process_sig.output_fields
		assert EvaluationMetric.PROMISE in process_sig.output_fields
		# Check that the evaluator has the correct bounds for PRM
		assert evaluator.prm_dimension_bounds[EvaluationMetric.SOUNDNESS] == (1, 7)
		assert evaluator.prm_dimension_bounds[EvaluationMetric.PROMISE] == (1, 7)

		# Check ORM signature has quality field with 1-7 bounds
		outcome_sig = evaluator._create_outcome_evaluator_signature()
		assert NodeField.QUALITY in outcome_sig.output_fields
		assert EvaluationMetric.SOUNDNESS not in outcome_sig.output_fields
		assert EvaluationMetric.PROMISE not in outcome_sig.output_fields
		# Check ORM bounds
		assert evaluator.orm_dimension_bounds[NodeField.QUALITY] == (1, 7)

	def test_empty_reasoning_assertion_for_prm(self, evaluator_with_mocked_predictors):
		"""Test that PRM evaluation raises ValueError for empty reasoning steps."""
		evaluator = evaluator_with_mocked_predictors

		# Create a PRM state (no output) with empty reasoning steps
		prm_state_empty_reasoning = State(
			input={"question": "What is the capital of France?"},
			reasoning={"reasoning_step": []},  # Empty reasoning
			output={},  # No output -> triggers PRM evaluation
		)

		# Should raise ValueError when trying to evaluate empty reasoning with PRM
		with pytest.raises(ValueError) as exc_info:
			evaluator(states=prm_state_empty_reasoning)

		# Verify the error message mentions PRM requirement
		assert "PRM evaluation requires at least one reasoning step" in str(
			exc_info.value
		)

	def test_empty_reasoning_allowed_for_orm(self, evaluator_with_mocked_predictors):
		"""Test that ORM evaluation allows empty reasoning steps."""
		evaluator = evaluator_with_mocked_predictors

		# Create an ORM state (with output) with empty reasoning steps
		orm_state_empty_reasoning = State(
			input={"question": "What is the capital of France?"},
			reasoning={"reasoning_step": []},  # Empty reasoning
			output={"answer": "Paris"},  # Has output -> triggers ORM evaluation
		)

		# Should NOT raise an error - ORM allows empty reasoning
		evaluation_results = evaluator(states=orm_state_empty_reasoning)

		# Verify evaluation succeeded
		assert len(evaluation_results) == 1
		assert isinstance(evaluation_results[0], list)
		assert len(evaluation_results[0]) > 0
		assert isinstance(evaluation_results[0][0], EvaluationResult)
		assert hasattr(evaluation_results[0][0], 'score')
		assert hasattr(evaluation_results[0][0], 'judge_evaluations')


class TestCustomEvaluatorSignatures:
	"""Test cases for custom evaluator signature functionality."""

	def test_custom_single_dimension_signature(self, simple_reasoning_signature):
		"""Test evaluator with custom single-dimension signature."""
		# Create custom evaluator signature with different bounds
		signature_fields = {
			"quality": (
				Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(10.0)],
				dspy.OutputField(desc="Quality score 0-10"),
			),
			"feedback": (str, dspy.OutputField(desc="Detailed feedback")),
		}
		custom_sig = dspy.Signature(signature_fields, "Evaluate quality on 0-10 scale")

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature=custom_sig,
		)

		# Verify field extraction (since custom_sig is used for both PRM and ORM)
		assert evaluator.prm_numeric_score_fields == ["quality"]
		assert evaluator.prm_feedback_field == "feedback"
		assert evaluator.prm_dimension_bounds == {"quality": (0.0, 10.0)}
		# ORM uses same signature
		assert evaluator.orm_numeric_score_fields == ["quality"]
		assert evaluator.orm_feedback_field == "feedback"
		assert evaluator.orm_dimension_bounds == {"quality": (0.0, 10.0)}

	def test_custom_multi_dimension_signature(self, simple_reasoning_signature):
		"""Test evaluator with custom multi-dimension signature."""
		# Create multi-dimension evaluator signature
		signature_fields = {
			EvaluationMetric.CORRECTNESS: (
				Annotated[float, annotated_types.Ge(1.0), annotated_types.Le(7.0)],
				dspy.OutputField(desc="Correctness 1-7"),
			),
			EvaluationMetric.CLARITY: (
				Annotated[float, annotated_types.Ge(1.0), annotated_types.Le(5.0)],
				dspy.OutputField(desc="Clarity 1-5"),
			),
			"completeness": (
				Annotated[float, annotated_types.Ge(1.0), annotated_types.Le(3.0)],
				dspy.OutputField(desc="Completeness 1-3"),
			),
			"feedback": (str, dspy.OutputField(desc="Detailed feedback")),
		}
		custom_sig = dspy.Signature(signature_fields, "Evaluate on multiple dimensions")

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature=custom_sig,
		)

		# Verify fields for both PRM and ORM (same custom signature used for both)
		assert evaluator.prm_numeric_score_fields == [
			EvaluationMetric.CORRECTNESS,
			EvaluationMetric.CLARITY,
			EvaluationMetric.COMPLETENESS,
		]
		assert evaluator.prm_feedback_field == ReasoningState.FEEDBACK
		assert evaluator.prm_dimension_bounds == {
			EvaluationMetric.CORRECTNESS: (1.0, 7.0),
			EvaluationMetric.CLARITY: (1.0, 5.0),
			EvaluationMetric.COMPLETENESS: (1.0, 3.0),
		}
		assert evaluator.orm_numeric_score_fields == evaluator.prm_numeric_score_fields
		assert evaluator.orm_feedback_field == evaluator.prm_feedback_field
		assert evaluator.orm_dimension_bounds == evaluator.prm_dimension_bounds

	def test_custom_signature_without_bounds(self, simple_reasoning_signature):
		"""Test that default bounds are added when not specified."""
		# Create signature without bounds metadata
		signature_fields = {
			"score": (float, dspy.OutputField(desc="Quality score")),
			"feedback": (str, dspy.OutputField(desc="Feedback")),
		}
		custom_sig = dspy.Signature(signature_fields, "Evaluate quality")

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature=custom_sig,
		)

		# Verify default bounds were added (1, 7) - default Likert scale
		assert evaluator.prm_dimension_bounds == {"score": (1, 7)}
		assert evaluator.orm_dimension_bounds == {"score": (1, 7)}

	def test_process_evaluator_output_fields_method(self, simple_reasoning_signature):
		"""Test _process_evaluator_output_fields method directly."""
		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature
		)

		# Create test signature
		signature_fields = {
			"dim1": (
				Annotated[int, annotated_types.Ge(0), annotated_types.Le(5)],
				dspy.OutputField(desc="Dimension 1"),
			),
			"dim2": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(100)],
				dspy.OutputField(desc="Dimension 2"),
			),
			"notes": (str, dspy.OutputField(desc="Notes")),
		}
		test_sig = dspy.Signature(signature_fields, "Test signature")

		# Process output fields
		numeric_fields, feedback_field, bounds, weights = (
			evaluator._process_evaluator_output_fields(test_sig)
		)

		assert numeric_fields == ["dim1", "dim2"]
		assert feedback_field == "notes"
		assert bounds == {"dim1": (0.0, 5.0), "dim2": (1.0, 100.0)}
		# No rubric_weight specified, so equal weights
		assert weights == {"dim1": 0.5, "dim2": 0.5}

	def test_custom_signature_invalid_no_numeric_field(
		self, simple_reasoning_signature
	):
		"""Test that signature without numeric field raises error."""
		# Create signature with only string fields
		signature_fields = {
			"feedback": (str, dspy.OutputField(desc="Feedback")),
			"notes": (str, dspy.OutputField(desc="Notes")),
		}
		custom_sig = dspy.Signature(signature_fields, "Invalid signature")

		# Should raise assertion error
		with pytest.raises(AssertionError) as exc_info:
			TreeOfThoughtEvaluator(
				generator_signature=simple_reasoning_signature,
				evaluator_signature=custom_sig,
			)

		assert "at least one numeric output field" in str(exc_info.value)

	def test_custom_signature_invalid_no_string_field(self, simple_reasoning_signature):
		"""Test that signature without string field raises error."""
		# Create signature with only numeric fields
		signature_fields = {
			"score1": (
				Annotated[float, annotated_types.Ge(1.0), annotated_types.Le(7.0)],
				dspy.OutputField(desc="Score 1"),
			),
			"score2": (
				Annotated[float, annotated_types.Ge(1.0), annotated_types.Le(7.0)],
				dspy.OutputField(desc="Score 2"),
			),
		}
		custom_sig = dspy.Signature(signature_fields, "Invalid signature")

		# Should raise assertion error
		with pytest.raises(AssertionError) as exc_info:
			TreeOfThoughtEvaluator(
				generator_signature=simple_reasoning_signature,
				evaluator_signature=custom_sig,
			)

		assert "one string output field" in str(exc_info.value)

	def test_custom_signature_multiple_feedback_fields_error(
		self, simple_reasoning_signature
	):
		"""Test that signature with multiple string fields raises error."""
		# Create signature with multiple string fields
		signature_fields = {
			"score": (
				Annotated[float, annotated_types.Ge(1.0), annotated_types.Le(7.0)],
				dspy.OutputField(desc="Score"),
			),
			"feedback": (str, dspy.OutputField(desc="Feedback")),
			"notes": (str, dspy.OutputField(desc="Additional notes")),
		}
		custom_sig = dspy.Signature(signature_fields, "Invalid signature")

		# Should raise assertion error
		with pytest.raises(AssertionError) as exc_info:
			TreeOfThoughtEvaluator(
				generator_signature=simple_reasoning_signature,
				evaluator_signature=custom_sig,
			)

		assert "one string output field" in str(exc_info.value)

	def test_custom_signature_invalid_same_bounds_error(
		self, simple_reasoning_signature
	):
		"""Test that signature with same lower and upper bounds raises error during init."""
		# Create signature with same lower and upper bounds
		signature_fields = {
			"score": (
				Annotated[float, annotated_types.Ge(5.0), annotated_types.Le(5.0)],
				dspy.OutputField(desc="Score with same bounds"),
			),
			"feedback": (str, dspy.OutputField(desc="Feedback")),
		}
		custom_sig = dspy.Signature(signature_fields, "Invalid signature")

		# Should raise assertion error during initialization
		with pytest.raises(AssertionError) as exc_info:
			TreeOfThoughtEvaluator(
				generator_signature=simple_reasoning_signature,
				evaluator_signature=custom_sig,
			)

		assert "Lower bound must be less than upper bound" in str(exc_info.value)

	def test_consolidate_scores_multi_dimension(
		self,
		simple_reasoning_signature,
	):
		"""Test score consolidation with multi-dimension custom signature."""
		# Create multi-dimension signature
		signature_fields = {
			EvaluationMetric.CORRECTNESS: (
				Annotated[float, annotated_types.Ge(1.0), annotated_types.Le(7.0)],
				dspy.OutputField(desc="Correctness 1-7"),
			),
			EvaluationMetric.CLARITY: (
				Annotated[float, annotated_types.Ge(1.0), annotated_types.Le(7.0)],
				dspy.OutputField(desc="Clarity 1-7"),
			),
			"feedback": (str, dspy.OutputField(desc="Feedback")),
		}
		custom_sig = dspy.Signature(signature_fields, "Multi-dimension evaluation")

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature=custom_sig,
		)

		# Create mock predictions with multi-dimension scores
		mock_predictions = []
		mock_pred = Mock()
		mock_pred.completions = {
			EvaluationMetric.CORRECTNESS: [2.0, 2.0, 3.0],  # 3 judges
			EvaluationMetric.CLARITY: [3.0, 6.0, 1.0],  # 3 judges
			ReasoningState.FEEDBACK: ["Good work", "Great job", "Excellent"],
		}
		mock_predictions.append(mock_pred)

		# Test consolidation (custom signature is used for both PRM and ORM)
		output_dicts = evaluator._consolidate_scores(mock_predictions, evaluator.prm_rubric)

		# Verify structure
		assert len(output_dicts) == 1
		eval_result = output_dicts[0]
		assert isinstance(eval_result, EvaluationResult)

		# Verify score is normalized (0-1 range)
		assert 0.0 <= eval_result.score <= 1.0

		# Verify total score is weighted average across dimensions, then averaged across judges
		# With equal weights (0.5 for correctness, 0.5 for clarity):
		# correctness: [2.0, 2.0, 3.0] -> normalized: [(2-1)/6, (2-1)/6, (3-1)/6]
		# clarity: [3.0, 6.0, 1.0] -> normalized: [(3-1)/6, (6-1)/6, (1-1)/6]
		correctness_normalized = [(2.0 - 1.0) / 6.0, (2.0 - 1.0) / 6.0, (3.0 - 1.0) / 6.0]
		clarity_normalized = [(3.0 - 1.0) / 6.0, (6.0 - 1.0) / 6.0, (1.0 - 1.0) / 6.0]
		# For each judge: weighted_score = 0.5 * correctness_norm + 0.5 * clarity_norm
		weighted_scores = [
			0.5 * correctness_normalized[i] + 0.5 * clarity_normalized[i]
			for i in range(3)
		]
		expected_total = np.mean(weighted_scores)
		assert eval_result.score == expected_total

		# Verify judge_evaluations structure
		assert len(eval_result.judge_evaluations) == 3
		for judge_idx, judge_eval in enumerate(eval_result.judge_evaluations):
			assert isinstance(judge_eval, JudgeEvaluation)
			assert "correctness" in judge_eval.normalized_scores
			assert "clarity" in judge_eval.normalized_scores
			assert "correctness" in judge_eval.raw_scores
			assert "clarity" in judge_eval.raw_scores

			# Check individual judge scores are normalized correctly
			if judge_idx == 0:
				assert judge_eval.raw_scores["correctness"] == 2.0
				assert judge_eval.raw_scores["clarity"] == 3.0
				assert judge_eval.normalized_scores["correctness"] == (2.0 - 1.0) / 6.0
				assert judge_eval.normalized_scores["clarity"] == (3.0 - 1.0) / 6.0
				assert judge_eval.feedback == "Good work"
			elif judge_idx == 1:
				assert judge_eval.raw_scores["correctness"] == 2.0
				assert judge_eval.raw_scores["clarity"] == 6.0
				assert judge_eval.normalized_scores["correctness"] == (2.0 - 1.0) / 6.0
				assert judge_eval.normalized_scores["clarity"] == (6.0 - 1.0) / 6.0
				assert judge_eval.feedback == "Great job"
			elif judge_idx == 2:
				assert judge_eval.raw_scores["correctness"] == 3.0
				assert judge_eval.raw_scores["clarity"] == 1.0
				assert judge_eval.normalized_scores["correctness"] == (3.0 - 1.0) / 6.0
				assert judge_eval.normalized_scores["clarity"] == (1.0 - 1.0) / 6.0
				assert judge_eval.feedback == "Excellent"

	def test_normalize_score_method(self, simple_reasoning_signature):
		"""Test _normalize_score method with different bounds."""
		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature
		)

		# Test normalization with 1-7 scale
		assert evaluator._normalize_score(1.0, 1.0, 7.0) == 0.0
		assert evaluator._normalize_score(4.0, 1.0, 7.0) == 0.5
		assert evaluator._normalize_score(7.0, 1.0, 7.0) == 1.0

		# Test normalization with 0-10 scale
		assert evaluator._normalize_score(0.0, 0.0, 10.0) == 0.0
		assert evaluator._normalize_score(5.0, 0.0, 10.0) == 0.5
		assert evaluator._normalize_score(10.0, 0.0, 10.0) == 1.0

	def test_bounds_validation_single_dimension(self, simple_reasoning_signature):
		"""Test that bounds validation raises error for out-of-bounds scores."""
		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature
		)

		# Create mock prediction with out-of-bounds score (default bounds are 1-7)
		mock_predictions = []
		mock_pred = Mock()
		mock_pred.completions = {
			EvaluationMetric.SOUNDNESS: 10.0,  # Out of bounds (> 7.0)
			EvaluationMetric.PROMISE: 5.0,  # Within bounds
			ReasoningState.FEEDBACK: "Test feedback",
		}
		mock_predictions.append(mock_pred)

		# Should raise assertion error about bounds
		with pytest.raises(AssertionError) as exc_info:
			evaluator._consolidate_scores(mock_predictions, evaluator.prm_rubric)

		assert "out of bounds" in str(exc_info.value).lower()

	def test_bounds_validation_multi_dimension(self, simple_reasoning_signature):
		"""Test bounds validation for multi-dimension scores."""
		# Create multi-dimension signature with different bounds
		signature_fields = {
			EvaluationMetric.CORRECTNESS: (
				Annotated[float, annotated_types.Ge(1.0), annotated_types.Le(7.0)],
				dspy.OutputField(desc="Correctness 1-7"),
			),
			EvaluationMetric.CLARITY: (
				Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(5.0)],
				dspy.OutputField(desc="Clarity 0-5"),
			),
			"feedback": (str, dspy.OutputField(desc="Feedback")),
		}
		custom_sig = dspy.Signature(signature_fields, "Multi-dimension evaluation")

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature=custom_sig,
		)

		# Test with out-of-bounds correctness score
		mock_predictions = []
		mock_pred = Mock()
		mock_pred.completions = {
			EvaluationMetric.CORRECTNESS: [8.0],  # Out of bounds (> 7.0)
			EvaluationMetric.CLARITY: [3.0],  # Within bounds
			ReasoningState.FEEDBACK: ["Good work"],
		}
		mock_predictions.append(mock_pred)

		with pytest.raises(AssertionError) as exc_info:
			evaluator._consolidate_scores(mock_predictions, custom_sig)

		assert EvaluationMetric.CORRECTNESS in str(exc_info.value).lower()
		assert "out of bounds" in str(exc_info.value).lower()
		assert "8.0" in str(exc_info.value)

		# Test with out-of-bounds clarity score
		mock_pred2 = Mock()
		mock_pred2.completions = {
			EvaluationMetric.CORRECTNESS: [5.0],  # Within bounds
			EvaluationMetric.CLARITY: [6.0],  # Out of bounds (> 5.0)
			ReasoningState.FEEDBACK: ["Great job"],
		}
		mock_predictions2 = [mock_pred2]

		with pytest.raises(AssertionError) as exc_info:
			evaluator._consolidate_scores(mock_predictions2, custom_sig)

		assert "clarity" in str(exc_info.value).lower()
		assert "out of bounds" in str(exc_info.value).lower()
		assert "6.0" in str(exc_info.value)

	def test_bounds_validation_accepts_valid_scores(self, simple_reasoning_signature):
		"""Test that bounds validation accepts valid scores."""
		# Create custom signature with specific bounds
		signature_fields = {
			"quality": (
				Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(10.0)],
				dspy.OutputField(desc="Quality 0-10"),
			),
			"feedback": (str, dspy.OutputField(desc="Feedback")),
		}
		custom_sig = dspy.Signature(signature_fields, "Quality evaluation")

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature=custom_sig,
		)

		# Create mock predictions with valid scores
		mock_predictions = []
		for score in [0.0, 5.0, 10.0]:  # Edge cases and middle value
			mock_pred = Mock()
			mock_pred.completions = {
				"quality": [score],
				"feedback": ["Test feedback"],
			}
			mock_predictions.append(mock_pred)

		# Should not raise any errors
		output_dicts = evaluator._consolidate_scores(mock_predictions, evaluator.prm_rubric)

		# Verify all scores were processed successfully
		assert len(output_dicts) == 3
		for eval_result in output_dicts:
			assert isinstance(eval_result, EvaluationResult)
			assert hasattr(eval_result, 'score')
			assert hasattr(eval_result, 'judge_evaluations')

	def test_demo_handling_default_signature(self, simple_reasoning_signature):
		"""Test that default signature is marked as non-custom."""
		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature
		)

		# Should be marked as non-custom for both PRM and ORM
		assert evaluator.is_custom_prm_signature is False
		assert evaluator.is_custom_orm_signature is False

	def test_demo_handling_explicit_demos(self, simple_reasoning_signature):
		"""Test that custom signature is properly detected."""
		# Create custom signature
		signature_fields = {
			"quality": (
				Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(10.0)],
				dspy.OutputField(desc="Quality 0-10"),
			),
			"feedback": (str, dspy.OutputField(desc="Feedback")),
		}
		custom_sig = dspy.Signature(signature_fields, "Custom rubric")

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature=custom_sig,
		)

		# Should be marked as custom signature for both PRM and ORM
		assert evaluator.is_custom_prm_signature is True
		assert evaluator.is_custom_orm_signature is True

	def test_demo_handling_custom_signature(
		self, simple_reasoning_signature
	):
		"""Test that custom signatures use zero-shot and don't get default demos to prevent field mismatch errors."""
		# Create multi-dimension custom signature with different field names
		signature_fields = {
			EvaluationMetric.CORRECTNESS: (
				Annotated[float, annotated_types.Ge(1.0), annotated_types.Le(5.0)],
				dspy.OutputField(desc="Correctness 1-5"),
			),
			EvaluationMetric.CLARITY: (
				Annotated[float, annotated_types.Ge(1.0), annotated_types.Le(5.0)],
				dspy.OutputField(desc="Clarity 1-5"),
			),
			"feedback": (str, dspy.OutputField(desc="Feedback")),
		}
		custom_sig = dspy.Signature(signature_fields, "Multi-dimension rubric")

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature=custom_sig,
		)

		# Verify custom signature flags are set (both PRM and ORM use same custom signature)
		assert evaluator.is_custom_prm_signature is True
		assert evaluator.is_custom_orm_signature is True

		# Verify field names in custom signature don't match default demo field names
		# Since we provided evaluator_signature only, both PRM and ORM use the same custom signature
		assert EvaluationMetric.SOUNDNESS not in evaluator.prm_numeric_score_fields
		assert EvaluationMetric.PROMISE not in evaluator.prm_numeric_score_fields
		assert EvaluationMetric.CORRECTNESS in evaluator.prm_numeric_score_fields
		assert EvaluationMetric.CLARITY in evaluator.prm_numeric_score_fields
		assert EvaluationMetric.CORRECTNESS in evaluator.orm_numeric_score_fields
		assert EvaluationMetric.CLARITY in evaluator.orm_numeric_score_fields

	def test_demo_handling_default_signature_zero_shot(self, simple_reasoning_signature):
		"""Test that zero-shot is the default behavior when no demos provided."""
		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
		)

		# Should be marked as non-custom (both PRM and ORM use defaults)
		assert evaluator.is_custom_prm_signature is False
		assert evaluator.is_custom_orm_signature is False


class TestRubricWeight:
	"""Tests for field-level rubric_weight functionality."""

	def test_default_equal_weights(self, simple_reasoning_signature):
		"""Test that default evaluator uses equal weights when no rubric_weight specified."""
		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature
		)

		# PRM should have equal weights for soundness and promise (0.5 each)
		assert evaluator.prm_dimension_weights == {"soundness": 0.5, "promise": 0.5}
		# ORM should have single weight for quality (1.0)
		assert evaluator.orm_dimension_weights == {"quality": 1.0}

	def test_field_level_weights(self, simple_reasoning_signature):
		"""Test that field-level rubric_weight is properly extracted and normalized."""
		# Create custom signature with rubric_weight in fields
		signature_fields = {
			"soundness": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
				OutputField(desc="Soundness 1-7", rubric_weight=0.7),
			),
			"promise": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
				OutputField(desc="Promise 1-7", rubric_weight=0.3),
			),
			"feedback": (str, OutputField(desc="Feedback")),
		}
		weighted_sig = dspy.Signature(signature_fields, "Weighted PRM rubric")

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature_prm=weighted_sig,
		)

		# Should extract and use the field-level weights
		assert evaluator.prm_dimension_weights == {"soundness": 0.7, "promise": 0.3}

	def test_weight_normalization(self, simple_reasoning_signature):
		"""Test that rubric_weight values are automatically normalized to sum to 1.0."""
		# Create signature with unnormalized weights
		signature_fields = {
			"soundness": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
				OutputField(desc="Soundness 1-7", rubric_weight=7),  # Will normalize to 0.7
			),
			"promise": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
				OutputField(desc="Promise 1-7", rubric_weight=3),  # Will normalize to 0.3
			),
			"feedback": (str, OutputField(desc="Feedback")),
		}
		weighted_sig = dspy.Signature(signature_fields, "Unnormalized weights")

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature_prm=weighted_sig,
		)

		# Should be normalized to 0.7 and 0.3
		assert evaluator.prm_dimension_weights["soundness"] == 0.7
		assert evaluator.prm_dimension_weights["promise"] == 0.3

	def test_weighted_consolidation(self, simple_reasoning_signature):
		"""Test that weighted average is computed correctly with field-level weights."""
		# Create weighted signature
		signature_fields = {
			"soundness": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
				OutputField(desc="Soundness 1-7", rubric_weight=0.7),
			),
			"promise": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
				OutputField(desc="Promise 1-7", rubric_weight=0.3),
			),
			"feedback": (str, OutputField(desc="Feedback")),
		}
		weighted_sig = dspy.Signature(signature_fields, "Weighted PRM")

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature_prm=weighted_sig,
		)

		# Create mock predictions with known scores
		mock_predictions = []
		mock_pred = Mock()
		mock_pred.completions = {
			"soundness": 6.0,  # normalized: (6-1)/(7-1) = 0.833
			"promise": 4.0,  # normalized: (4-1)/(7-1) = 0.5
			"feedback": "Test feedback",
		}
		mock_predictions.append(mock_pred)

		# Test consolidation
		output_dicts = evaluator._consolidate_scores(mock_predictions, weighted_sig)

		# Calculate expected weighted score
		soundness_norm = (6.0 - 1.0) / (7.0 - 1.0)  # 0.833
		promise_norm = (4.0 - 1.0) / (7.0 - 1.0)  # 0.5
		expected_score = 0.7 * soundness_norm + 0.3 * promise_norm

		assert len(output_dicts) == 1
		assert output_dicts[0].score == expected_score

	def test_weighted_multi_judge(self, simple_reasoning_signature):
		"""Test weighted average with multiple judges using field-level weights."""
		# Create weighted signature
		signature_fields = {
			"soundness": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
				OutputField(desc="Soundness 1-7", rubric_weight=0.6),
			),
			"promise": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
				OutputField(desc="Promise 1-7", rubric_weight=0.4),
			),
			"feedback": (str, OutputField(desc="Feedback")),
		}
		weighted_sig = dspy.Signature(signature_fields, "Weighted PRM")

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature_prm=weighted_sig,
		)

		# Create mock predictions with 2 judges
		mock_predictions = []
		mock_pred = Mock()
		mock_pred.completions = {
			"soundness": [5.0, 7.0],  # Two judges
			"promise": [3.0, 5.0],  # Two judges
			"feedback": ["Judge 1 feedback", "Judge 2 feedback"],
		}
		mock_predictions.append(mock_pred)

		# Test consolidation
		output_dicts = evaluator._consolidate_scores(mock_predictions, weighted_sig)

		# Calculate expected weighted score for each judge, then average
		# Judge 1: soundness=5, promise=3
		s1_norm = (5.0 - 1.0) / 6.0  # 0.667
		p1_norm = (3.0 - 1.0) / 6.0  # 0.333
		weighted_j1 = 0.6 * s1_norm + 0.4 * p1_norm

		# Judge 2: soundness=7, promise=5
		s2_norm = (7.0 - 1.0) / 6.0  # 1.0
		p2_norm = (5.0 - 1.0) / 6.0  # 0.667
		weighted_j2 = 0.6 * s2_norm + 0.4 * p2_norm

		expected_score = (weighted_j1 + weighted_j2) / 2

		assert len(output_dicts) == 1
		assert output_dicts[0].score == expected_score
		assert len(output_dicts[0].judge_evaluations) == 2

	def test_weights_with_multi_dimension_signature(self, simple_reasoning_signature):
		"""Test rubric_weight works with custom multi-dimension signatures."""
		# Create custom 3-dimension signature with rubric_weight
		signature_fields = {
			EvaluationMetric.CORRECTNESS: (
				Annotated[float, annotated_types.Ge(1.0), annotated_types.Le(5.0)],
				OutputField(desc="Correctness 1-5", rubric_weight=0.5),
			),
			EvaluationMetric.CLARITY: (
				Annotated[float, annotated_types.Ge(1.0), annotated_types.Le(5.0)],
				OutputField(desc="Clarity 1-5", rubric_weight=0.3),
			),
			EvaluationMetric.EFFICIENCY: (
				Annotated[float, annotated_types.Ge(1.0), annotated_types.Le(5.0)],
				OutputField(desc="Efficiency 1-5", rubric_weight=0.2),
			),
			"feedback": (str, OutputField(desc="Feedback")),
		}
		custom_sig = dspy.Signature(signature_fields, "Custom weighted rubric")

		# Create evaluator (custom_sig used for both PRM and ORM)
		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature=custom_sig,
		)

		# Verify both PRM and ORM weights are extracted correctly
		assert evaluator.prm_dimension_weights["correctness"] == 0.5
		assert evaluator.prm_dimension_weights["clarity"] == 0.3
		assert evaluator.prm_dimension_weights["efficiency"] == 0.2
		assert evaluator.orm_dimension_weights["correctness"] == 0.5
		assert evaluator.orm_dimension_weights["clarity"] == 0.3
		assert evaluator.orm_dimension_weights["efficiency"] == 0.2

		# Test with mock predictions
		mock_predictions = []
		mock_pred = Mock()
		mock_pred.completions = {
			EvaluationMetric.CORRECTNESS: 4.0,
			EvaluationMetric.CLARITY: 3.0,
			EvaluationMetric.EFFICIENCY: 5.0,
			ReasoningState.FEEDBACK: "Test feedback",
		}
		mock_predictions.append(mock_pred)

		output_dicts = evaluator._consolidate_scores(mock_predictions, custom_sig)

		# Calculate expected weighted score
		c_norm = (4.0 - 1.0) / (5.0 - 1.0)  # 0.75
		cl_norm = (3.0 - 1.0) / (5.0 - 1.0)  # 0.5
		e_norm = (5.0 - 1.0) / (5.0 - 1.0)  # 1.0
		expected_score = 0.5 * c_norm + 0.3 * cl_norm + 0.2 * e_norm

		assert len(output_dicts) == 1
		assert abs(output_dicts[0].score - expected_score) < 0.001

	def test_partial_weights_error(self, simple_reasoning_signature):
		"""Test that specifying rubric_weight for only some fields raises an error."""
		# Create signature with rubric_weight on only one field
		signature_fields = {
			"soundness": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
				OutputField(desc="Soundness 1-7", rubric_weight=0.7),
			),
			"promise": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
				OutputField(desc="Promise 1-7"),  # Missing rubric_weight
			),
			"feedback": (str, OutputField(desc="Feedback")),
		}
		partial_weighted_sig = dspy.Signature(signature_fields, "Partial weights")

		with pytest.raises(AssertionError) as exc_info:
			TreeOfThoughtEvaluator(
				generator_signature=simple_reasoning_signature,
				evaluator_signature_prm=partial_weighted_sig,
			)
		assert "must be specified for all" in str(exc_info.value)
		assert "Missing rubric_weight for" in str(exc_info.value)

	def test_negative_weights(self, simple_reasoning_signature):
		"""Test that negative rubric_weight values raise an error."""
		# Create signature with negative weight
		signature_fields = {
			"soundness": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
				OutputField(desc="Soundness 1-7", rubric_weight=-0.5),
			),
			"promise": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
				OutputField(desc="Promise 1-7", rubric_weight=1.5),
			),
			"feedback": (str, OutputField(desc="Feedback")),
		}
		negative_weighted_sig = dspy.Signature(signature_fields, "Negative weights")

		with pytest.raises(AssertionError) as exc_info:
			TreeOfThoughtEvaluator(
				generator_signature=simple_reasoning_signature,
				evaluator_signature_prm=negative_weighted_sig,
			)
		assert "must be positive" in str(exc_info.value)


class TestSymmetricAPI:
	"""Tests for the fully symmetric API (evaluator_signature_prm/orm with rubric_weight, demos_prm/orm)."""

	def test_separate_prm_orm_signatures(self, simple_reasoning_signature):
		"""Test that providing separate PRM and ORM signatures works correctly."""
		# Create different signatures for PRM and ORM
		prm_sig_fields = {
			"soundness": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(10)],
				dspy.OutputField(desc="Soundness 1-10"),
			),
			"foresight": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(10)],
				dspy.OutputField(desc="Foresight 1-10"),
			),
			"feedback": (str, dspy.OutputField(desc="Feedback")),
		}
		prm_sig = dspy.Signature(prm_sig_fields, "Custom PRM rubric")

		orm_sig_fields = {
			"quality": (
				Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(1.0)],
				dspy.OutputField(desc="Quality 0-1"),
			),
			"feedback": (str, dspy.OutputField(desc="Feedback")),
		}
		orm_sig = dspy.Signature(orm_sig_fields, "Custom ORM rubric")

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature_prm=prm_sig,
			evaluator_signature_orm=orm_sig,
		)

		# Verify PRM uses prm_sig
		assert evaluator.prm_rubric == prm_sig
		assert evaluator.prm_numeric_score_fields == ["soundness", "foresight"]
		assert evaluator.prm_dimension_bounds == {"soundness": (1, 10), "foresight": (1, 10)}

		# Verify ORM uses orm_sig
		assert evaluator.orm_rubric == orm_sig
		assert evaluator.orm_numeric_score_fields == ["quality"]
		assert evaluator.orm_dimension_bounds == {"quality": (0.0, 1.0)}


	def test_demos_parameter_precedence(self, simple_reasoning_signature):
		"""Test that demos_prm and demos_orm take precedence over base demos parameter."""
		# Create evaluator
		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
		)

		# Create test demos
		base_demos = [
			{
				ReasoningState.INPUT: {QuestionField.QUESTION: "Base"},
				ReasoningState.OUTPUT: {
					EvaluationMetric.SOUNDNESS: 5,
					EvaluationMetric.PROMISE: 5,
					ReasoningState.FEEDBACK: "Base",
				},
			},
		]
		prm_demos = PRM_DEMOS
		orm_demos = ORM_DEMOS

		# Create test states
		prm_state = State(
			input={QuestionField.QUESTION: "Test PRM"},
			reasoning={QuestionField.REASONING_STEP: ["Step 1"]},
			output={},  # No output -> PRM evaluation
		)
		orm_state = State(
			input={QuestionField.QUESTION: "Test ORM"},
			reasoning={QuestionField.REASONING_STEP: ["Step 1"]},
			output={MathField.ANSWER: "Test answer"},  # Has output -> ORM evaluation
		)

		# Track which demos were actually used by capturing them in closures
		captured_prm_demos = []
		captured_orm_demos = []

		def mock_prm_call(config=None, demos=None, **kwargs):
			captured_prm_demos.append(demos)
			predictions = []
			batch_size = len(list(kwargs.values())[0]) if kwargs else 1
			for i in range(batch_size):
				mock_prediction = Mock()
				mock_prediction.completions = {
					EvaluationMetric.SOUNDNESS: 3.0 + (i * 0.1),
					EvaluationMetric.PROMISE: 2.0 + (i * 0.1),
					ReasoningState.FEEDBACK: f"Test PRM {i}",
				}
				predictions.append(mock_prediction)
			return predictions

		def mock_orm_call(config=None, demos=None, **kwargs):
			captured_orm_demos.append(demos)
			predictions = []
			batch_size = len(list(kwargs.values())[0]) if kwargs else 1
			for i in range(batch_size):
				mock_prediction = Mock()
				mock_prediction.completions = {
					NodeField.QUALITY: 4.0 + (i * 0.1),
					ReasoningState.FEEDBACK: f"Test ORM {i}",
				}
				predictions.append(mock_prediction)
			return predictions

		evaluator.process_evaluator = Mock(side_effect=mock_prm_call)
		evaluator.outcome_evaluator = Mock(side_effect=mock_orm_call)

		# Test 1: Base demos are used when no specific demos provided
		captured_prm_demos.clear()
		captured_orm_demos.clear()
		evaluator(states=[prm_state, orm_state], demos=base_demos)
		assert captured_prm_demos == [base_demos], "PRM should use base_demos"
		assert captured_orm_demos == [base_demos], "ORM should use base_demos"

		# Test 2: demos_prm overrides base demos for PRM
		captured_prm_demos.clear()
		captured_orm_demos.clear()
		evaluator(states=[prm_state, orm_state], demos=base_demos, demos_prm=prm_demos)
		assert captured_prm_demos == [prm_demos], "PRM should use prm_demos (override)"
		assert captured_orm_demos == [base_demos], "ORM should use base_demos (fallback)"

		# Test 3: demos_orm overrides base demos for ORM
		captured_prm_demos.clear()
		captured_orm_demos.clear()
		evaluator(states=[prm_state, orm_state], demos=base_demos, demos_orm=orm_demos)
		assert captured_prm_demos == [base_demos], "PRM should use base_demos (fallback)"
		assert captured_orm_demos == [orm_demos], "ORM should use orm_demos (override)"

		# Test 4: Both demos_prm and demos_orm override base demos
		captured_prm_demos.clear()
		captured_orm_demos.clear()
		evaluator(states=[prm_state, orm_state], demos=base_demos, demos_prm=prm_demos, demos_orm=orm_demos)
		assert captured_prm_demos == [prm_demos], "PRM should use prm_demos"
		assert captured_orm_demos == [orm_demos], "ORM should use orm_demos"

		# Test 5: Zero-shot when no demos provideds
		captured_prm_demos.clear()
		captured_orm_demos.clear()
		evaluator(states=[prm_state, orm_state])
		assert captured_prm_demos == [None], "PRM should use None (zero-shot)"
		assert captured_orm_demos == [None], "ORM should use None (zero-shot)"

	def test_full_symmetric_overrides(self, simple_reasoning_signature):
		"""Test using all symmetric override parameters together with field-level weights."""
		# Create custom signatures with rubric_weight in fields
		prm_sig_fields = {
			EvaluationMetric.CORRECTNESS: (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(5)],
				OutputField(desc="Correctness 1-5", rubric_weight=1.0),
			),
			"feedback": (str, OutputField(desc="Feedback")),
		}
		prm_sig = dspy.Signature(prm_sig_fields, "PRM rubric")

		orm_sig_fields = {
			"accuracy": (
				Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(100.0)],
				OutputField(desc="Accuracy 0-100", rubric_weight=1.0),
			),
			"feedback": (str, OutputField(desc="Feedback")),
		}
		orm_sig = dspy.Signature(orm_sig_fields, "ORM rubric")

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature_prm=prm_sig,
			evaluator_signature_orm=orm_sig,
		)

		# Verify all overrides are applied correctly
		assert evaluator.prm_rubric == prm_sig
		assert evaluator.orm_rubric == orm_sig
		assert evaluator.prm_dimension_weights == {"correctness": 1.0}
		assert evaluator.orm_dimension_weights == {"accuracy": 1.0}
		assert evaluator.is_custom_prm_signature is True
		assert evaluator.is_custom_orm_signature is True



# =============================================================================
# Shared GPU Model Fixture
# =============================================================================

# TODO[P2]: Make the tests below parameterized as well.
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
			verbosity=Verbosity.INFO,
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


# =============================================================================
# Integration Test Data Structures
# =============================================================================

class EvaluatorComparisonTest(NamedTuple):
	"""Test case comparing better vs worse examples for evaluator validation."""

	better_state: State
	worse_state: State
	test_type: str  # "PRM" or "ORM"
	description: str
	reasoning_analysis: str


def create_prm_comparison_tests() -> list[EvaluatorComparisonTest]:
	"""Create PRM (Process Reward Model) test pairs with better vs worse reasoning."""
	return [
		# Test 1: Complete vs incomplete algebraic reasoning
		EvaluatorComparisonTest(
			better_state=State(
				input={MathField.MATH_PROBLEM: "Solve 5x - 12 = 18"},
				reasoning={
					MathField.MATH_OPERATION: [
						"I need to isolate x by moving constants to one side.",
						"Adding 12 to both sides: 5x - 12 + 12 = 18 + 12",
						"This gives me: 5x = 30",
						"Dividing both sides by 5: x = 6",
					]
				},
				output={},
			),
			worse_state=State(
				input={MathField.MATH_PROBLEM: "Solve 5x - 12 = 18"},
				reasoning={MathField.MATH_OPERATION: ["I need to solve for x.", "5x = 18 + 12"]},
				output={},
			),
			test_type="PRM",
			description="Complete vs incomplete algebraic steps",
			reasoning_analysis="Better shows all steps with proper justification, worse jumps steps and lacks detail",
		),
		# Test 2: Correct vs incorrect geometric reasoning
		EvaluatorComparisonTest(
			better_state=State(
				input={
					MathField.MATH_PROBLEM: "Find the area of a triangle with base 8 and height 6"
				},
				reasoning={
					MathField.MATH_OPERATION: [
						"The formula for triangle area is A = (1/2) * base * height",
						"Substituting values: A = (1/2) * 8 * 6",
						"A = (1/2) * 48 = 24 square units",
					]
				},
				output={},
			),
			worse_state=State(
				input={
					MathField.MATH_PROBLEM: "Find the area of a triangle with base 8 and height 6"
				},
				reasoning={
					MathField.MATH_OPERATION: [
						"Triangle area is base times height",
						"So area = 8 * 6 = 48",
					]
				},
				output={},
			),
			test_type="PRM",
			description="Correct vs incorrect formula application",
			reasoning_analysis="Better uses correct triangle formula, worse omits the 1/2 factor",
		),
		# Test 3: Systematic vs unsystematic problem solving
		EvaluatorComparisonTest(
			better_state=State(
				input={
					MathField.MATH_PROBLEM: "A number increased by 15% becomes 92. Find the original number."
				},
				reasoning={
					MathField.MATH_OPERATION: [
						"Let the original number be x",
						"After 15% increase: x + 0.15x = 92",
						"Combining like terms: 1.15x = 92",
					]
				},
				output={},
			),
			worse_state=State(
				input={
					MathField.MATH_PROBLEM: "A number increased by 15% becomes 92. Find the original number."
				},
				reasoning={
					MathField.MATH_OPERATION: [
						"92 minus 15% should give the original number",
						"15% of 92 is about 14, so 92 - 14 = 78",
					]
				},
				output={},
			),
			test_type="PRM",
			description="Systematic vs unsystematic percentage problem approach",
			reasoning_analysis="Better uses proper algebraic setup, worse uses flawed reverse calculation",
		),
		# Test 4: Detailed vs superficial calculus reasoning
		EvaluatorComparisonTest(
			better_state=State(
				input={MathField.MATH_PROBLEM: "Find the derivative of f(x) = x³ - 4x^2 + 7"},
				reasoning={
					MathField.MATH_OPERATION: [
						"I'll apply the power rule to each term: d/dx(x^n) = nx^(n-1)",
						"For x³: derivative is 3x^2",
						"For -4x^2: derivative is -8x",
						"For constant 7: derivative is 0",
						"Therefore: f'(x) = 3x^2 - 8x",
					]
				},
				output={},
			),
			worse_state=State(
				input={MathField.MATH_PROBLEM: "Find the derivative of f(x) = x^3 - 4x^2 + 7"},
				reasoning={MathField.MATH_OPERATION: ["Using power rule", "f'(x) = 3x^2 - 8x"]},
				output={},
			),
			test_type="PRM",
			description="Detailed vs superficial calculus explanation",
			reasoning_analysis="Better explains each step and rule application, worse gives minimal explanation",
		),
	]


def create_orm_comparison_tests() -> list[EvaluatorComparisonTest]:
	"""Create ORM (Outcome Reward Model) test pairs with better vs worse solutions."""
	return [
		# Test 1: Correct vs incorrectly rounded answer
		EvaluatorComparisonTest(
			better_state=State(
				input={MathField.MATH_PROBLEM: "Convert 3/8 to a decimal"},
				reasoning={
					MathField.MATH_OPERATION: [
						"To convert fraction to decimal, I divide numerator by denominator",
						"3 ÷ 8 = 0.375",
					]
				},
				output={MathField.ANSWER: "3/8 = 0.375"},
			),
			worse_state=State(
				input={MathField.MATH_PROBLEM: "Convert 3/8 to a decimal"},
				reasoning={
					MathField.MATH_OPERATION: [
						"To convert fraction to decimal, I divide numerator by denominator",
						"3 ÷ 8 = 0.38",
					]
				},
				output={MathField.ANSWER: "3/8 = 0.38"},
			),
			test_type="ORM",
			description="Correct vs incorrect decimal conversion",
			reasoning_analysis="Better gives correct answer 0.375, worse has calculation error giving 0.38",
		),
		# Test 2: Correct vs incorrect final answer
		EvaluatorComparisonTest(
			better_state=State(
				input={MathField.MATH_PROBLEM: "Who won the 2012 Presidential Election in the US?"},
				reasoning={
					MathField.MATH_OPERATION: [
						"The 2012 election was between Barack Obama and Mitt Romney",
						"Barack Obama was the incumbent president running for re-election",
					]
				},
				output={MathField.ANSWER: "Barack Obama"},
			),
			worse_state=State(
				input={MathField.MATH_PROBLEM: "Who won the 2012 Presidential Election in the US?"},
				reasoning={
					MathField.MATH_OPERATION: [
						"The 2012 election was between Barack Obama and Mitt Romney",
						"Mitt Romney challenged the incumbent",
					]
				},
				output={MathField.ANSWER: "Mitt Romney"},
			),
			test_type="ORM",
			description="Correct vs incorrect factual answer",
			reasoning_analysis="Better gives correct winner (Obama), worse gives incorrect winner (Romney)",
		),
		# Test 3: Complete vs incomplete solution
		EvaluatorComparisonTest(
			better_state=State(
				input={
					MathField.MATH_PROBLEM: "A pizza is cut into 8 equal slices. If 3 slices are eaten, what fraction remains?"
				},
				reasoning={
					MathField.MATH_OPERATION: [
						"Total slices = 8, eaten = 3, remaining = 8 - 3 = 5",
						"Fraction remaining = 5/8",
					]
				},
				output={MathField.ANSWER: "5/8 of the pizza remains"},
			),
			worse_state=State(
				input={
					MathField.MATH_PROBLEM: "A pizza is cut into 8 equal slices. If 3 slices are eaten, what fraction remains?"
				},
				reasoning={MathField.MATH_OPERATION: ["8 - 3 = 5 slices remain"]},
				output={MathField.ANSWER: "5 slices remain"},
			),
			test_type="ORM",
			description="Complete vs incomplete fraction solution",
			reasoning_analysis="Better gives proper fraction answer, worse only gives count without fraction",
		),
		# Test 4: Well-justified vs poorly justified solution
		EvaluatorComparisonTest(
			better_state=State(
				input={MathField.MATH_PROBLEM: "Is 17 a prime number?"},
				reasoning={
					MathField.MATH_OPERATION: [
						"A prime number has exactly two factors: 1 and itself",
						"I need to check if 17 has any factors other than 1 and 17",
						"Testing divisors up to √17 ≈ 4.1: 2, 3, 4",
						"17 ÷ 2 = 8.5 (not divisible), 17 ÷ 3 = 5.67 (not divisible), 17 ÷ 4 = 4.25 (not divisible)",
					]
				},
				output={
					MathField.ANSWER: "Yes, 17 is prime because it has no divisors other than 1 and 17"
				},
			),
			worse_state=State(
				input={MathField.MATH_PROBLEM: "Is 17 a prime number?"},
				reasoning={
					MathField.MATH_OPERATION: ["17 is not divisible by small numbers like 2, 3"]
				},
				output={MathField.ANSWER: "Yes, 17 is prime"},
			),
			test_type="ORM",
			description="Well-justified vs poorly justified prime check",
			reasoning_analysis="Better shows systematic checking method, worse gives minimal justification",
		),
		# Test 5: Appropriately detailed vs too brief solution
		EvaluatorComparisonTest(
			better_state=State(
				input={
					MathField.MATH_PROBLEM: "Find the slope of the line through points (2, 5) and (7, 15)"
				},
				reasoning={
					MathField.MATH_OPERATION: [
						"Using slope formula: m = (y₂ - y₁)/(x₂ - x₁)",
						"Points: (2, 5) and (7, 15), so x₁=2, y₁=5, x₂=7, y₂=15",
						"m = (15 - 5)/(7 - 2) = 10/5 = 2",
					]
				},
				output={MathField.ANSWER: "The slope is 2"},
			),
			worse_state=State(
				input={
					MathField.MATH_PROBLEM: "Find the slope of the line through points (2, 5) and (7, 15)"
				},
				reasoning={MathField.MATH_OPERATION: []},
				output={MathField.ANSWER: "2"},
			),
			test_type="ORM",
			description="Appropriately detailed vs too brief slope calculation",
			reasoning_analysis="Better shows formula and substitution, worse gives bare answer with no work",
		),
		# Test 6: Correct units vs missing units
		EvaluatorComparisonTest(
			better_state=State(
				input={
					MathField.MATH_PROBLEM: "A car travels 150 miles in 3 hours. What is its average speed?"
				},
				reasoning={
					MathField.MATH_OPERATION: [
						"Average speed = total distance / total time",
						"Speed = 150 miles / 3 hours = 50 miles per hour",
					]
				},
				output={MathField.ANSWER: "The average speed is 50 mph"},
			),
			worse_state=State(
				input={
					MathField.MATH_PROBLEM: "A car travels 150 miles in 3 hours. What is its average speed?"
				},
				reasoning={
					MathField.MATH_OPERATION: ["Speed = distance / time", "Speed = 150 / 3 = 50"]
				},
				output={MathField.ANSWER: "The average speed is 50"},
			),
			test_type="ORM",
			description="Correct units vs missing units in speed calculation",
			reasoning_analysis="Better includes proper units (mph), worse omits units from final answer",
		),
	]


def create_argument_example_states() -> list[State]:
	"""Create states for testing the evaluator on an argument generation task."""
	return [
		# Process evaluation - argument with reasoning steps
		State(
			input={ArgumentField.TOPIC: "Renewable energy", ArgumentField.STANCE: ArgumentStance.PRO},
			reasoning={
				ArgumentField.CLAIM: [
					"Renewable energy sources like solar and wind are becoming increasingly cost-competitive with fossil fuels."
				]
			},
			output={},
		),
		# Outcome evaluation - complete argument
		State(
			input={ArgumentField.TOPIC: "Universal basic income", ArgumentField.STANCE: ArgumentStance.PRO},
			reasoning={
				ArgumentField.CLAIM: [
					"Universal basic income provides economic security in an era of automation.",
					"It simplifies welfare systems and reduces administrative costs.",
					"Pilot programs have shown positive effects on mental health and employment.",
				]
			},
			output={
				"argument": "Universal basic income (UBI) offers a pragmatic solution to economic challenges posed by automation and job displacement. By providing a financial safety net, UBI enables individuals to pursue education, entrepreneurship, or caregiving without the fear of destitution. Evidence from pilot programs in Finland and Kenya demonstrates that UBI recipients experience reduced stress, improved health outcomes, and sustained employment rates. Furthermore, UBI streamlines bureaucratic welfare systems, reducing overhead costs while ensuring no one falls through the cracks."
			},
		),
		# Another complete argument
		State(
			input={ArgumentField.TOPIC: "Remote work policies", ArgumentField.STANCE: ArgumentStance.PRO},
			reasoning={
				ArgumentField.CLAIM: [
					"Remote work increases employee productivity and satisfaction.",
					"It reduces commuting time and environmental impact.",
					"Companies save on office space and overhead costs.",
				]
			},
			output={
				"argument": "Remote work policies represent a win-win for both employers and employees. Studies consistently show that remote workers are more productive, with fewer distractions and greater autonomy over their schedules. Employees benefit from eliminating lengthy commutes, gaining hours for personal pursuits and reducing their carbon footprint. Meanwhile, companies can reduce real estate costs and access a global talent pool unrestricted by geography."
			},
		),
	]


@pytestmark_gpu
class TestEvaluatorIntegration:
	"""Integration tests for the evaluator using real models (requires GPU)."""

	@pytest.fixture
	def local_lm(self, shared_gpu_model):
		"""Use the shared GPU model fixture."""
		return shared_gpu_model

	@pytest.fixture
	def evaluator(self, local_lm):
		"""Create an evaluator instance with real LM."""
		dspy.settings.configure(lm=local_lm)
		return TreeOfThoughtEvaluator(
			generator_signature=SolveMathProblemWithReasoning,
			consider_reasoning_in_final_eval=True,
			verbosity=Verbosity.INFO,
		)

	def validate_comparison(
		self,
		test: EvaluatorComparisonTest,
		better_result: list[list[EvaluationResult]],
		worse_result: list[list[EvaluationResult]],
	) -> None:
		"""Validate if evaluator correctly scored better example higher than worse example."""
		def extract_score(result):
			if result and len(result) > 0 and len(result[0]) > 0:
				eval_result = result[0][0]
				if isinstance(eval_result, EvaluationResult):
					score = eval_result.score
					return float(score) if isinstance(score, int | float) else None
			return None

		better_score = extract_score(better_result)
		worse_score = extract_score(worse_result)

		assert better_score is not None, "Could not extract score for better example"
		assert worse_score is not None, "Could not extract score for worse example"

		assert better_score > worse_score, (
			f"Expected better score ({better_score}) > worse score ({worse_score}). "
			f"Analysis: {test.reasoning_analysis}"
		)

	@pytest.mark.parametrize("test_case", create_prm_comparison_tests())
	def test_prm_semantic_validation(self, evaluator, test_case):
		"""Test PRM semantic validation (better vs worse reasoning)."""
		try:
			better_result = evaluator(
				states=test_case.better_state,
				n_samples_evaluator=1,
				demos=PRM_DEMOS,
			)
			worse_result = evaluator(
				states=test_case.worse_state,
				n_samples_evaluator=1,
				demos=PRM_DEMOS,
			)
			self.validate_comparison(test_case, better_result, worse_result)
		except Exception as e:
			pytest.fail(f"PRM validation failed: {e}")

	@pytest.mark.parametrize("test_case", create_orm_comparison_tests())
	def test_orm_semantic_validation(self, evaluator, test_case):
		"""Test ORM semantic validation (better vs worse solution)."""
		try:
			better_result = evaluator(
				states=test_case.better_state,
				n_samples_evaluator=1,
				demos=ORM_DEMOS,
			)
			worse_result = evaluator(
				states=test_case.worse_state,
				n_samples_evaluator=1,
				demos=ORM_DEMOS,
			)
			self.validate_comparison(test_case, better_result, worse_result)
		except Exception as e:
			pytest.fail(f"ORM validation failed: {e}")

	def test_argument_evaluator_execution(self, local_lm):
		"""Test execution of argument evaluators."""
		dspy.settings.configure(lm=local_lm)

		argument_evaluator = TreeOfThoughtEvaluator(
			generator_signature=GenerateArgumentWithReasoning,
			evaluator_signature=ArgumentEvaluatorMultiDimensional,
			consider_reasoning_in_final_eval=True,
			verbosity=Verbosity.INFO,
		)

		states = create_argument_example_states()

		try:
			results = argument_evaluator(
				states=states[:1],  # Just test one for speed
				n_samples_evaluator=1,
				evaluator_temperature=0.7,
			)
			assert results is not None
			assert len(results) > 0
			assert isinstance(results[0][0], EvaluationResult)
		except Exception as e:
			pytest.fail(f"Argument evaluator execution failed: {e}")


if __name__ == "__main__":
	import sys
	sys.exit(pytest.main(["-vv", __file__]))

