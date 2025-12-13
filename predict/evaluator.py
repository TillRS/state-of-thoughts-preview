"""
TreeOfThoughtEvaluator: Simplified evaluator for Tree-of-Thought reasoning.

This module implements the TreeOfThoughtEvaluator class for Tree-of-Thought reasoning.
Uses three specialized LocalPredict instances with auto-batching for efficient evaluation.
"""

import copy
import enum
import logging
from typing import Annotated, Any

# Third-party imports
import annotated_types
import dspy
import numpy as np
import pydantic
from dspy.primitives.prediction import Prediction

# Local imports
from constants import VERBOSITY_TO_LOGGING_LEVEL, Verbosity
from lm.lm_constants import SamplingParam
from misc_utils import format_list_of_fields
from predict.local_predict import LocalPredict
from signatures import (
	ReasoningSignature,
	ensure_reasoning_signature,
)
from signatures.field_constants import EvaluationMetric
from tree import (
	EvaluationResult,
	JudgeEvaluation,
	State,
)
from tree.tree_constants import (
	NodeField,
	ReasoningState,
	ToTField,
)

logger = logging.getLogger(__name__)


class EvaluationType(enum.Enum):
	PROCESS = "process"  # PRM - Process Reward Model
	OUTCOME = "outcome"  # ORM - Outcome Reward Model


class TreeOfThoughtEvaluator(dspy.Module):
	"""
	A module for evaluating reasoning steps and final outputs in Tree-of-Thought reasoning.

	Uses three specialized LocalPredict instances for different evaluation contexts:
	- First Step: Evaluates initial reasoning step quality and promise
	- Subsequent Step: Evaluates reasoning progression and coherence
	- Final Step: Evaluates final response quality and completeness

	Supports efficient batch processing and state management for Tree-of-Thought workflows.
	"""

	# TODO[P2][Till]: Add support for voting across multiple evaluations
	# Potential mixed approach: scores for preselection, then 'rank' to select top k from preselected
	# TODO[P3][Till]: Create custom adapter to support both PRM and ORM calls in a single batch (similar to generator)
	def __init__(
		self,
		generator_signature: type[ReasoningSignature],
		evaluator_signature: type[ReasoningSignature] | None = None,
		evaluator_signature_prm: type[ReasoningSignature] | None = None,
		evaluator_signature_orm: type[ReasoningSignature] | None = None,
		consider_reasoning_in_final_eval: bool = True,
		verbosity: Verbosity = Verbosity.WARNING,
	) -> None:
		"""
		Initialize the TreeOfThoughtEvaluator.

		Parameters:
		    generator_signature (ReasoningSignature): The signature of the generator being evaluated.
		    evaluator_signature (ReasoningSignature | None): Default custom evaluator signature (rubric)
		        defining evaluation criteria and score dimensions for both PRM and ORM evaluation.
		        If None, uses default signatures (soundness+promise for PRM, quality for ORM).
		        This signature should contain only output fields (score dimensions). Use the
		        rubric_weight parameter in OutputField() to specify dimension weights.
		    evaluator_signature_prm (ReasoningSignature | None): Optional custom evaluator signature (rubric)
		        specifically for PRM (Process Reward Model) evaluation. If provided, overrides
		        evaluator_signature for PRM only. If None, PRM uses evaluator_signature (or default).
		        Use rubric_weight in OutputField() to specify dimension weights.
		    evaluator_signature_orm (ReasoningSignature | None): Optional custom evaluator signature (rubric)
		        specifically for ORM (Outcome Reward Model) evaluation. If provided, overrides
		        evaluator_signature for ORM only. If None, ORM uses evaluator_signature (or default).
		        Use rubric_weight in OutputField() to specify dimension weights.
		    consider_reasoning_in_final_eval (bool): Whether to include reasoning in outcome evaluation.
		    verbosity (Verbosity): Verbosity level for logging (Verbosity enum).

		Note:
		    - Demos must be provided explicitly via the demos parameter in forward()/call.
		      If no demos are provided, zero-shot evaluation is used.
		    - Dimension weights are specified directly in the signature using the rubric_weight parameter:
		        soundness: int = dspy.OutputField(desc="...", ge=1, le=7, rubric_weight=0.7)
		        promise: int = dspy.OutputField(desc="...", ge=1, le=7, rubric_weight=0.3)
		      If no rubric_weight is specified, equal weights are used for all dimensions.
		"""
		super().__init__()

		self.generator_signature = ensure_reasoning_signature(generator_signature)
		self.consider_reasoning_in_final_eval = consider_reasoning_in_final_eval
		self._verbosity = verbosity

		# Set logger level based on verbosity
		logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL[verbosity])

		# Extract field information
		self.generator_input_field_names = list(
			self.generator_signature.input_fields.keys()
		)
		self.generator_output_field_names = list(
			self.generator_signature.output_fields.keys()
		)
		self.reasoning_field_name = list(self.generator_signature.reasoning_fields.keys())[0]
		self.reasoning_field = self.generator_signature.reasoning_fields.get(
			self.reasoning_field_name
		)

		# Process evaluator signatures (custom or default)
		# Determine PRM rubric: evaluator_signature_prm takes precedence, then evaluator_signature, then default
		# Type annotation: rubrics are always dspy.Signature instances after normalization
		self.prm_rubric: dspy.Signature
		self.orm_rubric: dspy.Signature

		if evaluator_signature_prm is not None:
			self.prm_rubric = dspy.ensure_signature(evaluator_signature_prm)
			self.is_custom_prm_signature = True
		elif evaluator_signature is not None:
			# Backward compatibility: use evaluator_signature for PRM if no specific PRM signature provided
			self.prm_rubric = dspy.ensure_signature(evaluator_signature)
			self.is_custom_prm_signature = True
		else:
			# Use default PRM rubric (soundness + promise)
			self.prm_rubric = self._create_default_prm_rubric()
			self.is_custom_prm_signature = False

		# Determine ORM rubric: evaluator_signature_orm → evaluator_signature → default
		if evaluator_signature_orm is not None:
			self.orm_rubric = dspy.ensure_signature(evaluator_signature_orm)
			self.is_custom_orm_signature = True
		elif evaluator_signature is not None:
			# Backward compatibility: use evaluator_signature for ORM
			self.orm_rubric = dspy.ensure_signature(evaluator_signature)
			self.is_custom_orm_signature = True
		else:
			# Use default ORM rubric (quality)
			self.orm_rubric = self._create_default_orm_rubric()
			self.is_custom_orm_signature = False

		# Process output fields separately for PRM and ORM rubrics
		# Extract and store field metadata for PRM (including normalized weights)
		(
			self.prm_numeric_score_fields,
			self.prm_feedback_field,
			self.prm_dimension_bounds,
			self.prm_dimension_weights,
		) = self._process_evaluator_output_fields(self.prm_rubric)

		# Extract and store field metadata for ORM (including normalized weights)
		(
			self.orm_numeric_score_fields,
			self.orm_feedback_field,
			self.orm_dimension_bounds,
			self.orm_dimension_weights,
		) = self._process_evaluator_output_fields(self.orm_rubric)

		# Initialize PRM and ORM evaluator predictors
		self.process_evaluator = LocalPredict(
			signature=self._create_process_evaluator_signature(),
			verbose=verbosity,
		)
		self.outcome_evaluator = LocalPredict(
			signature=self._create_outcome_evaluator_signature(),
			verbose=verbosity,
		)

	@property
	def verbosity(self) -> Verbosity:
		"""Get the current verbosity level."""
		return self._verbosity

	@verbosity.setter
	def verbosity(self, verbosity: Verbosity) -> None:
		"""Set the verbosity level and update logger."""
		self._verbosity = verbosity
		logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL[verbosity])

	def _create_default_prm_rubric(self) -> dspy.Signature:
		"""
		Create the default PRM (Process Reward Model) rubric with soundness and promise scores.

		PRM evaluates reasoning trajectories, so we need dual scores:
		- soundness: backward-looking correctness (is this step logically valid?)
		- promise: forward-looking trajectory quality (is this leading toward a solution?)

		Returns:
		    dspy.Signature: A signature with soundness, promise (both 1-7 Likert), and feedback output fields.
		"""
		signature_fields = {}

		# Add soundness output field with bounds metadata
		signature_fields[EvaluationMetric.SOUNDNESS] = (
			Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
			dspy.OutputField(
				desc="Soundness score (backward-looking correctness) on a 1 to 7 Likert scale",
			),
		)

		# Add promise output field with bounds metadata
		signature_fields[EvaluationMetric.PROMISE] = (
			Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
			dspy.OutputField(
				desc="Promise score (forward-looking trajectory quality) on a 1 to 7 Likert scale",
			),
		)

		# Add feedback field
		signature_fields[ReasoningState.FEEDBACK] = (
			str,
			dspy.OutputField(
				desc="Detailed feedback explaining the assigned scores", annotation=str
			),
		)

		# Expanded rubric descriptions for both dimensions
		soundness_descriptions = """
Soundness (backward-looking correctness) - Evaluate logical validity, factual accuracy, and coherence:

7 = Excellent: Flawless logic, perfect factual accuracy, and seamless coherence with problem context
	All claims are well-supported, no errors or inconsistencies present

6 = Strong: Logically sound with only very minor imperfections that don't affect overall correctness
	Strong factual accuracy and coherence throughout

5 = Good: Generally sound reasoning with mostly good coherence
	May have small gaps but no significant logical errors

4 = Adequate: Mostly correct but with notable gaps in logic, accuracy, or coherence
	Contains minor errors or unclear connections that detract from soundness

3 = Below Average: Significant logical errors, factual inaccuracies, or coherence problems
	Multiple issues that undermine the correctness of the reasoning

2 = Weak: Major flaws in logic or facts that seriously undermine validity
	Poor coherence with context or previous steps, substantial unsupported claims

1 = Poor: Fundamentally invalid or incoherent reasoning
	Contains critical logical errors, false claims, or complete disconnect from problem context
""".strip()

		promise_descriptions = """
Promise (forward-looking trajectory quality) - Assess likelihood of reaching a high-quality solution:

7 = Excellent: Optimal trajectory with clear path to high-quality solution or
already demonstrates a complete and correct solution

6 = Strong: Very promising approach with high likelihood of success
    Demonstrates strong progress along the most promising direction

5 = Good: Reasonable trajectory with decent solution potential
    Shows meaningful progress without major detours or inefficiencies

4 = Adequate: Acceptable direction but suboptimal or meandering
    Makes some progress but could be more direct or efficient

3 = Below Average: Questionable trajectory with unclear progress toward solution
    May be pursuing unproductive tangents or showing limited advancement

2 = Weak: Poor approach that is unlikely to lead to strong solution
    Appears stuck, inefficient, or heading in wrong direction

1 = Poor: Dead-end or counterproductive path
    Arrived at an incorrect solution, completely stuck, or actively moving away from solution
""".strip()

		quality_patterns = """
Look out for the following patterns that may indicate issues:
- Repetition of information without adding value
- Incoherent or irrelevant information
- Lack of clarity or specificity
- Incomplete or incorrect information
- Failure to follow the task instructions
- Use of unsupported or unverifiable claims
- Presence of logical fallacies or contradictions
- Getting stuck in unproductive loops
- Pursuing tangents that don't advance toward the solution
""".strip()

		instructions = (
			"Evaluate the reasoning using TWO distinct dimensions:\n\n"
			+ soundness_descriptions
			+ "\n\n"
			+ promise_descriptions
			+ "\n\n"
			+ quality_patterns
			+ "\n\nProvide separate numeric scores for soundness and promise based on the rubrics above."
		)

		return dspy.Signature(signature_fields, instructions)

	def _create_default_orm_rubric(self) -> dspy.Signature:
		"""
		Create the default ORM (Outcome Reward Model) rubric with single unified quality score.

		ORM evaluates final solutions, where we assess overall correctness and quality
		with a single holistic score rather than separate soundness/promise dimensions.

		Returns:
		    dspy.Signature: A signature with quality (1-7 Likert) and feedback output fields.
		"""
		signature_fields = {}

		# Add quality output field with bounds metadata
		signature_fields[NodeField.QUALITY] = (
			Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
			dspy.OutputField(
				desc="Quality score for the final solution on a 1 to 7 Likert scale",
			),
		)

		# Add feedback field
		signature_fields[ReasoningState.FEEDBACK] = (
			str,
			dspy.OutputField(
				desc="Detailed feedback explaining the assigned score", annotation=str
			),
		)

		# Unified quality rubric for final solution evaluation
		quality_descriptions = """
Quality (solution correctness and completeness) - Evaluate the final solution holistically:

7 = Excellent: Perfect or near-perfect solution that is correct, complete, and clearly presented
    Demonstrates flawless logic, accurate calculations, and excellent communication

6 = Strong: Very good solution with only very minor imperfections
    Correct and well-presented with strong clarity throughout

5 = Good: Generally correct solution with mostly good presentation
    May have small presentation issues but no significant errors

4 = Adequate: Mostly correct but with notable presentation issues or minor errors
    Solution is fundamentally sound but lacks clarity or has minor inaccuracies

3 = Below Average: Partially correct with significant errors or incomplete reasoning
    Contains mistakes that undermine the solution's correctness

2 = Weak: Largely incorrect with major errors in logic, calculation, or understanding
    Fundamental flaws that invalidate most of the solution

1 = Poor: Fundamentally wrong or completely missing the point
    Critical errors, wrong formula/approach, or complete misunderstanding of the problem
""".strip()

		quality_patterns = """
Consider these aspects when evaluating solution quality:
- Correctness: Is the final answer correct? Are calculations accurate?
- Completeness: Does it fully address the problem? Are there missing steps?
- Clarity: Is the solution well-explained and easy to follow?
- Methodology: Is the approach appropriate for the problem?
- Verification: Does the solution include units, proper notation, etc.?

Look out for patterns that indicate issues:
- Incorrect formulas or conceptual misunderstandings
- Arithmetic or algebraic errors
- Missing or incomplete solution steps
- Lack of proper units or notation
- Unclear or confusing presentation
- Failure to answer the actual question asked
""".strip()

		instructions = (
			"Evaluate the final solution using a single holistic quality score:"
			+ "\n\n"
			+ quality_descriptions
			+ "\n\n"
			+ quality_patterns
			+ "\n\n"
			+ "Provide a numeric quality score based on the rubric above."
		)

		return dspy.Signature(signature_fields, instructions)

	def _process_evaluator_output_fields(
		self,
		signature: type[dspy.Signature],
		default_lower_bound: int = 1,
		default_upper_bound: int = 7,
	) -> tuple[list[str], str, dict[str, tuple[float, float]], dict[str, float]]:
		"""
		Process evaluator output fields to identify, validate, and extract metadata.

		This method does a single pass through output fields to:
		1. Identify numeric score fields and string feedback field
		2. Validate at least one of each exists
		3. Extract bounds from metadata (or use defaults if not specified)
		4. Extract and validate rubric_weight from field metadata
		5. Generate equal weights if no rubric_weight specified

		Parameters:
		    signature (dspy.Signature): The evaluator signature to process.
		    default_lower_bound (int): Default lower bound if not specified in metadata.
		    default_upper_bound (int): Default upper bound if not specified in metadata.

		Returns:
		    Tuple[List[str], str, Dict[str, Tuple[int, int]], Dict[str, float]]:
		        - List of numeric score field names
		        - Feedback field name (string field)
		        - Dictionary mapping numeric field names to (lower, upper) bounds
		        - Dictionary mapping numeric field names to normalized weights (sum to 1.0)

		Raises:
			AssertionError: If evaluator signature does not have a single string feedback field
				and at least one numeric (int or float) score field, or if rubric_weight is
				specified for some but not all numeric fields.

		"""
		numeric_score_fields = []
		feedback_fields = []
		dimension_bounds = {}
		dimension_weights = {}

		for name, field in signature.output_fields.items():
			field: pydantic.fields.FieldInfo
			field_type = field.annotation

			if field_type in (float, int):
				numeric_score_fields.append(name)

				# Extract bounds from metadata, or use defaults if not present
				ge_bound = next(
					(meta.ge for meta in field.metadata if hasattr(meta, "ge")),
					default_lower_bound,
				)
				le_bound = next(
					(meta.le for meta in field.metadata if hasattr(meta, "le")),
					default_upper_bound,
				)
				dimension_bounds[name] = (ge_bound, le_bound)
				assert ge_bound < le_bound, (
					"Lower bound must be less than upper bound, "
					f"but got ge={ge_bound} and le={le_bound} for field '{name}'."
				)

				# Extract rubric_weight from json_schema_extra if present
				if hasattr(field, "json_schema_extra") and field.json_schema_extra:
					rubric_weight = field.json_schema_extra.get("rubric_weight")
					if rubric_weight is not None:
						dimension_weights[name] = float(rubric_weight)

			elif field_type is str:
				feedback_fields.append(name)

		# Validate required fields exist
		assert numeric_score_fields, (
			"The evaluator signature must have at least one numeric output field (float or int) "
			"for scoring."
		)
		assert feedback_fields and len(feedback_fields) == 1, (
			"The evaluator signature must have exactly one string output field for feedback."
		)

		# Process rubric_weights: either all fields have them, or none do
		if dimension_weights:
			# Some fields have rubric_weight - validate ALL fields have it
			missing_weights = set(numeric_score_fields) - set(dimension_weights.keys())
			assert not missing_weights, (
				f"If rubric_weight is specified for any dimension, it must be specified for all. "
				f"Missing rubric_weight for: {missing_weights}"
			)
			# Validate all weights are positive
			assert all(w > 0 for w in dimension_weights.values()), (
				f"All rubric_weight values must be positive. Given weights: {dimension_weights}"
			)
			# Normalize weights to sum to 1.0
			total = sum(dimension_weights.values())
			dimension_weights = {k: v / total for k, v in dimension_weights.items()}
		else:
			# No rubric_weight specified - use equal weights
			n = len(numeric_score_fields)
			dimension_weights = dict.fromkeys(numeric_score_fields, 1.0 / n)

		return numeric_score_fields, feedback_fields[0], dimension_bounds, dimension_weights

	def _create_process_evaluator_signature(self) -> dspy.Signature:
		"""
		Create signature for PRM (Process Reward Model) - evaluating reasoning step quality.

		Uses the PRM rubric to construct a PRM-specific signature by adding
		generator input fields and reasoning_trajectory field. Default PRM rubric uses
		soundness + promise dual scores.

		Returns:
		    dspy.Signature: PRM evaluation signature with inputs and score dimensions from PRM rubric.
		"""
		generator_inputs = format_list_of_fields(self.generator_input_field_names)
		generator_outputs = format_list_of_fields(self.generator_output_field_names)

		# Build instructions for PRM evaluation
		prm_rubric_instructions = self.prm_rubric.__doc__ or ""

		# Get numeric fields from PRM rubric
		prm_numeric_fields, prm_feedback_field, _, _ = self._process_evaluator_output_fields(
			self.prm_rubric
		)
		numeric_fields_str = ", ".join(prm_numeric_fields)

		reasoning_step_description = (
			self.reasoning_field.description
			if self.reasoning_field and self.reasoning_field.description
			else "a step toward solving the problem"
		)
		prm_instructions = f"""
Judge the quality of reasoning steps in a problem-solving process.
The problem requires producing {generator_outputs} given {generator_inputs}.
Reasoning steps are stored in `{self.reasoning_field_name}`.
A reasoning step is: {reasoning_step_description}

Additional steps may follow, so focus on reasoning quality and trajectory rather than completeness.
{prm_rubric_instructions}

For the numeric score fields ({numeric_fields_str}), provide only a numeric value.
Explanatory text should go in `{prm_feedback_field}`.
""".strip()

		# Create signature fields dictionary
		signature_fields = {}

		# Add base input fields from generator
		for name, field in self.generator_signature.input_fields.items():
			field_type = (
				field.annotation
				if hasattr(field, "annotation") and field.annotation
				else str
			)
			signature_fields[name] = (field_type, field)

		# Extract type and description from generator's reasoning field for more precise specification
		reasoning_element_type = self.reasoning_field.annotation or str
		reasoning_description = self.reasoning_field.description or "reasoning step"
		signature_fields[ToTField.REASONING_STEPS] = (
			list[reasoning_element_type],
			dspy.InputField(
				desc=f"List of '{reasoning_description}'s to evaluate toward producing {generator_outputs}",
				annotation=list[reasoning_element_type],
			),
		)

		# Add output fields from PRM rubric
		for name, field in self.prm_rubric.output_fields.items():
			field_type = field.annotation
			signature_fields[name] = (field_type, field)

		return dspy.Signature(signature_fields, prm_instructions)

	def _create_outcome_evaluator_signature(self) -> dspy.Signature:
		"""
		Create signature for ORM (Outcome Reward Model) - evaluating final solution quality.

		Uses the ORM rubric to construct an ORM-specific signature by adding
		generator input fields, generator output fields, and optionally reasoning_trajectory field.
		Default ORM rubric uses a single unified quality score.

		Returns:
		    dspy.Signature: ORM evaluation signature with inputs and score dimensions from ORM rubric.
		"""
		generator_inputs = format_list_of_fields(self.generator_input_field_names)
		generator_outputs = format_list_of_fields(self.generator_output_field_names)

		# Build instructions for ORM evaluation
		orm_rubric_instructions = self.orm_rubric.__doc__ or ""
		orm_evaluation_target = (
			"final solution"
			if not self.consider_reasoning_in_final_eval
			else "final solution including reasoning steps"
		)

		# Get numeric fields from ORM rubric
		orm_numeric_fields, orm_feedback_field, _, _ = self._process_evaluator_output_fields(
			self.orm_rubric
		)
		numeric_fields_str = ", ".join(orm_numeric_fields)

		reasoning_context_note = (
			f"\n\nThe reasoning steps that led to this solution are stored in `{ToTField.REASONING_STEPS}` for context."
			if self.consider_reasoning_in_final_eval
			else ""
		)
		orm_instructions = f"""
Judge the quality of a {orm_evaluation_target} to the problem at hand.
The problem requires producing {generator_outputs} given {generator_inputs}.
Consider correctness, completeness, clarity, and overall solution quality.
Make sure to fact check claims or aspects that don't seem immediately obvious, may be untrue, or may contain a mistake.
Penalize any omissions, inaccuracies, or inclusion of irrelevant information in the final solution.
{orm_rubric_instructions}

For the numeric score fields ({numeric_fields_str}), provide only a numeric value. Explanatory text should go in `{orm_feedback_field}`.{reasoning_context_note}
""".strip()

		# Create signature fields dictionary
		signature_fields = {}

		# Add base input fields from generator
		for name, field in self.generator_signature.input_fields.items():
			field_type = (
				field.annotation
				if hasattr(field, "annotation") and field.annotation
				else str
			)
			signature_fields[name] = (field_type, field)

		# Add generator output fields as inputs for ORM evaluation
		for field_name, field_info in self.generator_signature.output_fields.items():
			field_type = (
				field_info.annotation
				if hasattr(field_info, "annotation") and field_info.annotation
				else str
			)
			signature_fields[field_name] = (
				field_type,
				dspy.InputField(
					desc=f"The {field_name} solution to evaluate", annotation=field_type
				),
			)

		# Add reasoning steps if configured (as optional field)
		if self.consider_reasoning_in_final_eval:
			# Extract type and description from generator's reasoning field for more precise specification
			reasoning_element_type = self.reasoning_field.annotation or str
			reasoning_description = self.reasoning_field.description or "reasoning step"
			signature_fields[ToTField.REASONING_STEPS] = (
				list[reasoning_element_type],
				dspy.InputField(
					desc=f"List of '{reasoning_description}'s that led to the final {generator_outputs}",
					annotation=list[reasoning_element_type],
					default=[],
				),
			)

		# Add output fields from ORM rubric
		for name, field in self.orm_rubric.output_fields.items():
			field_type = field.annotation
			signature_fields[name] = (field_type, field)

		return dspy.Signature(signature_fields, orm_instructions)

	def _normalize_score(
		self, score: float, lower_bound: float, upper_bound: float
	) -> float:
		"""
		Normalize a score to the range [0, 1].

		Parameters:
		    score (float): The score to normalize.
		    lower_bound (float): The lower bound of the score range.
		    upper_bound (float): The upper bound of the score range.

		Returns:
		    float: The normalized score in [0, 1].
		"""
		return (score - lower_bound) / (upper_bound - lower_bound)

	def _consolidate_scores(
		self, predictions: list[Prediction], rubric: dspy.Signature
	) -> list[EvaluationResult]:
		"""
		Consolidate scores across multiple judges and dimensions for each prediction.

		When using multiple dimensions or multiple judges, this method:
		1. Extracts scores for each dimension across all judges
		2. Computes average score per dimension
		3. Normalizes scores using dimension bounds
		4. Computes a total score as the weighted average across all dimensions (using dimension_weights)
		5. Preserves individual judge evaluations with both raw and normalized scores

		Parameters:
		    predictions (List[Prediction]): Predictions from the evaluator.
		    rubric (dspy.Signature): The rubric signature (PRM or ORM) used for this evaluation.

		Returns:
		    List[EvaluationResult]: List of consolidated evaluation results.
		        Each EvaluationResult contains:
		        - score: consolidated total score (weighted average across dimensions and judges)
		        - judge_evaluations: list of JudgeEvaluation instances with raw and normalized scores
		"""
		output_results = []

		# Extract field information from the rubric
		numeric_score_fields, feedback_field, dimension_bounds, _ = (
			self._process_evaluator_output_fields(rubric)
		)

		# Determine which rubric this is (PRM or ORM) and use appropriate dimension weights
		# We compare the rubric against self.prm_rubric to determine which one it is
		if rubric == self.prm_rubric:
			# This is a PRM evaluation - use PRM dimension weights
			rubric_dimension_weights = self.prm_dimension_weights
		elif rubric == self.orm_rubric:
			# This is an ORM evaluation - use ORM dimension weights
			rubric_dimension_weights = self.orm_dimension_weights
		else:
			# Fallback: use equal weights (should not happen in normal usage)
			n = len(numeric_score_fields)
			rubric_dimension_weights = dict.fromkeys(numeric_score_fields, 1.0 / n)

		for prediction in predictions:
			# Get completions for all dimensions
			completions = prediction.completions

			# Use numeric score fields from the rubric
			score_dimension_names = [
				dim for dim in numeric_score_fields if dim in completions
			]

			# Extract feedback from all judges using feedback field from rubric
			feedback_completions = completions[feedback_field]
			if not isinstance(feedback_completions, list):
				feedback_completions = [feedback_completions]

			# Determine number of judges (n_samples_evaluator)
			n_judges = len(feedback_completions)

			# Consolidate scores across judges for each dimension
			dimension_raw_scores_by_judge = {dim: [] for dim in score_dimension_names}
			dimension_normalized_scores_by_judge = {
				dim: [] for dim in score_dimension_names
			}

			for dimension in score_dimension_names:
				dimension_scores = completions[dimension]
				if not isinstance(dimension_scores, list):
					dimension_scores = [dimension_scores]

				dimension_scores = [float(s) for s in dimension_scores]

				# Validate that all scores are within bounds (use local dimension_bounds from rubric)
				lower, upper = dimension_bounds[dimension]
				for score in dimension_scores:
					assert lower <= score <= upper, (
						f"Score for dimension '{dimension}' is out of bounds: {score} "
						f"(expected range: [{lower}, {upper}])"
					)

				# Store scores per judge
				dimension_raw_scores_by_judge[dimension] = dimension_scores
				dimension_normalized_scores_by_judge[dimension] = [
					self._normalize_score(score, lower, upper)
					for score in dimension_scores
				]

			assert dimension_raw_scores_by_judge, (
				"No numeric evaluator score dimensions found for consolidation."
			)

			# Compute total score: weighted average across dimensions, then averaged across judges
			# For each judge, compute weighted score across dimensions
			judge_weighted_scores = []
			for judge_idx in range(n_judges):
				# Get normalized score for each dimension for this judge
				judge_dimension_scores = {
					dim: dimension_normalized_scores_by_judge[dim][judge_idx]
					for dim in score_dimension_names
				}
				# Compute weighted sum for this judge using rubric-specific weights
				weighted_sum = sum(
					judge_dimension_scores[dim] * rubric_dimension_weights[dim]
					for dim in score_dimension_names
				)
				judge_weighted_scores.append(weighted_sum)

			# Average weighted scores across all judges
			total_score = float(np.mean(judge_weighted_scores))

			# Build individual judge evaluations with JudgeEvaluation dataclass
			judge_evaluations = []
			for judge_idx in range(n_judges):
				raw_scores = {
					dimension: dimension_raw_scores_by_judge[dimension][judge_idx]
					for dimension in score_dimension_names
				}
				normalized_scores = {
					dimension: dimension_normalized_scores_by_judge[dimension][
						judge_idx
					]
					for dimension in score_dimension_names
				}
				judge_eval = JudgeEvaluation(
					raw_scores=raw_scores,
					normalized_scores=normalized_scores,
					feedback=feedback_completions[judge_idx],
				)
				judge_evaluations.append(judge_eval)

			evaluation_result = EvaluationResult(
				score=total_score,
				judge_evaluations=judge_evaluations,
			)

			output_results.append(evaluation_result)

		return output_results

	def _state_to_evaluator_input(
		self, state: State, evaluation_type: EvaluationType
	) -> dict[str, Any]:
		"""
		Convert the state to input for the evaluator.

		Parameters:
		    state (State): The state to convert.
		    evaluation_type (str): The evaluation type (process or outcome).

		Returns:
		    Dict[str, Any]: The input dictionary for the evaluator.
		"""
		# Start with base input fields
		evaluator_input = copy.deepcopy(state.input)

		# Get reasoning steps - use reasoning field name from generator signature
		existing_reasoning_trajectory = state.reasoning.get(self.reasoning_field_name, [])
		if evaluation_type == EvaluationType.PROCESS:
			# PRM requires at least one reasoning step to evaluate
			if not existing_reasoning_trajectory:
				available_keys = list(state.reasoning.keys())
				raise ValueError(
					f"PRM evaluation requires at least one reasoning step. "
					f"Looking for reasoning field '{self.reasoning_field_name}' in state.reasoning, "
					f"but available keys are: {available_keys}. "
					f"State input: {state.input}, State output: {state.output}"
				)
			evaluator_input[ToTField.REASONING_STEPS] = existing_reasoning_trajectory

		elif evaluation_type == EvaluationType.OUTCOME:
			# ORM: evaluate final solution quality
			evaluator_input.update(state.output)
			if self.consider_reasoning_in_final_eval:
				evaluator_input[ToTField.REASONING_STEPS] = existing_reasoning_trajectory

		return evaluator_input

	def _get_evaluator_for_type(self, evaluation_type: EvaluationType) -> LocalPredict:
		"""Get the appropriate evaluator for the given type."""
		if evaluation_type == EvaluationType.PROCESS:
			return self.process_evaluator
		elif evaluation_type == EvaluationType.OUTCOME:
			return self.outcome_evaluator
		else:
			raise ValueError(f"Unknown evaluation type: {evaluation_type}")


	def forward(
		self,
		states: State | list[State],
		n_samples_evaluator: int = 1,
		evaluator_temperature: float = 0.7,
		max_tokens: int = 2000,
		demos: list[dict[str, Any]] | None = None,
		demos_prm: list[dict[str, Any]] | None = None,
		demos_orm: list[dict[str, Any]] | None = None,
	) -> list[list[EvaluationResult]]:
		"""
		Forward method that automatically evaluates states using PRM/ORM based on completion.

		PRM (Process Reward Model): Used for states without outputs (evaluates reasoning quality)
		ORM (Outcome Reward Model): Used for states with outputs (evaluates solution quality)

		Parameters:
		    states (Union[State, List[State]]): Single state or list of states to evaluate.
		    n_samples_evaluator (int): Number of generations per state (supports future voting).
		    evaluator_temperature (float): Temperature for judge sampling.
		    max_tokens (int): Maximum tokens per evaluation.
		    demos (Optional[List[Dict[str, Any]]]): Default demo examples for both PRM and ORM evaluation.
		        If None, uses default demos based on signature type for each evaluation type.
		        To use no demos for both, pass an empty list [].
		    demos_prm (Optional[List[Dict[str, Any]]]): Demo examples specifically for PRM (process) evaluation.
		        If provided, overrides `demos` for PRM evaluation only. If None, PRM uses `demos` (or defaults).
		    demos_orm (Optional[List[Dict[str, Any]]]): Demo examples specifically for ORM (outcome) evaluation.
		        If provided, overrides `demos` for ORM evaluation only. If None, ORM uses `demos` (or defaults).

		Returns:
		    List[List[EvaluationResult]]: List of evaluation results for each state
		        - For single state: [results_for_state]
		        - For multiple states: [results_for_state1, results_for_state2, ...]
		        Each results_for_state contains EvaluationResult instances with scores and feedback.
		"""
		states = [states] if not isinstance(states, list) else states

		# Divide states into PRM and ORM groups based on completion status
		process_states_info = []  # (index, state) pairs for PRM evaluation
		outcome_states_info = []  # (index, state) pairs for ORM evaluation

		for i, state in enumerate(states):
			evaluation_type = (
				EvaluationType.OUTCOME if state.output else EvaluationType.PROCESS
			)
			if evaluation_type == EvaluationType.PROCESS:
				process_states_info.append((i, state))
			else:
				outcome_states_info.append((i, state))

		# Initialize results in original order
		final_results: list[list[EvaluationResult] | None] = [None] * len(states)

		config = {
			SamplingParam.N: n_samples_evaluator,
			SamplingParam.TEMPERATURE: evaluator_temperature,
			SamplingParam.MAX_TOKENS: max_tokens,
		}

		# Process PRM batch if any
		if process_states_info:
			process_states = [info[1] for info in process_states_info]

			# Convert states to evaluator inputs
			evaluator_inputs = [
				self._state_to_evaluator_input(state, EvaluationType.PROCESS)
				for state in process_states
			]

			# Create batched kwargs
			field_names = list(evaluator_inputs[0].keys())
			evaluator_inputs_batched_kwargs = {}
			for field_name in field_names:
				evaluator_inputs_batched_kwargs[field_name] = [
					input_dict[field_name] for input_dict in evaluator_inputs
				]

			# Run PRM evaluator with appropriate demos
			# Use demos_prm if provided, otherwise fall back to demos
			process_demos = demos_prm if demos_prm is not None else demos
			if process_demos is not None:
				logger.info(f"Using {len(process_demos)} user-provided demos for PRM evaluation")
			else:
				logger.info("Using zero-shot evaluation for PRM (no demos provided)")
			process_evaluator = self._get_evaluator_for_type(EvaluationType.PROCESS)
			process_predictions = process_evaluator(
				config=config, demos=process_demos, **evaluator_inputs_batched_kwargs
			)
			process_output_dicts = self._consolidate_scores(
				process_predictions, self.prm_rubric
			)

			# Group and assign results
			evaluations_per_state = (
				len(process_output_dicts) // len(process_states)
				if process_states
				else 0
			)

			for i, (original_idx, _) in enumerate(process_states_info):
				start_idx = i * evaluations_per_state
				end_idx = start_idx + evaluations_per_state
				state_evaluations = process_output_dicts[start_idx:end_idx]

				# Store results in original order
				final_results[original_idx] = state_evaluations

		# Process ORM batch if any
		if outcome_states_info:
			outcome_states = [info[1] for info in outcome_states_info]

			# Convert states to evaluator inputs
			outcome_batch_inputs = [
				self._state_to_evaluator_input(state, EvaluationType.OUTCOME)
				for state in outcome_states
			]

			# Create batched kwargs
			field_names = list(outcome_batch_inputs[0].keys())
			outcome_batched_kwargs = {}
			for field_name in field_names:
				outcome_batched_kwargs[field_name] = [
					input_dict[field_name] for input_dict in outcome_batch_inputs
				]

			# Run ORM evaluator with appropriate demos
			# Use demos_orm if provided, otherwise fall back to demos
			outcome_demos = demos_orm if demos_orm is not None else demos
			if outcome_demos is not None:
				logger.info(f"Using {len(outcome_demos)} user-provided demos for ORM evaluation")
			else:
				logger.info("Using zero-shot evaluation for ORM (no demos provided)")
			outcome_evaluator = self._get_evaluator_for_type(EvaluationType.OUTCOME)
			outcome_predictions = outcome_evaluator(
				config=config, demos=outcome_demos, **outcome_batched_kwargs
			)
			outcome_output_dicts = self._consolidate_scores(
				outcome_predictions, self.orm_rubric
			)

			# Group and assign results
			evaluations_per_state = (
				len(outcome_output_dicts) // len(outcome_states)
				if outcome_states
				else 0
			)

			for i, (original_idx, _) in enumerate(outcome_states_info):
				start_idx = i * evaluations_per_state
				end_idx = start_idx + evaluations_per_state
				state_evaluations = outcome_output_dicts[start_idx:end_idx]

				# Store results in original order
				final_results[original_idx] = state_evaluations

		return final_results # pyright: ignore[reportReturnType]
