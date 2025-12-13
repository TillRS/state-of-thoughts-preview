"""
Examples of reasoning-based signatures used across both unit tests and experiments.
These signatures are meant to express relalistic tasks that may benefit from reasoning.
"""
import enum

import dspy

from signatures.field import InputField, OutputField, ReasoningField
from signatures.field_constants import EvaluationMetric
from signatures.signature import ReasoningSignature
from tree.tree_constants import ReasoningState


class MathField(enum.StrEnum):
	"""Enum for math-related fields."""
	MATH_PROBLEM = "math_problem"
	MATH_OPERATION = "math_operation"
	ANSWER = "answer"


class ArgumentField(enum.StrEnum):
	"""Enum for argument-related fields."""
	TOPIC = "topic"
	STANCE = "stance"
	CLAIM = "claim"
	ARGUMENT = "argument"


class QuestionField(enum.StrEnum):
	"""Enum for question-answering fields."""
	QUESTION = "question"
	REASONING_STEP = "reasoning_step"
	ANSWER = "answer"


class TextAnalysisField(enum.StrEnum):
	"""Enum for text analysis fields."""
	INPUT_TEXT = "input_text"
	SUMMARY = "summary"
	SENTIMENT = "sentiment"
	KEYWORDS = "keywords"


class ArgumentStance(enum.StrEnum):
	"""Enumeration representing the stance to take on an argument topic."""
	PRO = "PRO"
	ANTI = "ANTI"

	def __str__(self) -> str:
		return self.value

class QuestionAnsweringWithReasoning(ReasoningSignature):
	"""Question answering signature with reasoning field."""

	question: str = InputField(
		desc="The question to answer",
		alias=QuestionField.QUESTION,
	)
	reasoning_step: str = ReasoningField(
		desc="Step-by-step reasoning",
		alias=QuestionField.REASONING_STEP,
	)
	answer: str = OutputField(
		desc="The answer to the question",
		alias=MathField.ANSWER,
	)


class SolveMathProblemWithReasoning(ReasoningSignature):
	"""
	Solve the provided math problem and return its answer.
	"""

	math_problem: str = InputField(
		desc="The math problem to solve",
		alias=MathField.MATH_PROBLEM,
	)
	math_operation: str = ReasoningField(
		desc="A math operation towards solving the math problem",
		alias=MathField.MATH_OPERATION,
	)
	answer: str = OutputField(
		desc="The answer to the math problem",
		alias=MathField.ANSWER,
	)

class AnalyzeTextWithReasoning(ReasoningSignature):
	"""
	Analyze the inputted text and perform the following tasks:
	- Summarize the text
	- Determine the sentiment of the text
	- Extract key words from the text
	"""

	input_text: str = InputField(
		desc="Input text to process",
		alias=TextAnalysisField.INPUT_TEXT,
	)
	reasoning_step: str = ReasoningField(
		desc="Step-by-step reasoning",
		alias=QuestionField.REASONING_STEP,
	)
	summary: str = OutputField(
		desc="A summary of the input",
		alias=TextAnalysisField.SUMMARY,
	)
	sentiment: str = OutputField(
		desc="The sentiment of the input",
		alias=TextAnalysisField.SENTIMENT,
	)
	keywords: list[str] = OutputField(
		desc="Key words from the input",
		alias=TextAnalysisField.KEYWORDS,
	)

class GenerateArgumentWithReasoning(ReasoningSignature):
	"""
	Generate an argument which takes the provided stance towards the provided topic.
	"""

	topic: str = InputField(
		desc="The topic to generate an argument about",
		alias=ArgumentField.TOPIC,
	)
	stance: ArgumentStance = InputField(
		desc="The stance to take on the topic",
		alias=ArgumentField.STANCE,
	)
	claim: str = ReasoningField(
		desc="A component of the argument that advocates for the given stance towards the topic",
		alias=ArgumentField.CLAIM,
	)
	argument: str = OutputField(
		desc="The generated argument",
		alias=ArgumentField.ARGUMENT,
	)


# Evaluator Signatures for Argument Generation
# Note: Evaluator signatures include:
# - constants.INPUT fields: the generator's input fields + output fields (what we're evaluating)
# - constants.OUTPUT fields: at least one numeric scoring field + exactly one feedback field


class ArgumentEvaluatorMultiDimensional(dspy.Signature):
	"""Evaluate the generated argument along three dimensions:

	PERSUASIVENESS (1-7): How convincing and compelling the argument is
	- 7 = Highly persuasive with strong evidence and reasoning
	- 5-6 = Moderately persuasive with reasonable support
	- 3-4 = Somewhat persuasive but lacks strong support
	- 1-2 = Weak or unconvincing argument

	COHERENCE (1-7): How well-structured and logically organized the argument is
	- 7 = Perfectly coherent with clear logical flow
	- 5-6 = Generally coherent with minor organizational issues
	- 3-4 = Somewhat coherent but with notable structural problems
	- 1-2 = Poorly organized or incoherent

	RELEVANCE (1-7): How well the argument addresses the topic and stance
	- 7 = Perfectly aligned with topic and stance
	- 5-6 = Generally relevant with minor deviations
	- 3-4 = Partially relevant but misses key aspects
	- 1-2 = Off-topic or misaligned with stance

	Provide separate numeric scores for each dimension.
	"""

	# Input fields (from generator)
	topic: str = dspy.InputField(
		desc="The topic the argument is about",
		alias=ArgumentField.TOPIC,
	)
	stance: ArgumentStance = dspy.InputField(
		desc="The stance taken on the topic",
		alias=ArgumentField.STANCE,
	)
	argument: str = dspy.InputField(
		desc="The generated argument to evaluate",
		alias=ArgumentField.ARGUMENT,
	)

	# Output fields (evaluation scores)
	persuasiveness: int = OutputField(
		desc="Persuasiveness score", ge=1, le=7, alias=EvaluationMetric.PERSUASIVENESS,
	)
	coherence: int = OutputField(
		desc="Coherence score", ge=1, le=7, alias=EvaluationMetric.COHERENCE,
	)
	relevance: int = OutputField(
		desc="Relevance score", ge=1, le=7, alias=EvaluationMetric.RELEVANCE,
	)
	feedback: str = OutputField(
		desc="Detailed feedback explaining the assigned scores",
		alias=ReasoningState.FEEDBACK,
	)


class ArgumentEvaluatorSingleScore(dspy.Signature):
	"""Evaluate the overall quality of the generated argument:

	10 = Exceptional: Highly persuasive, perfectly coherent, and fully addresses the topic/stance
	8-9 = Strong: Very good argument with only minor weaknesses
	6-7 = Good: Solid argument but with some notable issues
	4-5 = Adequate: Acceptable but with significant room for improvement
	2-3 = Weak: Poor quality with major flaws in persuasiveness, coherence, or relevance
	1 = Very Poor: Fails to meet basic standards for an argument

	Provide a single overall quality score.
	"""

	# Input fields (from generator)
	topic: str = dspy.InputField(desc="The topic the argument is about")
	stance: ArgumentStance = dspy.InputField(desc="The stance taken on the topic")
	argument: str = dspy.InputField(desc="The generated argument to evaluate")

	# Output fields (evaluation score)
	overall_quality: int =OutputField(desc="Overall argument quality score", ge=1, le=10)
	feedback: str =OutputField(desc="Detailed feedback explaining the assigned score")


# Weighted Evaluator Rubric Examples
# These demonstrate the rubric_weight parameter for custom dimension weighting


class WeightedPRMRubric(dspy.Signature):
	"""Weighted PRM rubric emphasizing soundness over promise.

	Evaluates reasoning steps using two dimensions with custom weights:
	- soundness (70%): Backward-looking correctness of the reasoning step
	- promise (30%): Forward-looking trajectory quality

	This weighting prioritizes logical correctness over solution potential.
	"""

	soundness: int = OutputField(
		desc="Soundness score (backward-looking correctness) on a 1 to 7 Likert scale",
		ge=1,
		le=7,
		rubric_weight=0.7,
	)
	promise: int = OutputField(
		desc="Promise score (forward-looking trajectory quality) on a 1 to 7 Likert scale",
		ge=1,
		le=7,
		rubric_weight=0.3,
	)
	feedback: str = OutputField(
		desc="Detailed feedback explaining the assigned scores", annotation=str
	)


class WeightedMultiDimensionRubric(dspy.Signature):
	"""Multi-dimension rubric with custom weights for each dimension.

	Evaluates reasoning or solutions across three dimensions:
	- correctness (50%): Accuracy of facts, formulas, and calculations
	- clarity (30%): How well-explained and understandable the reasoning is
	- efficiency (20%): Directness and conciseness of the approach

	This weighting emphasizes correctness most, then clarity, then efficiency.
	"""

	correctness: float = OutputField(
		desc="Correctness score 1-5: accuracy of facts, formulas, and calculations",
		ge=1.0,
		le=5.0,
		rubric_weight=0.5,
	)
	clarity: float = OutputField(
		desc="Clarity score 1-5: how well-explained and understandable the reasoning is",
		ge=1.0,
		le=5.0,
		rubric_weight=0.3,
	)
	efficiency: float = OutputField(
		desc="Efficiency score 1-5: directness and conciseness of the approach",
		ge=1.0,
		le=5.0,
		rubric_weight=0.2,
	)
	feedback: str = OutputField(
		desc="Detailed feedback covering all three dimensions (correctness, clarity, efficiency)"
	)


class BalancedArgumentRubric(dspy.Signature):
	"""Balanced argument evaluation rubric with equal weights.

	Evaluates arguments across three equally-weighted dimensions:
	- persuasiveness (33.3%): How convincing the argument is
	- coherence (33.3%): Logical structure and organization
	- relevance (33.3%): Alignment with topic and stance

	Note: When no rubric_weight is specified, dimensions are weighted equally.
	"""

	persuasiveness: int = OutputField(
		desc="Persuasiveness score", ge=1, le=7
		# No rubric_weight specified = equal weights (33.3% each)
	)
	coherence: int = OutputField(
		desc="Coherence score", ge=1, le=7
		# No rubric_weight specified = equal weights
	)
	relevance: int = OutputField(
		desc="Relevance score", ge=1, le=7
		# No rubric_weight specified = equal weights
	)
	feedback: str = OutputField(desc="Detailed feedback explaining the assigned scores")
