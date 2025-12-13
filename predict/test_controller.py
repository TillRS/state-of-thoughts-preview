"""
Tests for the controller module.

Expected usage:
```bash
pytest predict/test_controller.py -vv
```
"""

# Standard library imports
import json
import logging
import os
from typing import Any, Literal

# Third-party imports
import dspy
import pytest
import torch

# Local imports
from constants import CandidateGenerationMethod, OpenSourceModel, Verbosity
from lm.generative_local_lm import GenerativeLocalVLLM
from predict.controller import TreeOfThoughtsController
from predict.controller_constants import (
	DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
	ActionSpaceJsonKeys,
	ControllerActionParameters,
	ControllerContinueReasoningChoice,
	ControllerOutput,
	ControllerOutputParameters,
)
from predict.controller_utils import (
	DEFAULT_TOOL,
	FINISH_TOOL,
	ControllerPrediction,
	ReasoningIntervention,
	remove_duplicate_actions_with_counts,
	return_action_if_single_option,
)
from predict.demos.controller_demos import (
	ARGUMENT_CONTINUE_FINISH_DEMOS,
	CONTROLLER_DEMOS,
	STRUCTURE_CONTROLLER_DEMOS,
	STYLE_CONTROLLER_DEMOS,
	STYLE_STRUCTURE_CONTROLLER_DEMOS,
)
from signatures import (
	ArgumentStance,
	GenerateArgumentWithReasoning,
	QuestionAnsweringWithReasoning,
	ReasoningSignature,
	SolveMathProblemWithReasoning,
)
from signatures.example_signatures import (
	ArgumentField,
	MathField,
	QuestionField,
)
from tree import State
from tree.tree_constants import ReasoningState
from utilities_for_tests import (
	MockGenerativeLocalVLLM,
	MockPredict,
)

# Set up logging
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# =============================================================================
# Test Fixtures for Action Space JSONs
# =============================================================================


@pytest.fixture
def temp_action_space_styles(tmp_path):
	"""Create a temporary JSON file for causal styles with 2 options."""
	styles_json = {
		ActionSpaceJsonKeys.DIMENSION_NAME: "style",
		ActionSpaceJsonKeys.DIMENSION_DEFINITION: (
			"Forces the next reasoning step to adopt a specific rhetorical style or "
			"expressive technique, controlling how arguments are articulated and presented. "
			"Interventions along this dimension ensure the next step uses a particular mode "
			"of expression (e.g., figurative language, statistical evidence, narrative "
			"storytelling, formal tone, or direct audience engagement)."
		),
		ActionSpaceJsonKeys.DIMENSION_CHOICES: {
			"Figurative Language": {
				ActionSpaceJsonKeys.DIMENSION_DEFINITION: "Use metaphor, simile, analogy, or symbolism to make ideas concrete.",
				ActionSpaceJsonKeys.CHOICE_INTERNAL_REASONING: (
					"I should employ non-literal comparison to make abstract concepts vivid. "
				),
			},
			"Statistical & Data-Driven": {
				ActionSpaceJsonKeys.DIMENSION_DEFINITION: "Present numerical data, statistics, or quantified evidence.",
				ActionSpaceJsonKeys.CHOICE_INTERNAL_REASONING: (
					"I should use numbers and data to provide concrete, measurable support. "
				),
			},
		},
	}
	styles_path = tmp_path / "style.json"
	with open(styles_path, "w") as f:
		json.dump(styles_json, f, indent="\t")
	return str(styles_path)


@pytest.fixture
def temp_action_space_structures(tmp_path):
	"""Create a temporary JSON file for causal structures with 2 options."""
	structures_json = {
		ActionSpaceJsonKeys.DIMENSION_NAME: "structure",
		ActionSpaceJsonKeys.DIMENSION_DEFINITION: (
			"Forces the next reasoning step to adhere to a specific discourse structure, "
			"controlling how ideas connect and relate to each other. Interventions along "
			"this dimension ensure the next step follows a particular organizational pattern "
			"(e.g., presenting a counterpoint, providing evidence, drawing a causal inference, "
			"or offering an example)."
		),
		ActionSpaceJsonKeys.DIMENSION_CHOICES: {
			"Causal Reasoning": {
				ActionSpaceJsonKeys.DIMENSION_DEFINITION: "State causes, effects, consequences, or logical implications.",
				ActionSpaceJsonKeys.CHOICE_INTERNAL_REASONING: "I should use causal reasoning.",
				ActionSpaceJsonKeys.CHOICE_PREFIX: "Therefore",
			},
			"Evidence & Support": {
				ActionSpaceJsonKeys.DIMENSION_DEFINITION: "Cite facts, studies, expert testimony, or documented sources.",
				ActionSpaceJsonKeys.CHOICE_PREFIX: "According to",
			},
			"Contrast": {
				ActionSpaceJsonKeys.DIMENSION_DEFINITION: "Present contrasting viewpoints, counterarguments, or exceptions.",
				ActionSpaceJsonKeys.CHOICE_PREFIX: "However",
			},
			"Chronological Sequence": {
				ActionSpaceJsonKeys.DIMENSION_DEFINITION: "Order events or points in time.",
				ActionSpaceJsonKeys.CHOICE_PREFIX: "First",
			},
		},
	}
	structures_path = tmp_path / "structure.json"
	with open(structures_path, "w") as f:
		json.dump(structures_json, f, indent="\t")
	return str(structures_path)


@pytest.fixture
def temp_action_space_subtopics(tmp_path):
	"""Create a temporary JSON file for causal subtopics with 2 options."""
	subtopics_json = {
		ActionSpaceJsonKeys.DIMENSION_NAME: "stock_issue",
		ActionSpaceJsonKeys.DIMENSION_DEFINITION: (
			"Forces the next reasoning step to address a specific thematic angle or "
			"argumentative dimension. Interventions along this dimension ensure the next "
			"step focuses on a particular aspect of the debate (e.g., economic impact, "
			"social justice, health and safety, environmental concerns, personal freedom, "
			"practical feasibility, moral principles, or legal issues)."
		),
		ActionSpaceJsonKeys.DIMENSION_CHOICES: {
			"General Introduction": {
				ActionSpaceJsonKeys.DIMENSION_DEFINITION: "Provide a broad overview, context, or introductory statements without specific thematic focus.",
				ActionSpaceJsonKeys.CHOICE_INTERNAL_REASONING: (
					"I should provide a general introduction to the topic. "
				),
			},
			"Economic Impact": {
				ActionSpaceJsonKeys.DIMENSION_DEFINITION: "Discuss financial costs, benefits, market effects, or resource allocation.",
				ActionSpaceJsonKeys.CHOICE_INTERNAL_REASONING: (
					"I should focus on the economic and financial implications of this issue. "
				),
			},
			"Social Justice & Equity": {
				ActionSpaceJsonKeys.DIMENSION_DEFINITION: "Address fairness, equality, discrimination, or marginalized groups.",
				ActionSpaceJsonKeys.CHOICE_INTERNAL_REASONING: (
					"I should consider the fairness and equity dimensions of this issue. "
				),
			},
		},
	}
	subtopics_path = tmp_path / "stock_issue.json"
	with open(subtopics_path, "w") as f:
		json.dump(subtopics_json, f, indent="\t")
	return str(subtopics_path)


# =============================================================================
# Helper Functions for Creating Expected Tools
# =============================================================================


def create_expected_tool_for_styles() -> dspy.Tool:
	"""Create expected tool for causal styles dimension only."""

	def tool_func(
		style: Literal["Figurative Language", "Statistical & Data-Driven"],
	) -> ReasoningIntervention:
		# Use the actual internal_reasoning from the fixture
		internal_reasoning_map = {
			"Figurative Language": "I should employ non-literal comparison to make abstract concepts vivid. ",
			"Statistical & Data-Driven": "I should use numbers and data to provide concrete, measurable support. ",
		}
		return ReasoningIntervention(
			continue_reasoning=True,
			internal_reasoning=internal_reasoning_map[style],
			prefix="",
		)

	return dspy.Tool(
		name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
		func=tool_func,
		desc="""
Determine the best choice in each dimension (corresponding to an argument of this function) to improve the quality of the next reasoning step.
You **must** select **one** choice for **each** of the provided dimensions.
Be mindful of the impact that each choice has on the next reasoning step.
When providing a choice, avoid enumeration or unnecessary prefixes/suffixes -- just provide your choice directly.
""".strip(),
		args={
			"style": """
Forces the next reasoning step to adopt a specific rhetorical style or expressive technique, controlling how arguments are articulated and presented. Interventions along this dimension ensure the next step uses a particular mode of expression (e.g., figurative language, statistical evidence, narrative storytelling, formal tone, or direct audience engagement).
Options:
- "Figurative Language": Use metaphor, simile, analogy, or symbolism to make ideas concrete.
- "Statistical & Data-Driven": Present numerical data, statistics, or quantified evidence.
""".strip()
		},
		arg_types={"style": Literal["Figurative Language", "Statistical & Data-Driven"]},
	)


def create_expected_tool_for_structures() -> dspy.Tool:
	"""Create expected tool for causal structures dimension only."""

	def tool_func(
		structure: Literal["Causal Reasoning", "Evidence & Support", "Contrast", "Chronological Sequence"],
	) -> ReasoningIntervention:
		# Use the actual prefix from the fixture
		prefix_map = {
			"Causal Reasoning": "Therefore",
			"Evidence & Support": "According to",
			"Contrast": "However",
			"Chronological Sequence": "First",
		}
		return ReasoningIntervention(
			continue_reasoning=True,
			internal_reasoning="",
			prefix=prefix_map[structure],
		)

	return dspy.Tool(
		name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
		func=tool_func,
		desc="""
Determine the best choice in each dimension (corresponding to an argument of this function) to improve the quality of the next reasoning step.
You **must** select **one** choice for **each** of the provided dimensions.
Be mindful of the impact that each choice has on the next reasoning step.
When providing a choice, avoid enumeration or unnecessary prefixes/suffixes -- just provide your choice directly.
""".strip(),
		args={
			"structure": """
Forces the next reasoning step to adhere to a specific discourse structure, controlling how ideas connect and relate to each other. Interventions along this dimension ensure the next step follows a particular organizational pattern (e.g., presenting a counterpoint, providing evidence, drawing a causal inference, or offering an example).
Options:
- "Causal Reasoning": State causes, effects, consequences, or logical implications.
- "Evidence & Support": Cite facts, studies, expert testimony, or documented sources.
- "Contrast": Present contrasting viewpoints, counterarguments, or exceptions.
- "Chronological Sequence": Order events or points in time.
""".strip()
		},
		arg_types={"structure": Literal["Causal Reasoning", "Evidence & Support", "Contrast", "Chronological Sequence"]},
	)


def create_expected_tool_for_styles_and_structures() -> dspy.Tool:
	"""Create expected tool for causal styles and structures dimensions combined."""

	def tool_func(
		style: Literal["Figurative Language", "Statistical & Data-Driven"],
		structure: Literal["Causal Reasoning", "Evidence & Support", "Contrast", "Chronological Sequence"],
	) -> ReasoningIntervention:
		# Use the actual internal_reasoning and prefix from the fixtures
		internal_reasoning_map = {
			"Figurative Language": "I should employ non-literal comparison to make abstract concepts vivid. ",
			"Statistical & Data-Driven": "I should use numbers and data to provide concrete, measurable support. ",
		}
		prefix_map = {
			"Causal Reasoning": "Therefore",
			"Evidence & Support": "According to",
			"Contrast": "However",
			"Chronological Sequence": "First",
		}
		return ReasoningIntervention(
			continue_reasoning=True,
			internal_reasoning=internal_reasoning_map[style],
			prefix=prefix_map[structure],
		)

	return dspy.Tool(
		name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
		func=tool_func,
		desc="""
Determine the best choice in each dimension (corresponding to an argument of this function) to improve the quality of the next reasoning step.
You **must** select **one** choice for **each** of the provided dimensions.
Be mindful of the impact that each choice has on the next reasoning step.
When providing a choice, avoid enumeration or unnecessary prefixes/suffixes -- just provide your choice directly.
""".strip(),
		args={
			"style": """
Forces the next reasoning step to adopt a specific rhetorical style or expressive technique, controlling how arguments are articulated and presented. Interventions along this dimension ensure the next step uses a particular mode of expression (e.g., figurative language, statistical evidence, narrative storytelling, formal tone, or direct audience engagement).
Options:
- "Figurative Language": Use metaphor, simile, analogy, or symbolism to make ideas concrete.
- "Statistical & Data-Driven": Present numerical data, statistics, or quantified evidence.
""".strip(),
			"structure": """
Forces the next reasoning step to adhere to a specific discourse structure, controlling how ideas connect and relate to each other. Interventions along this dimension ensure the next step follows a particular organizational pattern (e.g., presenting a counterpoint, providing evidence, drawing a causal inference, or offering an example).
Options:
- "Causal Reasoning": State causes, effects, consequences, or logical implications.
- "Evidence & Support": Cite facts, studies, expert testimony, or documented sources.
- "Contrast": Present contrasting viewpoints, counterarguments, or exceptions.
- "Chronological Sequence": Order events or points in time.
""".strip(),
		},
		arg_types={
			"style": Literal["Figurative Language", "Statistical & Data-Driven"],
			"structure": Literal["Causal Reasoning", "Evidence & Support", "Contrast", "Chronological Sequence"],
		},
	)


def create_expected_tool_for_subtopics_styles_and_structures() -> dspy.Tool:
	"""Create expected tool for all three dimensions: subtopics, styles, and structures."""

	def tool_func(
		stock_issue: Literal["General Introduction", "Economic Impact", "Social Justice & Equity"],
		style: Literal["Figurative Language", "Statistical & Data-Driven"],
		structure: Literal["Causal Reasoning", "Evidence & Support", "Contrast", "Chronological Sequence"],
	) -> ReasoningIntervention:
		# Use the actual internal_reasoning and prefix from the fixtures
		# Combine internal_reasoning from both stock_issue and style
		stock_issue_internal_reasoning_map = {
			"General Introduction": "I should provide a general introduction to the topic. ",
			"Economic Impact": "I should focus on the economic and financial implications of this issue. ",
			"Social Justice & Equity": "I should consider the fairness and equity dimensions of this issue. ",
		}
		style_internal_reasoning_map = {
			"Figurative Language": "I should employ non-literal comparison to make abstract concepts vivid. ",
			"Statistical & Data-Driven": "I should use numbers and data to provide concrete, measurable support. ",
		}
		prefix_map = {
			"Causal Reasoning": "Therefore",
			"Evidence & Support": "According to",
			"Contrast": "However",
			"Chronological Sequence": "First",
		}
		combined_internal_reasoning = (
			stock_issue_internal_reasoning_map[stock_issue] + style_internal_reasoning_map[style]
		)
		return ReasoningIntervention(
			continue_reasoning=True,
			internal_reasoning=combined_internal_reasoning,
			prefix=prefix_map[structure],
		)

	return dspy.Tool(
		name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
		func=tool_func,
		desc="""
Determine the best choice in each dimension (corresponding to an argument of this function) to improve the quality of the next reasoning step.
You **must** select **one** choice for **each** of the provided dimensions.
Be mindful of the impact that each choice has on the next reasoning step.
When providing a choice, avoid enumeration or unnecessary prefixes/suffixes -- just provide your choice directly.
""".strip(),
		args={
			"stock_issue": """
Forces the next reasoning step to address a specific thematic angle or argumentative dimension. Interventions along this dimension ensure the next step focuses on a particular aspect of the debate (e.g., economic impact, social justice, health and safety, environmental concerns, personal freedom, practical feasibility, moral principles, or legal issues).
Options:
- "General Introduction": Provide a broad overview, context, or introductory statements without specific thematic focus.
- "Economic Impact": Discuss financial costs, benefits, market effects, or resource allocation.
- "Social Justice & Equity": Address fairness, equality, discrimination, or marginalized groups.
""".strip(),
			"style": """
Forces the next reasoning step to adopt a specific rhetorical style or expressive technique, controlling how arguments are articulated and presented. Interventions along this dimension ensure the next step uses a particular mode of expression (e.g., figurative language, statistical evidence, narrative storytelling, formal tone, or direct audience engagement).
Options:
- "Figurative Language": Use metaphor, simile, analogy, or symbolism to make ideas concrete.
- "Statistical & Data-Driven": Present numerical data, statistics, or quantified evidence.
""".strip(),
			"structure": """
Forces the next reasoning step to adhere to a specific discourse structure, controlling how ideas connect and relate to each other. Interventions along this dimension ensure the next step follows a particular organizational pattern (e.g., presenting a counterpoint, providing evidence, drawing a causal inference, or offering an example).
Options:
- "Causal Reasoning": State causes, effects, consequences, or logical implications.
- "Evidence & Support": Cite facts, studies, expert testimony, or documented sources.
- "Contrast": Present contrasting viewpoints, counterarguments, or exceptions.
- "Chronological Sequence": Order events or points in time.
""".strip(),
		},
		arg_types={
			"stock_issue": Literal["General Introduction", "Economic Impact", "Social Justice & Equity"],
			"style": Literal["Figurative Language", "Statistical & Data-Driven"],
			"structure": Literal["Causal Reasoning", "Evidence & Support", "Contrast", "Chronological Sequence"],
		},
	)


# =============================================================================
# GPU Skip Markers
# =============================================================================

pytestmark_gpu = pytest.mark.skipif(
	not torch.cuda.is_available(),
	reason="GPU tests require GPU access",
)


# =============================================================================
# Test State Data
# =============================================================================



BATCH_STATES_INPUTS = [
	{QuestionField.QUESTION: "What is 2+2?"},
	{QuestionField.QUESTION: "What is 3*5?"},
	{QuestionField.QUESTION: "What is the capital of France?"},
]

BATCH_STATES_REASONING = [
	{ReasoningState.REASONING: ["Addition problem"]},
	{ReasoningState.REASONING: ["Multiplication problem"]},
	{ReasoningState.REASONING: ["Geography question"]},
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def configure_mock_lm():
	"""Automatically configure MockGenerativeLocalVLLM for all non-GPU tests."""
	mock_lm = MockGenerativeLocalVLLM()
	dspy.settings.configure(lm=mock_lm)
	yield


@pytest.fixture
def simple_state():
	"""Create a simple test state for controller testing."""
	return State(
		input={QuestionField.QUESTION: "What is 2+2?"},
		reasoning={QuestionField.REASONING_STEP: ["First, I need to think about addition."]},
		controller_outputs=[],
		feedback=[],
		output={},
	)


@pytest.fixture
def batch_states():
	"""Create multiple test states for batch testing."""
	states = []
	for i in range(len(BATCH_STATES_INPUTS)):
		state = State(
			input=BATCH_STATES_INPUTS[i],
			reasoning=BATCH_STATES_REASONING[i],
			controller_outputs=[],
			feedback=[],
			output={},
		)
		states.append(state)
	return states


# =============================================================================
# Test: Controller Initialization (__init__)
# =============================================================================


@pytest.mark.parametrize(
	# Parameter names
	[
		"signature",
		"max_reasoning_steps",
		"tools",
		"action_space_paths",
		"early_stopping_enabled",
		"expected_reasoning_field",
		"expected_tools",
		"expected_decide_input_fields",
		"expected_decide_output_fields",
		"expected_exception",
	],
	[
		pytest.param(
			QuestionAnsweringWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			None,								# action_space_paths
			True,								# early_stopping_enabled
			QuestionField.REASONING_STEP,		# expected_reasoning_field
			{									# expected_tools
				ControllerContinueReasoningChoice.CONTINUE_REASONING: DEFAULT_TOOL,
				ControllerContinueReasoningChoice.FINISH: FINISH_TOOL,
			},
			[									# expected_decide_input_fields
				QuestionField.QUESTION,
				QuestionField.REASONING_STEP,
				"number_of_additional_reasoning_steps",
			],
			[									# expected_decide_output_fields
				ControllerActionParameters.CONSIDERATIONS,
				ControllerActionParameters.ACTION,
			],
			None,								# expected_exception
			id="qa_signature_default_tools_early_stopping",
		),
		pytest.param(
			QuestionAnsweringWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			None,								# action_space_paths
			False,								# early_stopping_enabled
			QuestionField.REASONING_STEP,		# expected_reasoning_field
			{ControllerContinueReasoningChoice.CONTINUE_REASONING: DEFAULT_TOOL},
			[									# expected_decide_input_fields
				QuestionField.QUESTION,
				QuestionField.REASONING_STEP,
				"number_of_additional_reasoning_steps",
			],
			[									# expected_decide_output_fields
				# With only DEFAULT_TOOL (no arguments), ARGUMENTS is not included
				ControllerActionParameters.CONSIDERATIONS,
				ControllerActionParameters.ACTION,
			],
			None,								# expected_exception
			id="qa_signature_default_tools_no_early_stopping",
		),
		pytest.param(
			SolveMathProblemWithReasoning,		# signature
			10,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			None,								# action_space_paths
			True,								# early_stopping_enabled
			MathField.MATH_OPERATION,			# expected_reasoning_field
			{									# expected_tools
				ControllerContinueReasoningChoice.CONTINUE_REASONING: DEFAULT_TOOL,
				ControllerContinueReasoningChoice.FINISH: FINISH_TOOL,
			},
			[									# expected_decide_input_fields
				MathField.MATH_PROBLEM,
				MathField.MATH_OPERATION,
				"number_of_additional_reasoning_steps",
			],
			[									# expected_decide_output_fields
				ControllerActionParameters.CONSIDERATIONS,
				ControllerActionParameters.ACTION,
			],
			None,								# expected_exception
			id="math_signature_default_tools",
		),
		pytest.param(
			GenerateArgumentWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			"styles",							# action_space_paths
			True,								# early_stopping_enabled
			ArgumentField.CLAIM,				# expected_reasoning_field
			{									# expected_tools
				"intervene_on_next_reasoning_step": create_expected_tool_for_styles(),
				ControllerContinueReasoningChoice.FINISH: FINISH_TOOL,
			},
			[									# expected_decide_input_fields
				ArgumentField.TOPIC,
				ArgumentField.STANCE,
				ArgumentField.CLAIM,
				"number_of_additional_reasoning_steps",
			],
			[									# expected_decide_output_fields
				ControllerActionParameters.CONSIDERATIONS,
				ControllerActionParameters.ACTION,
				ControllerActionParameters.ARGUMENTS,
			],
			None,								# expected_exception
			id="generate_argument_style_intervention",
		),
		pytest.param(
			GenerateArgumentWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			"structures",						# action_space_paths
			True,								# early_stopping_enabled
			ArgumentField.CLAIM,				# expected_reasoning_field
			{									# expected_tools
				"intervene_on_next_reasoning_step": create_expected_tool_for_structures(),
				ControllerContinueReasoningChoice.FINISH: FINISH_TOOL,
			},
			[									# expected_decide_input_fields
				ArgumentField.TOPIC,
				ArgumentField.STANCE,
				ArgumentField.CLAIM,
				"number_of_additional_reasoning_steps",
			],
			[									# expected_decide_output_fields
				ControllerActionParameters.CONSIDERATIONS,
				ControllerActionParameters.ACTION,
				ControllerActionParameters.ARGUMENTS,
			],
			None,								# expected_exception
			id="generate_argument_structure_intervention",
		),
		pytest.param(
			GenerateArgumentWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			"styles_structures",				# action_space_paths
			True,								# early_stopping_enabled
			ArgumentField.CLAIM,				# expected_reasoning_field
			{									# expected_tools
				"intervene_on_next_reasoning_step": create_expected_tool_for_styles_and_structures(),
				ControllerContinueReasoningChoice.FINISH: FINISH_TOOL,
			},
			[									# expected_decide_input_fields
				ArgumentField.TOPIC,
				ArgumentField.STANCE,
				ArgumentField.CLAIM,
				"number_of_additional_reasoning_steps",
			],
			[									# expected_decide_output_fields
				ControllerActionParameters.CONSIDERATIONS,
				ControllerActionParameters.ACTION,
				ControllerActionParameters.ARGUMENTS,
			],
			None,								# expected_exception
			id="generate_argument_style_structure_intervention",
		),
		pytest.param(
			QuestionAnsweringWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			"subtopics_styles_structures",		# action_space_paths
			True,								# early_stopping_enabled
			QuestionField.REASONING_STEP,		# expected_reasoning_field
			{									# expected_tools
				"intervene_on_next_reasoning_step": create_expected_tool_for_subtopics_styles_and_structures(),
				ControllerContinueReasoningChoice.FINISH: FINISH_TOOL,
			},
			[									# expected_decide_input_fields
				QuestionField.QUESTION,
				QuestionField.REASONING_STEP,
				"number_of_additional_reasoning_steps",
			],
			[									# expected_decide_output_fields
				ControllerActionParameters.CONSIDERATIONS,
				ControllerActionParameters.ACTION,
				ControllerActionParameters.ARGUMENTS,
			],
			None,								# expected_exception
			id="qa_signature_subtopic_style_structure_intervention",
		),
		pytest.param(
			QuestionAnsweringWithReasoning,		# signature
			3,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			None,								# action_space_paths
			True,								# early_stopping_enabled
			QuestionField.REASONING_STEP,		# expected_reasoning_field
			{									# expected_tools
				ControllerContinueReasoningChoice.CONTINUE_REASONING: DEFAULT_TOOL,
				ControllerContinueReasoningChoice.FINISH: FINISH_TOOL,
			},
			[									# expected_decide_input_fields
				QuestionField.QUESTION,
				QuestionField.REASONING_STEP,
				"number_of_additional_reasoning_steps",
			],
			[									# expected_decide_output_fields
				ControllerActionParameters.CONSIDERATIONS,
				ControllerActionParameters.ACTION,
			],
			None,								# expected_exception
			id="no_tools_no_action_space_defaults_to_default_tool",
		),
	],
)
def test_controller_initialization(
	signature: type[ReasoningSignature],
	max_reasoning_steps: int,
	tools: list[dspy.Tool] | None,
	action_space_paths: str | None,
	early_stopping_enabled: bool,
	expected_reasoning_field: str,
	expected_tools: dict[str, dspy.Tool] | None,
	expected_decide_input_fields: list[str],
	expected_decide_output_fields: list[str],
	expected_exception: type[Exception] | None,
	temp_action_space_styles,
	temp_action_space_structures,
	temp_action_space_subtopics,
) -> None:
	"""
	Test TreeOfThoughtsController initialization with various configurations.

	This test verifies that:
	1. The controller properly initializes with different reasoning signatures
	2. The exact expected tools are created, including descriptions, args, and arg_types
	3. The input and output fields of decide_next_step_single are validated

	Parameters:
	    signature: The reasoning signature class to test
	    max_reasoning_steps: Maximum number of reasoning steps allowed
	    tools: Tools configuration (None uses defaults)
		action_space_paths: Fixture key for action space paths or None
	    expected_reasoning_field: Expected name of the primary reasoning field
		expected_tools: Dictionary mapping tool names to expected Tool objects
		expected_decide_input_fields: Expected input fields for decide_next_step_single
		expected_decide_output_fields: Expected output fields for decide_next_step_single
	    early_stopping_enabled: Whether to enable early stopping
	    expected_exception: Expected exception type or None if no exception
	"""
	# Convert fixture key to actual paths
	if action_space_paths == "styles":
		actual_paths = [temp_action_space_styles]
	elif action_space_paths == "structures":
		actual_paths = [temp_action_space_structures]
	elif action_space_paths == "styles_structures":
		actual_paths = [temp_action_space_styles, temp_action_space_structures]
	elif action_space_paths == "subtopics_styles_structures":
		actual_paths = [
			temp_action_space_subtopics,
			temp_action_space_styles,
			temp_action_space_structures,
		]
	else:
		actual_paths = None

	if expected_exception is not None:
		with pytest.raises(expected_exception):
			TreeOfThoughtsController(
				signature=signature,
				max_reasoning_steps=max_reasoning_steps,
				tools=tools,
				action_space_paths=actual_paths,
				early_stopping_enabled=early_stopping_enabled,
			)
		return

	# We expect initialization to succeed
	controller = TreeOfThoughtsController(
		signature=signature,
		max_reasoning_steps=max_reasoning_steps,
		tools=tools,
		action_space_paths=actual_paths,
		early_stopping_enabled=early_stopping_enabled,
	)

	# Verify basic initialization
	assert controller.max_reasoning_steps == max_reasoning_steps
	assert controller.reasoning_field_name == expected_reasoning_field
	assert controller.input_field_names == list(signature.input_fields.keys())
	assert controller.output_field_names == list(signature.output_fields.keys())

	# Verify decide_next_step_single module signature
	decide_signature = controller.decide_next_step_single.signature

	assert list(decide_signature.input_fields.keys()) == expected_decide_input_fields, (
		f"Input fields mismatch.\nExpected: {expected_decide_input_fields}\n"
		f"Actual: {list(decide_signature.input_fields.keys())}"
	)
	assert list(decide_signature.output_fields.keys()) == expected_decide_output_fields, (
		f"Output fields mismatch.\nExpected: {expected_decide_output_fields}\n"
		f"Actual: {list(decide_signature.output_fields.keys())}"
	)

	# Verify exact tool names match expected
	assert set(controller.tools.keys()) == set(expected_tools.keys()), (
		f"Tool names mismatch.\nExpected: {set(expected_tools.keys())}\n"
		f"Actual: {set(controller.tools.keys())}"
	)

	# Verify each tool matches expected specifications
	for tool_name, expected_tool in expected_tools.items():
		actual_tool = controller.tools[tool_name]
		assert isinstance(actual_tool, dspy.Tool)
		assert actual_tool.name == expected_tool.name
		assert actual_tool.desc == expected_tool.desc
		assert actual_tool.args == expected_tool.args
		assert actual_tool.arg_types == expected_tool.arg_types


# =============================================================================
# Helper Functions for Creating Mock Responses and Expected Predictions
# =============================================================================


def create_dummy_controller_output(
	action: str = DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
	action_arguments: dict[str, Any] | None = None,
	tool_descriptions: str = "Dummy tool description",
	considerations: str = "Dummy considerations",
	continue_reasoning: bool = True,
	internal_reasoning: str = "Dummy internal reasoning",
	prefix: str = "Dummy prefix",
) -> ControllerOutput:
	"""Create a dummy ControllerOutput for testing."""
	return ControllerOutput(
		action=action,
		action_arguments=action_arguments or {},
		tool_descriptions=tool_descriptions,
		considerations=considerations,
		continue_reasoning=continue_reasoning,
		internal_reasoning=internal_reasoning,
		prefix=prefix,
	)


def create_mock_response(
	action: str,
	considerations: str = "Mock considerations",
	arguments: dict[str, Any] | None = None,
) -> str:
	"""Create a mock LLM response string in the expected format.

	The format follows the controller signature order: considerations, action, arguments.

	Parameters:
		action: The action name to include in the response
		considerations: The considerations text
		arguments: Optional arguments dict (will be JSON serialized)

	Returns:
		Formatted mock response string
	"""
	if arguments is not None:
		args_str = json.dumps(arguments)
		return (
			f"## {ControllerActionParameters.CONSIDERATIONS}\n"
			f"{considerations}\n"
			f"## {ControllerActionParameters.ACTION}\n"
			f"{action}\n"
			f"## {ControllerActionParameters.ARGUMENTS}\n"
			f"{args_str}"
		)
	return (
		f"## {ControllerActionParameters.CONSIDERATIONS}\n"
		f"{considerations}\n"
		f"## {ControllerActionParameters.ACTION}\n"
		f"{action}"
	)


def create_expected_prediction(
	tool_name: str,
	chosen_values: dict[str, Any],
	intervention_kwargs: dict[str, Any],
	tool_description: str = "Mock tool description",
	considerations: str = "Mock considerations",
	tool_execution_error: str = "",
	num_occurrences: int = 1,
) -> ControllerPrediction:
	"""Create an expected ControllerPrediction object for testing.

	Since the actual tool instance isn't available during parameterization,
	we create a dummy tool with the correct name.

	Parameters:
		tool_name: Name of the expected tool
		chosen_values: Expected arguments
		intervention_kwargs: Dict to construct ReasoningIntervention (continue_reasoning, internal_reasoning, prefix)
		tool_description: Description of the tool
		considerations: Expected considerations text
		num_occurrences: Expected occurrence count

	Returns:
		ControllerPrediction instance with expected values
	"""
	if tool_name == DEFAULT_REASONING_INTERVENTION_TOOL_NAME:
		# Infer tool structure from chosen_values keys
		tool = dspy.Tool(
			name=tool_name,
			desc=tool_description,
			func=lambda **kwargs: ReasoningIntervention(**intervention_kwargs),
		)
	elif tool_name == ControllerContinueReasoningChoice.FINISH:
		tool = FINISH_TOOL
	elif tool_name == ControllerContinueReasoningChoice.CONTINUE_REASONING:
		tool = DEFAULT_TOOL
	else:
		tool = dspy.Tool(name=tool_name, func=lambda: None, desc=tool_description)

	if intervention_kwargs:
		intervention = ReasoningIntervention(**intervention_kwargs)
	else:
		intervention = tool.func(**chosen_values)

	return ControllerPrediction(
		tool=tool,
		chosen_values=chosen_values,
		intervention=intervention,
		considerations=considerations,
		tool_execution_error=tool_execution_error,
		num_occurrences=num_occurrences,
	)


# =============================================================================
# Test: Controller Forward Method
# =============================================================================


@pytest.mark.parametrize(
	[
		"state",
		"signature",
		"action_space_keys",
		"early_stopping_enabled",
		"n_samples_generation",
		"temperature",
		"forced_choice_function",
		"mock_responses_func",
		"expected_predictions",
	],
	[
		# Test 1: Default tools, single sample, no reasoning
		pytest.param(
			State(									# state
				input={QuestionField.QUESTION: "What is 2+2?"},
				reasoning={QuestionField.REASONING_STEP: []},
				controller_output_trajectory=[],
				controller_outputs=[],
				feedback=[],
				output={},
			),
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[
					create_mock_response(ControllerContinueReasoningChoice.CONTINUE_REASONING)
				]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=ControllerContinueReasoningChoice.CONTINUE_REASONING,
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "",
						"prefix": ""
					},
				)
			]],
			id="default_tools_single_sample_no_reasoning",
		),
		# Test 2: Default tools, single sample, some reasoning
		pytest.param(
			State(									# state
				input={QuestionField.QUESTION: "What is 2+2?"},
				reasoning={QuestionField.REASONING_STEP: ["First, I need to think about addition."]},
				controller_output_trajectory=[create_dummy_controller_output()],
				controller_outputs=[],
				feedback=["Good start, but lacking in content."],
				output={},
			),
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			lambda: [[[create_mock_response(ControllerContinueReasoningChoice.CONTINUE_REASONING)]]],
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=ControllerContinueReasoningChoice.CONTINUE_REASONING,
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "",
						"prefix": ""
					},
				)
			]],
			id="default_tools_single_sample_some_reasoning",
		),
		# Test 3: Default tools, multiple samples - all duplicates (controller deduplicates)
		# When all samples produce the same action+args, controller returns 1 unique prediction
		# with num_occurrences tracking the count
		pytest.param(
			State(									# state
				input={QuestionField.QUESTION: "What is 2+2?"},
				reasoning={QuestionField.REASONING_STEP: ["First, I need to think about addition."]},
				controller_output_trajectory=[create_dummy_controller_output()],
				controller_outputs=[],
				feedback=["Not enough content."],
				output={},
			),
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			3,										# n_samples_generation
			0.7,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[
					create_mock_response(
						action=ControllerContinueReasoningChoice.CONTINUE_REASONING,
						considerations="First",
					),
					create_mock_response(
						action=ControllerContinueReasoningChoice.CONTINUE_REASONING,
						considerations="Second",
					),
					create_mock_response(
						action=ControllerContinueReasoningChoice.CONTINUE_REASONING,
						considerations="Third",
					),
				]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=ControllerContinueReasoningChoice.CONTINUE_REASONING,
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="First",  # Controller behavior: takes the first encountered considerations for the unique action set
					num_occurrences=3,
				),
			]],
			id="default_tools_multiple_samples_all_duplicates",
		),
		# Test 4: Default tools, multiple samples - mixed (continue and finish)
		# 2 continue + 1 finish = 2 unique predictions
		pytest.param(
			State(									# state
				input={QuestionField.QUESTION: "What is 2+2?"},
				reasoning={QuestionField.REASONING_STEP: ["First, I need to think about addition."]},
				controller_output_trajectory=[create_dummy_controller_output()],
				controller_outputs=[],
				feedback=["Not enough content."],
				output={},
			),
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			3,										# n_samples_generation
			0.7,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[
					create_mock_response(
						action=ControllerContinueReasoningChoice.CONTINUE_REASONING,
						considerations="Continue 1",
					),
					create_mock_response(
						action=ControllerContinueReasoningChoice.FINISH,
						considerations="Finish 1",
					),
					create_mock_response(
						action=ControllerContinueReasoningChoice.CONTINUE_REASONING,
						considerations="Continue 2",
					),
				]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=ControllerContinueReasoningChoice.CONTINUE_REASONING,
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="Continue 1",
					num_occurrences=2,
				),
				create_expected_prediction(
					tool_name=ControllerContinueReasoningChoice.FINISH,
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": False,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="Finish 1",
					num_occurrences=1,
				),
			]],
			id="default_tools_multiple_samples_mixed_actions",
		),
		# Test 5: Action space (styles), single sample
		pytest.param(
			State(									# state
				input={
					ArgumentField.TOPIC: "Climate change action",
					ArgumentField.STANCE: ArgumentStance.PRO,
				},
				reasoning={
					ArgumentField.CLAIM: [
						"Our planet is like a feverish patient needing urgent care."
					]
				},
				controller_output_trajectory=[
					create_dummy_controller_output(
						action=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						action_arguments={"style": "Figurative Language"},
						tool_descriptions="Use metaphor, simile, analogy, or symbolism to make ideas concrete.",
						considerations="I should employ non-literal comparison to make abstract concepts vivid. ",
					)
				],
				controller_outputs=[],
				feedback=["Vivid metaphor."],
				output={},
			),
			GenerateArgumentWithReasoning,			# signature
			["styles"],								# action_space_key
			True,									# early_stopping_enabled
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			# mock_responses_func (a different style than the previous step):
			(
				lambda: [[[
					create_mock_response(
						action=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						considerations="I want to use statistics and data.",
						arguments={"style": "Statistical & Data-Driven"},
					)
				]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"style": "Statistical & Data-Driven"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should use numbers and data to provide concrete, measurable support. ",
						"prefix": "",
					},
					considerations="I want to use statistics and data.",
				)
			]],
			id="styles_action_space_single_sample",
		),
		# Test 6: Action space (structures), single sample
		pytest.param(
			State(									# state
				input={
					ArgumentField.TOPIC: "Govermnets should do more to prevent climate change",
					ArgumentField.STANCE: ArgumentStance.PRO,
				},
				reasoning={ArgumentField.CLAIM: ["According to recent studies, climate change is accelerating."]},
				controller_output_trajectory=[
					create_dummy_controller_output(action_arguments={"structure": "Evidence & Support"})
				],
				controller_outputs=[],
				feedback=["Good usage of evidence to support the claim."],
				output={},
			),
			GenerateArgumentWithReasoning,			# signature
			["structures"],							# action_space_key
			True,									# early_stopping_enabled
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[
					create_mock_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						"Using causal reasoning",
						{"structure": "Causal Reasoning"},
					)
				]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"structure": "Causal Reasoning"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should use causal reasoning.",
						"prefix": "Therefore",
					},
					considerations="Using causal reasoning",
				)
			]],
			id="structures_action_space_single_sample",
		),
		# Test 7: Action space (styles + structures), single sample
		pytest.param(
			State(									# state
				input={
					ArgumentField.TOPIC: "Climate change action",
					ArgumentField.STANCE: ArgumentStance.PRO,
				},
				reasoning={ArgumentField.CLAIM: ["According to 97% of climate scientists, the warming trend is undeniable."]},
				controller_output_trajectory=[
					create_dummy_controller_output(
						action_arguments={"style": "Statistical & Data-Driven", "structure": "Evidence & Support"}
					)
				],
				controller_outputs=[],
				feedback=["Effective use of statistics and evidence."],
				output={},
			),
			GenerateArgumentWithReasoning,			# signature
			["styles", "structures"],				# action_space_key
			True,									# early_stopping_enabled
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[
					create_mock_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						"I want to provide statistical and data-driven evidence",
						{"style": "Statistical & Data-Driven", "structure": "Contrast"},
					)
				]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"style": "Statistical & Data-Driven", "structure": "Contrast"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should use numbers and data to provide concrete, measurable support. ",
						"prefix": "However",
					},
					considerations="I want to provide statistical and data-driven evidence",
				)
			]],
			id="styles_structures_action_space_single_sample",
		),
		# Test 8: Action space, multiple samples with different choices (no duplicates)
		pytest.param(
			State(									# state
				input={
					ArgumentField.TOPIC: "Climate change action",
					ArgumentField.STANCE: ArgumentStance.PRO,
				},
				reasoning={ArgumentField.CLAIM: ["Our planet is like a feverish patient."]},
				controller_output_trajectory=[
					create_dummy_controller_output(action_arguments={"style": "Figurative Language"})
				],
				controller_outputs=[],
				feedback=["Vivid metaphor."],
				output={},
			),
			GenerateArgumentWithReasoning,			# signature
			["styles"],								# action_space_key
			True,									# early_stopping_enabled
			2,										# n_samples_generation
			0.7,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[
					create_mock_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						"Figurative choice",
						{"style": "Figurative Language"},
					),
					create_mock_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						"Statistical choice",
						{"style": "Statistical & Data-Driven"},
					),
				]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"style": "Figurative Language"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should employ non-literal comparison to make abstract concepts vivid. ",
						"prefix": "",
					},
					considerations="Figurative choice",
				),
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"style": "Statistical & Data-Driven"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should use numbers and data to provide concrete, measurable support. ",
						"prefix": "",
					},
					considerations="Statistical choice",
				),
			]],
			id="styles_action_space_multiple_different_choices",
		),
		# Test 9: Action space, multiple samples with duplicate choices
		# 2 figurative + 1 statistical = 2 unique predictions
		pytest.param(
			State(									# state
				input={
					ArgumentField.TOPIC: "Climate change action",
					ArgumentField.STANCE: ArgumentStance.PRO,
				},
				reasoning={ArgumentField.CLAIM: ["Our planet is like a feverish patient."]},
				controller_output_trajectory=[
					create_dummy_controller_output(action_arguments={"style": "Figurative Language"})
				],
				controller_outputs=[],
				feedback=["Good metaphor."],
				output={},
			),
			GenerateArgumentWithReasoning,			# signature
			["styles"],								# action_space_key
			True,									# early_stopping_enabled
			3,										# n_samples_generation
			0.7,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[
					create_mock_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						"First figurative",
						{"style": "Figurative Language"},
					),
					create_mock_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						"Statistical",
						{"style": "Statistical & Data-Driven"},
					),
					create_mock_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						"Second figurative",
						{"style": "Figurative Language"},
					),
				]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"style": "Figurative Language"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should employ non-literal comparison to make abstract concepts vivid. ",
						"prefix": "",
					},
					considerations="First figurative",
					num_occurrences=2,
				),
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"style": "Statistical & Data-Driven"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should use numbers and data to provide concrete, measurable support. ",
						"prefix": "",
					},
					considerations="Statistical",
					num_occurrences=1,
				),
			]],
			id="styles_action_space_with_duplicate_choices",
		),
		# Test 10: Forced choice function - single forced action
		pytest.param(
			State(									# state
				input={QuestionField.QUESTION: "What is 2+2?"},
				reasoning={QuestionField.REASONING_STEP: ["First, I need to think about addition."]},
				controller_output_trajectory=[create_dummy_controller_output()],
				controller_outputs=[],
				feedback=[],
				output={},
			),
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			1,										# n_samples_generation
			0.0,									# temperature
			(										# forced_choice_function
				lambda tools, state: [(ControllerContinueReasoningChoice.FINISH, {}, "Forced finish")]
			),
			None,									# mock_responses_func (no mock needed)
			[[										# expected_predictions
				create_expected_prediction(
					ControllerContinueReasoningChoice.FINISH,
					{},
					{
						"continue_reasoning": False,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="Forced finish",
				)
			]],
			id="forced_choice_single_action",
		),
		# Test 11: Forced choice function - multiple forced actions
		pytest.param(
			State(									# state
				input={QuestionField.QUESTION: "What is 2+2?"},
				reasoning={QuestionField.REASONING_STEP: ["First, I need to think about addition."]},
				controller_output_trajectory=[create_dummy_controller_output()],
				controller_outputs=[],
				feedback=[],
				output={},
			),
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			2,										# n_samples_generation
			0.0,									# temperature
			(										# forced_choice_function
				lambda tools, state: [
					(ControllerContinueReasoningChoice.CONTINUE_REASONING, {}, "Forced continue"),
					(ControllerContinueReasoningChoice.FINISH, {}, "Forced finish"),
				]
			),
			None,									# mock_responses_func (no mock needed)
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=ControllerContinueReasoningChoice.CONTINUE_REASONING,
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="Forced continue",
				),
				create_expected_prediction(
					tool_name=ControllerContinueReasoningChoice.FINISH,
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": False,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="Forced finish",
				),
			]],
			id="forced_choice_multiple_actions",
		),
		# Test 12: No early stopping
		# When early_stopping_enabled=False, only DEFAULT_TOOL exists
		# Since tools have no arguments, ARGUMENTS field is not included
		pytest.param(
			State(									# state
				input={QuestionField.QUESTION: "What is 2+2?"},
				reasoning={QuestionField.REASONING_STEP: ["First, I need to think about addition."]},
				controller_output_trajectory=[create_dummy_controller_output()],
				controller_outputs=[],
				feedback=["Good step."],
				output={},
			),
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			False,									# early_stopping_enabled
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[create_mock_response(
					ControllerContinueReasoningChoice.CONTINUE_REASONING,
					"Only option",
				)]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=ControllerContinueReasoningChoice.CONTINUE_REASONING,
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="Forced choice: 'continue_reasoning' was the only available action. No other tools were provided to the controller.",
				)
			]],
			id="no_early_stopping",
		),
		# Test 13: Ready for final response (many reasoning steps)
		pytest.param(
			State(									# state
				input={MathField.MATH_PROBLEM: "2+2"},
				reasoning={
					MathField.MATH_OPERATION: [
						"This is a simple addition problem.",
						"Adding 2 and 2 together.",
						"The answer is 4.",
					]
				},
				controller_output_trajectory=[create_dummy_controller_output()] * 3,
				controller_outputs=[],
				feedback=["I agree.", "Reasonable next step.", "Correct."],
				output={},
			),
			SolveMathProblemWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[create_mock_response(ControllerContinueReasoningChoice.FINISH, "Solution complete")]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=ControllerContinueReasoningChoice.FINISH,
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": False,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="Solution complete",
				)
			]],
			id="ready_for_final_response",
		),
		# Test 14: Combined styles+structures, multiple samples with mixed duplicates
		# Choices 1 and 3 are duplicates, so 3 unique predictions
		pytest.param(
			State(									# state
				input={
					ArgumentField.TOPIC: "Climate change action",
					ArgumentField.STANCE: ArgumentStance.PRO,
				},
				reasoning={ArgumentField.CLAIM: ["We need to consider the evidence."]},
				controller_output_trajectory=[],
				controller_outputs=[],
				feedback=[],
				output={},
			),
			GenerateArgumentWithReasoning,			# signature
			["styles", "structures"],				# action_space_key
			True,									# early_stopping_enabled
			4,										# n_samples_generation
			0.8,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[
					create_mock_response(
						action=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						considerations="Choice 1",
						arguments={"style": "Figurative Language", "structure": "Causal Reasoning"},
					),
					create_mock_response(
						action=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						considerations="Choice 2",
						arguments={"style": "Statistical & Data-Driven", "structure": "Evidence & Support"},
					),
					create_mock_response(
						action=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						considerations="Choice 3 (dup of 1)",
						arguments={"style": "Figurative Language", "structure": "Causal Reasoning"},
					),
					create_mock_response(
						action=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						considerations="Choice 4",
						arguments={"style": "Figurative Language", "structure": "Contrast"},
					),
				]]]
			),
			# expected_predictions
			[[
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"style": "Figurative Language", "structure": "Causal Reasoning"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should employ non-literal comparison to make abstract concepts vivid.  I should use causal reasoning.",
						"prefix": "Therefore",
					},
					considerations="Choice 1",
					num_occurrences=2,
				),
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"style": "Statistical & Data-Driven", "structure": "Evidence & Support"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should use numbers and data to provide concrete, measurable support. ",
						"prefix": "According to",
					},
					considerations="Choice 2",
					num_occurrences=1,
				),
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"style": "Figurative Language", "structure": "Contrast"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should employ non-literal comparison to make abstract concepts vivid. ",
						"prefix": "However",
					},
					considerations="Choice 4",
					num_occurrences=1,
				),
			]],
			id="styles_structures_multiple_with_duplicates",
		),
		# Test 15: Single state with reasoning, aligned controller output (prefix)
		pytest.param(
			State(									# state
				input={
					ArgumentField.TOPIC: "Climate change action",
					ArgumentField.STANCE: ArgumentStance.PRO,
				},
				reasoning={ArgumentField.CLAIM: ["We need to consider the evidence."]},
				controller_output_trajectory=[create_dummy_controller_output()],
				controller_outputs=[],
				feedback=[],
				output={},
			),
			GenerateArgumentWithReasoning,			# signature
			["structures"],							# action_space_key
			True,									# early_stopping_enabled
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[
					create_mock_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						"Using causal reasoning",
						{"structure": "Causal Reasoning"},
					)
				]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"structure": "Causal Reasoning"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should use causal reasoning.",
						"prefix": "Therefore", # Verify prefix alignment
					},
					considerations="Using causal reasoning",
				)
			]],
			id="single_state_aligned_output",
		),
		# Test 16: Layer of 2 states (same parent), each generating 3 thoughts/predictions
		# State 1: 3 continue
		# State 2: 2 continue, 1 finish
		pytest.param(
			[										# states (list of 2 states)
				State(
					input={QuestionField.QUESTION: "What is 2+2?"},
					reasoning={QuestionField.REASONING_STEP: ["Common parent reasoning"]},
					controller_output_trajectory=[create_dummy_controller_output()],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				State(
					input={QuestionField.QUESTION: "What is 2+2?"},
					reasoning={QuestionField.REASONING_STEP: ["Common parent reasoning"]},
					controller_output_trajectory=[create_dummy_controller_output()],
					controller_outputs=[],
					feedback=[],
					output={},
				),
			],
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			2,										# n_samples_generation
			0.7,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [
					# Layer 0 (Batched)
					[
						# State 1 predictions
						[
							create_mock_response(
								action=ControllerContinueReasoningChoice.CONTINUE_REASONING,
								considerations="S1 C1",
								arguments={},
							),
							create_mock_response(
								action=ControllerContinueReasoningChoice.CONTINUE_REASONING,
								considerations="S1 C2",
								arguments={},
							),
						],
						# State 2 predictions
						[
							create_mock_response(
								action=ControllerContinueReasoningChoice.CONTINUE_REASONING,
								considerations="S2 C1",
								arguments={},
							),
							create_mock_response(
								action=ControllerContinueReasoningChoice.FINISH,
								considerations="S2 F1",
								arguments={},
							),
						]
					]
				]
			),
			# expected_predictions
			[
				# State 1 Expectations
				[
					create_expected_prediction(
						tool_name=ControllerContinueReasoningChoice.CONTINUE_REASONING,
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="S1 C1",
						num_occurrences=2,
					),
				],
				# State 2 Expectations
				[
					create_expected_prediction(
						tool_name=ControllerContinueReasoningChoice.CONTINUE_REASONING,
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="S2 C1",
					),
					create_expected_prediction(
						tool_name=ControllerContinueReasoningChoice.FINISH,
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": False,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="S2 F1",
					),
				],
			],
			id="layer_method_two_states_same_parent",
		),
		# Test 17: Layer of 2 states with different reasoning trajectories
		# One early state, one later state.
		# Note: In real usage, states in a layer usually have same depth, but controller handles independent states
		# so depth difference is fine for testing robustness.
		pytest.param(
			[										# states
				State(		# Early state
					input={QuestionField.QUESTION: "What is 2+2?"},
					reasoning={QuestionField.REASONING_STEP: ["Start reasoning", "Step 2", "Step 3"]},
					controller_output_trajectory=[create_dummy_controller_output()] * 3,
					controller_outputs=[],
					feedback=[],
					output={},
				),
				State(		# Late state
					input={QuestionField.QUESTION: "What is 2+2?"},
					reasoning={QuestionField.REASONING_STEP: ["Start", "Middle", "Almost done"]},
					controller_output_trajectory=[create_dummy_controller_output()] * 3,
					controller_outputs=[],
					feedback=[],
					output={},
				),
			],
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [
					[
						[create_mock_response(
							action=ControllerContinueReasoningChoice.CONTINUE_REASONING,
							considerations="Keep going",
							arguments={},
						)],
						[create_mock_response(
							action=ControllerContinueReasoningChoice.FINISH,
							considerations="Values done",
							arguments={},
						)],
					]
				]
			),
			[										# expected_predictions
				[
					create_expected_prediction(
						tool_name=ControllerContinueReasoningChoice.CONTINUE_REASONING,
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="Keep going",
					)
				],
				[
					create_expected_prediction(
						tool_name=ControllerContinueReasoningChoice.FINISH,
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": False,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="Values done",
					)
				],
			],
			id="layer_method_different_trajectories",
		),
		# Test 18: Batch processing - Two states, single sample each
		pytest.param(
			[										# states (2 simple states)
				State(
					input={QuestionField.QUESTION: "Q1"},
					reasoning={QuestionField.REASONING_STEP: ["R1"]},
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				State(
					input={QuestionField.QUESTION: "Q2"},
					reasoning={QuestionField.REASONING_STEP: ["R2"]},
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				)
			],
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_keys
			True,									# early_stopping_enabled
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [
					[
						[create_mock_response(ControllerContinueReasoningChoice.CONTINUE_REASONING, "S0")],
						[create_mock_response(ControllerContinueReasoningChoice.FINISH, "S1")],
					]
				]
			),
			[										# expected_predictions
				[
					create_expected_prediction(
						tool_name=ControllerContinueReasoningChoice.CONTINUE_REASONING,
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="S0",
					)
				],
				[
					create_expected_prediction(
						tool_name=ControllerContinueReasoningChoice.FINISH,
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": False,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="S1",
					)
				],
			],
			id="batch_two_states_single_sample",
		),
		# Test 19: Batch processing - Three states, multiple samples from each
		pytest.param(
			[										# states (3 simple states)
				State(
					input={QuestionField.QUESTION: "Q0"},
					reasoning={},
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				State(
					input={QuestionField.QUESTION: "Q1"},
					reasoning={},
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				State(
					input={QuestionField.QUESTION: "Q2"},
					reasoning={},
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
			],
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_keys
			True,									# early_stopping_enabled
			2,										# n_samples_generation
			0.1,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [
					[
						# State 0: 2 continue
						[
							create_mock_response(ControllerContinueReasoningChoice.CONTINUE_REASONING, "S0-1"),
							create_mock_response(ControllerContinueReasoningChoice.CONTINUE_REASONING, "S0-2"),
						],
						# State 1: 1 finish, 1 continue
						[
							create_mock_response(ControllerContinueReasoningChoice.FINISH, "S1-1"),
							create_mock_response(ControllerContinueReasoningChoice.CONTINUE_REASONING, "S1-2"),
						],
						# State 2: 1 continue, 1 finish
						[
							create_mock_response(ControllerContinueReasoningChoice.CONTINUE_REASONING, "S2-1"),
							create_mock_response(ControllerContinueReasoningChoice.FINISH, "S2-2"),
						],
					]
				]
			),
			[										# expected_predictions
				[	# State 0: Deduped to 1
					create_expected_prediction(
						tool_name=ControllerContinueReasoningChoice.CONTINUE_REASONING,
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="S0-1",
						num_occurrences=2,
					)
				],
				[	# State 1: 2 unique
					create_expected_prediction(
						tool_name=ControllerContinueReasoningChoice.FINISH,
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": False,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="S1-1",
					),
					create_expected_prediction(
						tool_name=ControllerContinueReasoningChoice.CONTINUE_REASONING,
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "",
							"prefix": ""
						},
						considerations="S1-2",
					)
				],
				[	# State 2: 2 unique
					create_expected_prediction(
						tool_name=ControllerContinueReasoningChoice.CONTINUE_REASONING,
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "",
							"prefix": ""
						},
						considerations="S2-1",
					),
					create_expected_prediction(
						tool_name=ControllerContinueReasoningChoice.FINISH,
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": False,
							"internal_reasoning": "",
							"prefix": ""
						},
						considerations="S2-2",
					)
				],
			],
			id="batch_three_states_multiple_samples",
		),
		# Test 20: Batch processing - Two states, 3 samples each ('Structure' action space)
		pytest.param(
			[							# states (2 states with 2 reasoning layers)
				State(					# State 1: first trajectory
					input={
						ArgumentField.TOPIC: "Regulating AI Development",
						ArgumentField.STANCE: ArgumentStance.PRO,
					},
					reasoning={
						ArgumentField.CLAIM: [
							"First, AI capabilities are advancing faster than safety research.",
							"Therefore, we cannot predict or control emergent behaviors in advanced systems.",
						],
					},
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				State(
					input={
						ArgumentField.TOPIC: "Regulating AI Development",
						ArgumentField.STANCE: ArgumentStance.ANTI,
					},
					reasoning={
						ArgumentField.CLAIM: [
							"First, heavy compliance burdens will hurt startups more than incumbents.",
							"According to a 2023 study by Stanford HAI, compliance costs could reduce open-source innovation by 40%.",
						],
					},
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
			],
			GenerateArgumentWithReasoning,			# signature
			["structures"],							# action_space_keys (triggers STRUCTURE_CONTROLLER_DEMOS)
			True,									# early_stopping_enabled
			3,										# n_samples_generation
			0.1,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [
					[
						# State 0 (PRO): 2 "Causal Reasoning", 1 "Finish"
						# Controller suggests "Causal Reasoning" -> prefix "Therefore"
						[
							create_mock_response(
								"intervene_on_next_reasoning_step",
								"Reasoning logic: Need to explain the consequence of uncontrollability.",
								{"structure": "Causal Reasoning"},
							),
							create_mock_response(
								"intervene_on_next_reasoning_step",
								"Reasoning logic (duplicate): Need to conclude why this justifies regulation.",
								{"structure": "Causal Reasoning"},
							),
							create_mock_response(
								ControllerContinueReasoningChoice.FINISH,
								"Argument is complete and conclusion follows logically.",
								arguments={},
							),
						],
						# State 1 (CON): 1 "Causal Reasoning", 2 "Contrast"
						# Controller suggests "Contrast" -> prefix "However"
						[
							create_mock_response(
								"intervene_on_next_reasoning_step",
								"Reasoning logic: Need to show the effect of reduced innovation on national security.",
								{"structure": "Evidence & Support"},
							),
							create_mock_response(
								"intervene_on_next_reasoning_step",
								"Reasoning logic: Need to offset compliance benefits with innovation risks.",
								{"structure": "Contrast"},
							),
							create_mock_response(
								"intervene_on_next_reasoning_step",
								"Reasoning logic (duplicate): Need to present the counterpoint on cost.",
								{"structure": "Contrast"},
							),
						],
					]
				]
			),
			[										# expected_predictions
				[	# State 0 Results
					create_expected_prediction(
						tool_name="intervene_on_next_reasoning_step",
						chosen_values={"structure": "Causal Reasoning"},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "I should use causal reasoning.",
							"prefix": "Therefore", # Prefilled by create_expected_tool_for_structures?
						},
						considerations="Reasoning logic: Need to explain the consequence of uncontrollability.",
						num_occurrences=2,
					),
					create_expected_prediction(
						tool_name=ControllerContinueReasoningChoice.FINISH,
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": False,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="Argument is complete and conclusion follows logically.",
						num_occurrences=1,
					),
				],
				[	# State 1 Results
					create_expected_prediction(
						tool_name="intervene_on_next_reasoning_step",
						chosen_values={"structure": "Evidence & Support"},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "",
							"prefix": "According to",
						},
						considerations="Reasoning logic: Need to show the effect of reduced innovation on national security.",
						num_occurrences=1,
					),
					create_expected_prediction(
						tool_name="intervene_on_next_reasoning_step",
						chosen_values={"structure": "Contrast"},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "",
							"prefix": "However", # "Contrast" implies "However" prefix
						},
						considerations="Reasoning logic: Need to offset compliance benefits with innovation risks.",
						num_occurrences=2,
					),
				],
			],
			id="batch_two_states_shared_duplicate_actions",
		),
	],
)
def test_controller_forward(
	state: State | list[State],
	signature: type[ReasoningSignature],
	action_space_keys: list[str] | None,
	early_stopping_enabled: bool,
	n_samples_generation: int,
	temperature: float,
	forced_choice_function,
	mock_responses_func,
	expected_predictions: list[list[ControllerPrediction]],
	temp_action_space_styles,
	temp_action_space_structures,
	temp_action_space_subtopics,
) -> None:
	"""
	Test controller forward method with various configurations.

	This test uses MockPredict and MockGenerativeLocalVLLM from utilities_for_tests.py
	to simulate LLM responses. It verifies that the controller properly handles:
	- Different action space configurations
	- Different state types (no reasoning, some reasoning, ready for final)
	- Different sampling parameters
	- Forced choice functions
	- Early stopping on/off
	- Duplicate and unique action responses

	Parameters:
		state: The state to test with
		signature: The reasoning signature class to test
		action_space_keys: List of keys for action space fixtures or None.
			Valid keys: "styles", "structures", "subtopics"
		early_stopping_enabled: Whether to enable early stopping
		n_samples_generation: Number of samples to generate
		temperature: Temperature for generation
		forced_choice_function: Optional forced choice function
		mock_responses_func: Function that returns mock responses for MockPredict
		expected_predictions: Expected list of list of ControllerPrediction objects.
	"""
	# Convert fixture keys to actual paths
	if action_space_keys is None:
		action_space_paths = None
	else:
		action_space_paths = []
		for key in action_space_keys:
			if key == "styles":
				action_space_paths.append(temp_action_space_styles)
			elif key == "structures":
				action_space_paths.append(temp_action_space_structures)
			elif key == "subtopics":
				action_space_paths.append(temp_action_space_subtopics)

	# Create controller
	controller = TreeOfThoughtsController(
		signature=signature,
		max_reasoning_steps=10,
		action_space_paths=action_space_paths,
		early_stopping_enabled=early_stopping_enabled,
		forced_choice_function=forced_choice_function,
	)

	# State is provided as a parameter

	# Set up mock predictor if we have mock responses to use
	if mock_responses_func is not None:
		controller_signature = controller._create_controller_signature_single_candidate()
		mock_responses = mock_responses_func()
		mock_predictor = MockPredict(mock_responses, signature=controller_signature)
		controller.decide_next_step_single = mock_predictor

	# Call forward
	result = controller.forward(
		states=state,
		n_samples_generation=n_samples_generation,
		temperature=temperature,
		candidate_generation_method=CandidateGenerationMethod.SINGLE_CANDIDATE_CALLS,
	)

	# Verify result structure
	assert isinstance(result, list), "Result should be a list"
	assert len(result) == len(expected_predictions), \
		"Result length should match expected_predictions length"

	for state_idx, (actual_list, expected_list) in enumerate(
		zip(result, expected_predictions, strict=True)
	):
		assert isinstance(actual_list, list), (
			f"State {state_idx}: Each state entry should be a list of predictions"
		)

		# Sort both lists to ensure consistent comparison order
		# Key: tool name, sorted args, considerations
		# TODO[P3]: Remove helpers like this. Consider moving this to the top of the file, or using
		# in-line logic.
		def sort_key(p):
			return (p.tool.name, tuple(sorted(p.chosen_values.items())), p.considerations)

		actual_list_sorted = sorted(actual_list, key=sort_key)
		expected_list_sorted = sorted(expected_list, key=sort_key)

		assert len(actual_list_sorted) == len(expected_list_sorted), (
			f"State {state_idx}: Expected {len(expected_list_sorted)} unique predictions, "
			f"got {len(actual_list_sorted)}"
		)

		for i, (actual, expected) in enumerate(
			zip(actual_list_sorted, expected_list_sorted, strict=True)
		):
			assert isinstance(actual, ControllerPrediction), (
				f"State {state_idx}, Prediction {i} should be ControllerPrediction, "
				f"got {type(actual)}"
			)

			# Verify tool name
			assert actual.tool.name == expected.tool.name, (
				f"State {state_idx}, Prediction {i}: Expected tool '{expected.tool.name}', "
				f"got '{actual.tool.name}'"
			)

			# Verify chosen values
			assert actual.chosen_values == expected.chosen_values, (
				f"State {state_idx}, Prediction {i}: Expected arguments {expected.chosen_values}, "
				f"got {actual.chosen_values}"
			)

			# Verify intervention fields
			assert actual.intervention.continue_reasoning == expected.intervention.continue_reasoning, (
				f"State {state_idx}, Prediction {i}: Expected continue_reasoning {expected.intervention.continue_reasoning}, "
				f"got {actual.intervention.continue_reasoning}"
			)

			assert actual.intervention.internal_reasoning == expected.intervention.internal_reasoning, (
				f"\n\nState {state_idx}, Prediction {i}: Internal reasoning mismatch.\n"
				f"Expected: {repr(expected.intervention.internal_reasoning)}\n"
				f"Actual:   {repr(actual.intervention.internal_reasoning)}"
			)

			assert actual.intervention.prefix == expected.intervention.prefix, (
				f"\n\nState {state_idx}, Prediction {i}: Prefix mismatch.\n"
				f"Expected: '{expected.intervention.prefix}'\n"
				f"Actual:   '{actual.intervention.prefix}'"
			)

			# Verify considerations - check if expected consideration matches actual
			# We might relax this if exact match isn't passed in parametrization,
			# but assuming we use create_expected_prediction which defaults to "Mock considerations"
			assert actual.considerations == expected.considerations, (
				f"\n\nState {state_idx}, Prediction {i}: Considerations mismatch"
			)

			# Verify occurrences
			assert actual.num_occurrences == expected.num_occurrences, (
				f"\n\nState {state_idx}, Prediction {i}: Expected {expected.num_occurrences} occurrences, got {actual.num_occurrences}"
			)


# =============================================================================
# Test: Forced Choice Function
# =============================================================================


@pytest.mark.parametrize(
	[
		"available_tools",
		"expected_action",
		"expected_arguments",
		"expected_considerations_contains",
	],
	[
		pytest.param(
			{ControllerContinueReasoningChoice.CONTINUE_REASONING: DEFAULT_TOOL},
			ControllerContinueReasoningChoice.CONTINUE_REASONING,
			{},
			"only available action",
			id="forced_continue_reasoning",
		),
		pytest.param(
			{ControllerContinueReasoningChoice.FINISH: FINISH_TOOL},
			ControllerContinueReasoningChoice.FINISH,
			{},
			"only available action",
			id="forced_finish",
		),
		pytest.param(
			{
				ControllerContinueReasoningChoice.CONTINUE_REASONING: DEFAULT_TOOL,
				ControllerContinueReasoningChoice.FINISH: FINISH_TOOL,
			},
			None,
			None,
			None,
			id="no_forced_choice",
		),
	],
)
def test_forced_choice_function(
	simple_state: State,
	available_tools: dict[str, dspy.Tool],
	expected_action: str | None,
	expected_arguments: dict[str, Any] | None,
	expected_considerations_contains: str | None,
) -> None:
	"""Test the return_action_if_single_option forced choice function."""
	result = return_action_if_single_option(available_tools, simple_state)

	if expected_action is None:
		assert result is None
	else:
		assert result is not None
		assert isinstance(result, list)
		assert len(result) == 1
		action_name, action_arguments, considerations = result[0]
		assert action_name == expected_action
		assert action_arguments == expected_arguments
		assert expected_considerations_contains in considerations


# =============================================================================
# Test: Controller Error Handling
# =============================================================================


def test_controller_error_handling_nonexistent_tool(simple_state: State) -> None:
	"""Test controller error handling when forced choice returns nonexistent tool."""

	def broken_choice_function(tools, state):
		return [("nonexistent_tool", {}, "This tool doesn't exist")]

	controller = TreeOfThoughtsController(
		signature=QuestionAnsweringWithReasoning,
		max_reasoning_steps=5,
		forced_choice_function=broken_choice_function,
	)

	with pytest.raises(AssertionError) as exc_info:
		controller.forward(
			states=simple_state,
			candidate_generation_method=CandidateGenerationMethod.SINGLE_CANDIDATE_CALLS,
		)
	assert "nonexistent_tool" in str(exc_info.value)


# =============================================================================
# Test: Duplicate Removal
# =============================================================================


@pytest.mark.parametrize(
	# Parameter names
	[
	"input_dicts",
	"expected_results",
	],
	# Parameter values
	[
		pytest.param(
		[
				{
					ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.CONTINUE_REASONING,
					ControllerActionParameters.ARGUMENTS: {},
				ControllerActionParameters.CONSIDERATIONS: "First",
				},
				{
					ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.CONTINUE_REASONING,
					ControllerActionParameters.ARGUMENTS: {},
				ControllerActionParameters.CONSIDERATIONS: "Second",
			},
		],
		[
				{
					ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.CONTINUE_REASONING,
					ControllerActionParameters.ARGUMENTS: {},
				ControllerActionParameters.CONSIDERATIONS: "First",
				ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT: 2,
			}
		],
		id="duplicate_continue_reasoning",
		),
		pytest.param(
		[
			{
				ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.CONTINUE_REASONING,
					ControllerActionParameters.ARGUMENTS: {},
				ControllerActionParameters.CONSIDERATIONS: "First",
				},
				{
					ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.FINISH,
					ControllerActionParameters.ARGUMENTS: {},
				ControllerActionParameters.CONSIDERATIONS: "Different",
			},
		],
		[
				{
					ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.CONTINUE_REASONING,
					ControllerActionParameters.ARGUMENTS: {},
				ControllerActionParameters.CONSIDERATIONS: "First",
				ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT: 1,
			},
				{
					ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.FINISH,
					ControllerActionParameters.ARGUMENTS: {},
				ControllerActionParameters.CONSIDERATIONS: "Different",
				ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT: 1,
			},
			],
		id="no_duplicates",
		),
		pytest.param(
		[
			{
				ControllerActionParameters.ACTION: "intervene_on_next_reasoning_step",
				ControllerActionParameters.ARGUMENTS: {"style": "Power", "structure": "Cause"},
				ControllerActionParameters.CONSIDERATIONS: "First",
			},
			{
				ControllerActionParameters.ACTION: "intervene_on_next_reasoning_step",
				ControllerActionParameters.ARGUMENTS: {"style": "Knowledge", "structure": "Cause"},
				ControllerActionParameters.CONSIDERATIONS: "Second",
			},
			{
				ControllerActionParameters.ACTION: "intervene_on_next_reasoning_step",
				ControllerActionParameters.ARGUMENTS: {"structure": "Cause", "style": "Power"},
				ControllerActionParameters.CONSIDERATIONS: "Third",
			},
		],
		[
			{
				ControllerActionParameters.ACTION: "intervene_on_next_reasoning_step",
				ControllerActionParameters.ARGUMENTS: {"style": "Power", "structure": "Cause"},
				ControllerActionParameters.CONSIDERATIONS: "First",
				ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT: 2,
			},
			{
				ControllerActionParameters.ACTION: "intervene_on_next_reasoning_step",
				ControllerActionParameters.ARGUMENTS: {"style": "Knowledge", "structure": "Cause"},
				ControllerActionParameters.CONSIDERATIONS: "Second",
				ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT: 1,
			},
		],
		id="duplicate_with_complex_args",
		),
	],
)
def test_duplicate_removal(
	input_dicts: list[dict[str, Any]],
	expected_results: list[dict[str, Any]],
) -> None:
	"""Test the duplicate action removal functionality."""
	result = remove_duplicate_actions_with_counts(input_dicts)

	assert len(result) == len(expected_results)
	for i, expected in enumerate(expected_results):
		assert result[i][ControllerActionParameters.ACTION] == expected[ControllerActionParameters.ACTION]
		assert result[i][ControllerActionParameters.ARGUMENTS] == expected[ControllerActionParameters.ARGUMENTS]
		assert result[i][ControllerActionParameters.CONSIDERATIONS] == expected[ControllerActionParameters.CONSIDERATIONS]
		assert result[i][ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT] == expected[ControllerOutputParameters.UNIQUE_ACTION_RESPONSE_COUNT]


# =============================================================================
# Test: Controller Action Validation
# =============================================================================


@pytest.mark.parametrize(
	["action_value", "expected_error_substring"],
	[
		pytest.param("", "empty or invalid action", id="empty_string"),
		pytest.param(None, "empty or invalid action", id="none_value"),
		pytest.param("   ", "empty or invalid action", id="whitespace_only"),
		pytest.param("invalid_tool_name", "unknown action", id="unknown_action"),
	],
)
def test_invalid_action_raises_error(
	action_value: Any,
	expected_error_substring: str,
) -> None:
	"""Test that invalid actions raise AssertionError."""
	controller = TreeOfThoughtsController(
		signature=QuestionAnsweringWithReasoning,
		max_reasoning_steps=5,
	)
	signature = controller._create_controller_signature_single_candidate()

	prediction = dspy.Prediction.from_completions(
		[
			{
				ControllerActionParameters.ACTION: action_value,
				ControllerActionParameters.CONSIDERATIONS: "Some considerations",
			}
		],
		signature=signature,
	)

	with pytest.raises(AssertionError) as exc_info:
		controller.create_controller_predictions(prediction)

	assert expected_error_substring in str(exc_info.value)


@pytest.mark.parametrize(
	["quoted_action", "expected_action"],
	[
		pytest.param(
			'"continue_reasoning"',
			ControllerContinueReasoningChoice.CONTINUE_REASONING,
			id="double_quotes",
		),
		pytest.param(
			"'continue_reasoning'",
			ControllerContinueReasoningChoice.CONTINUE_REASONING,
			id="single_quotes",
		),
		pytest.param(
			"`continue_reasoning`",
			ControllerContinueReasoningChoice.CONTINUE_REASONING,
			id="backticks",
		),
	],
)
def test_action_with_quotes_strips_correctly(
	quoted_action: str,
	expected_action: str,
) -> None:
	"""Test that actions with surrounding quotes are stripped correctly."""
	controller = TreeOfThoughtsController(
		signature=QuestionAnsweringWithReasoning,
		max_reasoning_steps=5,
	)
	signature = controller._create_controller_signature_single_candidate()

	prediction = dspy.Prediction.from_completions(
		[{ControllerActionParameters.ACTION: quoted_action, ControllerActionParameters.CONSIDERATIONS: "Test"}],
		signature=signature,
	)

	result = controller.create_controller_predictions(prediction)

	assert len(result) == 1
	assert result[0].tool.name == expected_action


# =============================================================================
# Integration Tests (GPU Required)
# =============================================================================


@pytest.fixture(scope="module")
def shared_gpu_model():
	"""Shared GenerativeLocalVLLM fixture for all GPU integration tests."""
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
	finally:
		if lm is not None:
			logger.info("Cleaning up shared GPU model...")
			lm.kill()




@pytestmark_gpu
class TestControllerIntegration:
	"""Integration tests for the controller using real models (requires GPU)."""

	@pytest.fixture(scope="class")
	def local_lm(self, shared_gpu_model):
		"""Use the shared GPU model fixture."""
		return shared_gpu_model

	@pytest.fixture
	def controller(self, local_lm):
		"""Create a controller instance with real LM."""
		dspy.settings.configure(lm=local_lm)
		return TreeOfThoughtsController(
			signature=SolveMathProblemWithReasoning,
			max_reasoning_steps=5,
			tools=[DEFAULT_TOOL],
			early_stopping_enabled=True,
			verbosity=Verbosity.INFO,
		)

	@pytest.mark.parametrize(
		["example_name", "demos", "n_samples", "temperature"],
		[
			pytest.param("No Demos", None, 1, 0.1, id="no_demos"),
			pytest.param("With Demos", CONTROLLER_DEMOS, 1, 0.1, id="with_demos"),
			pytest.param("Multiple Samples", None, 2, 0.2, id="multiple_samples"),
		],
	)
	def test_controller_basic_execution(
		self,
		controller,
		example_name: str,
		demos: list[dict] | None,
		n_samples: int,
		temperature: float,
	) -> None:
		"""Test basic controller execution with different configurations."""
		single_state = State(
			input={
				MathField.MATH_PROBLEM: "Solve the system of equations: 3x + 2y = 7 and 5x - 4y = 1",
			},
			reasoning={
				MathField.MATH_OPERATION: [
					"I need to solve this system, but I'm not sure whether to use substitution or elimination.",
				]
			},
		)

		try:
			result = controller(
				states=single_state,
				n_samples_generation=n_samples,
				temperature=temperature,
				candidate_generation_method=CandidateGenerationMethod.SINGLE_CANDIDATE_CALLS,
				demos=demos,
			)
			assert result is not None
			assert len(result) > 0
			if isinstance(result, list):
				assert len(result) == 1
				assert len(result[0]) >= 1
		except Exception as e:
			pytest.fail(f"Controller execution failed: {e}")

	@pytest.mark.parametrize(
		# Parameter names
		[
			"state",
			"action_space_keys",
			"expected_choices_to_make",
			"expected_choices_to_avoid",
			"decision_rationale",
		],
		# Parameter values
		[
			# Structural Transition Tests
			# Clear cut Rebuttal (However)
			pytest.param(
				State(
					input={
						ArgumentField.TOPIC: "The earth is flat.",
						ArgumentField.STANCE: ArgumentStance.ANTI,
					},
					reasoning={
						ArgumentField.CLAIM: [
							"Note that from a local perspective on the ground, the horizon appears to be a flat line."
						]
					},
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				["structures"],
				[												# Expected choices to make
					(
						# The earth is not flat, so we expect a "However".
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"structure": "Contrast"},
					),
					(
						# Evidence that the earth is round or evidence refuting the flat view is also valid
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"structure": "Evidence & Support"},
					)
				],
				[												# Expected choices to avoid
					(
						# The model is unlikely to explain the implication of the eath being flat.
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"structure": "Causal Reasoning"},
					)
				],
				"Misleading 'Note that' premise requires 'However' transition to correct it",
				id="structure_preference_contrast_for_rebuttal",
			),
			pytest.param(
				State(
					input={
						ArgumentField.TOPIC: "The earth is flat.",
						ArgumentField.STANCE: ArgumentStance.ANTI,
					},
					reasoning={
						ArgumentField.CLAIM: [
							"Proponents claim that from a local perspective, the horizon looks flat."
						]
					},
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				["structures"],
				[												# Expected choices to make
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"structure": "Contrast"},
					),
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"structure": "Evidence & Support"},
					)
				],
				[												# Expected choices to avoid
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"structure": "Causal Reasoning"},
					)
				],
				"Misleading claim requires 'However' transition to rebut it",
				id="structure_preference_contrast_for_rebuttal",
			),
			pytest.param(
				State(
					input={
						ArgumentField.TOPIC: "Early education provides benefits that justify increased funding.",
						ArgumentField.STANCE: ArgumentStance.PRO,
					},
					reasoning={
						ArgumentField.CLAIM: [
							"Studies show that early education has long-term benefits.",
						]
					},
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				["structures"],
				[
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"structure": "Evidence & Support"},
					)
				],
				[
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"structure": "Contrast"},
					)
				],
				"Claim needs 'Evidence & Support' to back it up",
				id="structure_preference_evidence_for_claim",
			),
			# Clear cut Deduction (Therefore)
			pytest.param(
				State(
					input={
						ArgumentField.TOPIC: "Socrates is mortal.",
						ArgumentField.STANCE: ArgumentStance.PRO,
					},
					reasoning={
						ArgumentField.CLAIM: [
							"Note that all human beings are mortal, and that Socrates is a human being."
						]
					},
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				["structures"],
				[
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"structure": "Causal Reasoning"},
					),
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"structure": "Evidence & Support"},
					)
				],
				[
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"structure": "Contrast"},
					)
				],
				"Syllogism premises 'Note that...' require 'Therefore' conclusion",
				id="structure_preference_deductive_syllogism",
			),
			# Style Transition Tests
			pytest.param(
				State(
					input={
						ArgumentField.TOPIC: "Renewable energy investment",
						ArgumentField.STANCE: ArgumentStance.PRO,
					},
					reasoning={
						ArgumentField.CLAIM: [
							"Renewable energy is like planting seeds for future generations - "
							"we invest today to harvest clean power tomorrow."
						]
					},
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				["styles"],  # Only styles action space
				[
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"style": "Statistical & Data-Driven"},
					)
				],
				None,
				"After figurative language, should transition to data/statistical style for balance",
				id="style_figurative_to_statistical",
			),
			pytest.param(
				State(
					input={
						ArgumentField.TOPIC: "Education reform",
						ArgumentField.STANCE: ArgumentStance.PRO,
					},
					reasoning={
						ArgumentField.CLAIM: [
							"Studies show that smaller class sizes improve student outcomes by 15-20%.",
							"Data from OECD countries reveals a strong correlation between teacher-student "
							"ratios and academic performance."
						]
					},
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				["styles"],
				[
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"style": "Figurative Language"},
					)
				],
				[
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"style": "Statistical & Data-Driven"},
					)
				],
				"After multiple statistical statements, should vary style (e.g., narrative or figurative)",
				id="style_avoid_statistical_repetition",
			),
			# Early Stopping Tests (2 choices only - should pick the clear winner)
			pytest.param(
				State(
					input={
						ArgumentField.TOPIC: "Space exploration funding",
						ArgumentField.STANCE: ArgumentStance.PRO,
					},
					reasoning={
						ArgumentField.CLAIM: [
							"Space exploration drives technological innovation that benefits society.",
							"Historical evidence: NASA research led to GPS, memory foam, and water purification.",
							"The economic multiplier effect: every dollar invested returns $7-14 to the economy.",
							"Therefore, increasing space exploration funding is a wise investment in our future."
						]
					},
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				None,  # No action space - only continue_reasoning vs finish
				[
					(
						ControllerContinueReasoningChoice.FINISH,
						None,
					)
				],
				[
					(
						ControllerContinueReasoningChoice.CONTINUE_REASONING,
						None,
					)
				],
				"Complete argument with claim, evidence, and conclusion - should finish",
				id="early_stopping_complete_argument_finish",
			),
			pytest.param(
				State(
					input={
						ArgumentField.TOPIC: "Digital privacy rights",
						ArgumentField.STANCE: ArgumentStance.PRO,
					},
					reasoning={
						ArgumentField.CLAIM: [
							"Digital privacy is a fundamental human right in the modern age.",
						]
					},
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				None,
				[
					(
						ControllerContinueReasoningChoice.CONTINUE_REASONING,
						None,
					)
				],
				[
					(
						ControllerContinueReasoningChoice.FINISH,
						None,
					)
				],
				"Only a claim without evidence or conclusion - should continue reasoning",
				id="early_stopping_incomplete_argument_continue",
			),
			# Combined Structural + Style Test
			pytest.param(
				State(
					input={
						ArgumentField.TOPIC: "AI regulation",
						ArgumentField.STANCE: ArgumentStance.PRO,
					},
					reasoning={
						ArgumentField.CLAIM: [
							"AI systems must be regulated to prevent societal harm.",
							"Consider AI like a powerful river - without proper channels and dams, "
							"it can flood and destroy everything in its path."
						]
					},
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),

				["styles", "structures"],  # Multi-dimensional: Style + Structure
				[
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"style": "Statistical & Data-Driven"},
					)
				],
				None,
				"Claim with figurative language needs statistical data (style transition)",
				id="combined_claim_with_figurative_needs_evidence",
			),
			# Subtopic Selection Test
			# Context makes one subtopic clearly relevant (Economic) vs clearly irrelevant (Social Justice)
			pytest.param(
				State(
					input={
						ArgumentField.TOPIC: "Impact of inflation on consumer purchasing power",
						ArgumentField.STANCE: ArgumentStance.PRO,
					},
					reasoning={
						ArgumentField.CLAIM: [
							"First, rising prices reduce aggregate demand and slow GDP growth."
						]
					},
					controller_output_trajectory=[
						create_dummy_controller_output(
							action=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
							action_arguments={"stock_issue": "General Introduction"},
							considerations="I want to introduce the concept of inflation.",
						)
					],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				["subtopics"],  # Just subtopics
				[
					(
						# Likely to talk about economic impact because it's a clear economic issue
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"stock_issue": "Economic Impact"},
					),
					(
						# Likely to talk about social justice because the economic issue has social implications
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"stock_issue": "Social Justice & Equity"},
					)
				],
				None,
				"Topic of inflation/purchasing power requires Economic Impact subtopic",
				id="subtopic_relevance_economic",
			),
			# Multi-dimensional: Subtopic + Style
			pytest.param(
				State(
					input={
						ArgumentField.TOPIC: "Poetry is good for the soul",
						ArgumentField.STANCE: ArgumentStance.PRO,
					},
					reasoning={
						ArgumentField.CLAIM: [
							"First, here is a beautiful poem by Emily Dickinson:"
						]
					},
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				["subtopics", "styles"],
				[
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{
							"stock_issue": "General Introduction",
							"style": "Figurative Language"
						},
					),
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{
							"stock_issue": "Economic Impact",
							"style": "Figurative Language"
						},
					),
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{
							"stock_issue": "Social Justice & Equity",
							"style": "Figurative Language"
						},
					),
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,

						{
							"stock_issue": "Health and Safety",
							"style": "Figurative Language"
						},
					),
				],
				None,
				"Poetry analysis requires Figurative Language style",
				id="subtopic_and_style_relevance",
			),
		],
	)
	def test_semantic_validation(
		self,
		controller,
		temp_action_space_styles,
		temp_action_space_structures,
		temp_action_space_subtopics,
		state: State,
		action_space_keys: list[str] | None,
		expected_choices_to_make: list[tuple[str, dict[str, Any] | None]] | None,
		expected_choices_to_avoid: list[tuple[str, dict[str, Any] | None]] | None,
		decision_rationale: str,
	):
		"""Test semantic correctness of controller decisions with real LM.

		This test validates that the controller makes semantically appropriate decisions
		for structural transitions, style transitions, and early stopping scenarios.
		"""
		# Convert fixture keys to actual paths
		action_space_paths = None
		if action_space_keys is not None:
			action_space_paths = []
			for key in action_space_keys:
				if key == "styles":
					action_space_paths.append(temp_action_space_styles)
				elif key == "structures":
					action_space_paths.append(temp_action_space_structures)
				elif key == "subtopics":
					action_space_paths.append(temp_action_space_subtopics)

		# Create controller with appropriate action space
		test_controller = TreeOfThoughtsController(
			signature=GenerateArgumentWithReasoning,
			max_reasoning_steps=5,
			action_space_paths=action_space_paths,
			early_stopping_enabled=True,
		)

		# Select demos based on action space configuration
		# Uses argument-based demos since test uses GenerateArgumentWithReasoning
		# - No action space (continue/finish only): use argument continue/finish demos
		# - Styles only: use style demos
		# - Structures only: use structure demos
		# - Both: use combined style+structure demos
		if action_space_keys is None:
			demos_to_use = ARGUMENT_CONTINUE_FINISH_DEMOS  # argument fields + continue/finish
		elif action_space_keys == ["styles"]:
			demos_to_use = STYLE_CONTROLLER_DEMOS
		elif action_space_keys == ["structures"]:
			demos_to_use = STRUCTURE_CONTROLLER_DEMOS
		else:  # Both styles and structures
			demos_to_use = STYLE_STRUCTURE_CONTROLLER_DEMOS

		try:
			controller_result = test_controller(
				states=state,
				n_samples_generation=1,
				temperature=0.1,
				candidate_generation_method=CandidateGenerationMethod.SINGLE_CANDIDATE_CALLS,
				demos=demos_to_use,
			)

			# Extract actual decision
			# Extract actual decision and args
			actual_tool_name = None
			actual_args = None
			if (
				controller_result
				and isinstance(controller_result, list)
				and len(controller_result) > 0
			):
				first_state_result = controller_result[0]
				if isinstance(first_state_result, list) and len(first_state_result) > 0:
					first_response = first_state_result[0]
					if hasattr(first_response, "tool"):
						actual_tool_name = first_response.tool.name
						actual_args = first_response.chosen_values

			# Check expected choices to make (ANY of these)
			if expected_choices_to_make is not None:
				found_match = False
				for expected_tool_name, expected_args in expected_choices_to_make:
					if actual_tool_name == expected_tool_name:
						if expected_args is None:
							found_match = True  # Tool match is sufficient if no args expected
							break
						# Check if expected args are subset of actual args
						if actual_args and all(
							actual_args.get(k) == v for k, v in expected_args.items()
						):
							found_match = True
							break

				assert found_match, (
					f"Semantic validation failed: {decision_rationale}\n"
					f"Expected one of: {expected_choices_to_make}\n"
					f"Got: {actual_tool_name} with args {actual_args}\n"
					f"State input: {state.input}\n"
					f"State reasoning: {state.reasoning}"
				)

			# Check expected choices to avoid (NONE of these)
			if expected_choices_to_avoid is not None:
				for avoid_tool_name, avoid_args in expected_choices_to_avoid:
					is_match = False
					if actual_tool_name == avoid_tool_name:
						if avoid_args is None:
							is_match = True
						elif actual_args and all(
							actual_args.get(k) == v for k, v in avoid_args.items()
						):
							is_match = True

					assert not is_match, (
						f"Semantic validation failed: {decision_rationale}\n"
						f"Should have avoided: {avoid_tool_name} with args {avoid_args}\n"
						f"But got exactly that.\n"
						f"State input: {state.input}\n"
						f"State reasoning: {state.reasoning}"
					)


			logger.info(f"✓ Semantic validation passed: {decision_rationale}")

		except Exception as e:
			pytest.fail(f"Semantic validation failed: {e}")


if __name__ == "__main__":
	gpu_available = torch.cuda.is_available()
	if not gpu_available:
		pytest.main([__file__, "-vv"])
	else:
		pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO"])
