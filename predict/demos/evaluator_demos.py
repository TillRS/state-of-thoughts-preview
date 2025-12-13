"""
Evaluator Demo Constants

This module contains demonstration examples for the TreeOfThoughtEvaluator
that show how to evaluate reasoning chains (PRM) and final solutions (ORM)
with both positive and negative examples across different quality levels.

These demos are structured to match the vllm_adapter expected format and
provide clear examples of the 1-7 Likert scale evaluation criteria.
"""

# Standard library imports
from typing import Any

# Local imports
from signatures.example_signatures import MathField
from signatures.field_constants import EvaluationMetric
from tree.tree_constants import NodeField, ReasoningState, ToTField

# PRM Demos - Process Reward Model (Reasoning Chain Evaluation)
PRM_DEMOS: list[dict[str, Any]] = [
	{
		# Score 7/7: Excellent reasoning - systematic and thorough
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "If all managers attend meetings and Sarah attends all meetings, can we conclude Sarah is a manager?",
			ToTField.REASONING_STEPS: [
				"This is a logical reasoning problem about set relationships",
				"Let M = managers, A = people who attend meetings",
				"Given: All managers attend meetings (M ⊆ A) and Sarah attends meetings (Sarah ∈ A)",
				"We cannot conclude Sarah ∈ M because the relationship is one-way",
				"There could be non-managers who also attend meetings - the logic is invalid",
			],
		},
		ReasoningState.OUTPUT: {
			EvaluationMetric.SOUNDNESS: 7.0,
			EvaluationMetric.PROMISE: 7.0,
			ReasoningState.FEEDBACK: "Excellent reasoning demonstrating mastery of logical analysis. SOUNDNESS (7/7): Flawless logic with proper mathematical notation (M ⊆ A), correctly identifies the logical fallacy (affirming the consequent), and explicitly explains why the implication only works one way. All claims are well-supported with no errors. PROMISE (7/7): Optimal trajectory that systematically builds from problem identification to formal representation to logical analysis to conclusion. Shows exceptional progress toward a complete and rigorous solution with clear pedagogical value.",
		},
	},
	{
		# Score 6/5: Strong reasoning - correct with good explanation
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "Solve 3x + 7 = 22",
			ToTField.REASONING_STEPS: [
				"I need to isolate x by moving terms to opposite sides",
				"Subtract 7 from both sides: 3x + 7 - 7 = 22 - 7",
				"Simplify: 3x = 15",
				"Divide both sides by 3: x = 5",
			],
		},
		ReasoningState.OUTPUT: {
			EvaluationMetric.SOUNDNESS: 6.0,
			EvaluationMetric.PROMISE: 5.0,
			ReasoningState.FEEDBACK: "Strong reasoning with excellent algebraic technique. SOUNDNESS (6/7): Logically sound with only very minor imperfections - clearly states the goal, shows each operation with proper justification, and maintains equation balance. The work is mathematically correct and easy to follow. PROMISE (5/7): Reasonable trajectory showing meaningful progress. The approach is standard and effective, though not particularly innovative or insightful. Will clearly lead to the correct solution without detours.",
		},
	},
	{
		# Score 4/3: Average reasoning - adequate but could be clearer
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "Find the area of a parallelogram with base 6 and height 4",
			ToTField.REASONING_STEPS: [
				"parallelogram area uses base and height",
				"Area = base * height = (1/2) * 6 * 4 = 24",
			],
		},
		ReasoningState.OUTPUT: {
			EvaluationMetric.SOUNDNESS: 4.0,
			EvaluationMetric.PROMISE: 3.0,
			ReasoningState.FEEDBACK: "Adequate reasoning with notable issues. SOUNDNESS (4/7): Mostly correct approach but contains a significant conceptual error - incorrectly applies the triangle area formula (1/2 × base × height) to a parallelogram. While coincidentally getting the numerically correct answer (24), this shows confusion about geometric formulas. Lacks clarity in formula explanation. PROMISE (3/7): Questionable trajectory as the conceptual error suggests uncertain understanding that could lead to errors in related problems. The approach is too abbreviated and would mislead learners.",
		},
	},
	{
		# Score 2/2: Weak reasoning - major mathematical error
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "What is 15% of 80?",
			ToTField.REASONING_STEPS: [
				"15% means 15 out of 100",
				"So I need to find 15 * 80 / 100",
				"15 * 80 = 1200, and 1200 / 100 = 120",
			],
		},
		ReasoningState.OUTPUT: {
			EvaluationMetric.SOUNDNESS: 2.0,
			EvaluationMetric.PROMISE: 2.0,
			ReasoningState.FEEDBACK: "Weak reasoning with critical issues. SOUNDNESS (2/7): Major flaws that seriously undermine validity. The conceptual approach is sound (15% = 15/100), but contains a fundamental arithmetic error: states '15 × 80 = 1200, and 1200 ÷ 100 = 120' when 1200 ÷ 100 = 12. This computational error completely invalidates the result. PROMISE (2/7): Poor approach unlikely to lead to a strong solution. Despite correct method setup, the arithmetic error and failure to verify the result suggest weak problem-solving habits that will propagate errors.",
		},
	},
	{
		# Score 1/1: Poor reasoning - false formulae
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "Find the volume of a cube of side length 2.",
			ToTField.REASONING_STEPS: [
				"The volume a cube is V = s^2",
				"V = 2^2 = 4",
			],
		},
		ReasoningState.OUTPUT: {
			EvaluationMetric.SOUNDNESS: 1.0,
			EvaluationMetric.PROMISE: 1.0,
			ReasoningState.FEEDBACK: "Poor reasoning that is fundamentally flawed. SOUNDNESS (1/7): Fundamentally invalid reasoning with critical conceptual error. Uses the 2D area formula V = s^2 instead of the correct 3D volume formula V = s³ for a cube. This demonstrates complete disconnect from basic geometric principles and confusion between dimensions (2D vs 3D). PROMISE (1/7): Dead-end path with no meaningful progress. The fundamental misunderstanding of dimensionality means this approach cannot lead to the correct solution or build toward geometric understanding. Would seriously mislead learners.",
		},
	},
]

# ORM Demos - Outcome Reward Model (Final Solution Evaluation)
ORM_DEMOS: list[dict[str, Any]] = [
	{
		# Score 7: Excellent solution - complete, clear, and insightful
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "A recipe calls for 2/3 cup of flour, but you want to make 1.5 times the recipe. How much flour do you need?",
			ToTField.REASONING_STEPS: [
				"I need to find 1.5 times the original amount of 2/3 cup",
				"Converting 1.5 to a fraction: 1.5 = 3/2",
				"Multiplying fractions: (3/2) * (2/3) = 6/6 = 1",
			],
			MathField.ANSWER: "To find 1.5 times 2/3 cup: (3/2) * (2/3) = 6/6 = 1 cup of flour needed",
		},
		ReasoningState.OUTPUT: {
			NodeField.QUALITY: 7.0,
			ReasoningState.FEEDBACK: "Excellent solution that is correct, complete, and clearly presented (7/7). Demonstrates flawless mathematical reasoning - properly converts 1.5 to fraction form (3/2), correctly multiplies fractions, simplifies to final answer, and includes appropriate units. All steps are logically valid and factually correct. Shows clear progression from problem understanding to conversion to calculation to final answer with excellent communication.",
		},
	},
	{
		# Score 6: Very good solution - correct and well-presented
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "What is the slope of the line passing through (2, 5) and (8, 17)?",
			ToTField.REASONING_STEPS: [
				"I'll use the slope formula: m = (y_2 - y_1)/(x_2 - x_1)",
				"Identifying points: (x_1, y_1) = (2, 5) and (x_2, y_2) = (8, 17)",
				"Substituting: m = (17 - 5)/(8 - 2) = 12/6 = 2",
			],
			MathField.ANSWER: "Using the slope formula m = (y_2 - y_1)/(x_2 - x_1) = (17 - 5)/(8 - 2) = 12/6 = 2",
		},
		ReasoningState.OUTPUT: {
			NodeField.QUALITY: 6.0,
			ReasoningState.FEEDBACK: "Very good solution with correct formula application and accurate calculation (6/7). Correctly applies the slope formula, accurately identifies and substitutes coordinates, and performs arithmetic correctly. Clear and mathematically sound throughout with systematic formula identification, point labeling, and substitution. Only very minor imperfections in presentation prevent a perfect score.",
		},
	},
	{
		# Score 4: Adequate solution - correct but basic presentation
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "Find the area of a circle with radius 3",
			ToTField.REASONING_STEPS: [
				"Using the formula for circle area",
				"Area = π * r² where r = 3",
			],
			MathField.ANSWER: "Area = π * 3² = 9π",
		},
		ReasoningState.OUTPUT: {
			NodeField.QUALITY: 4.0,
			ReasoningState.FEEDBACK: "Adequate solution that is mathematically correct but lacks detailed explanation (4/7). The formula is correct and calculation is accurate, arriving at the right answer of 9π. However, the presentation is notably abbreviated - lacks clarity in showing the substitution steps and proper formula notation (A = πr²). The reasoning is too terse and could mislead learners about proper problem-solving methodology. A more thorough presentation would elevate this solution.",
		},
	},
	{
		# Score 3: Partially correct - right approach, calculation error
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "What is 25% of 120?",
			ToTField.REASONING_STEPS: ["25% = 25/100 = 0.25", "0.25 * 120 = 25"],
			MathField.ANSWER: "25% of 120 is 25",
		},
		ReasoningState.OUTPUT: {
			NodeField.QUALITY: 3.0,
			ReasoningState.FEEDBACK: "Partially correct solution with right method but significant calculation error (3/7). The approach is sound - correctly converts 25% to 0.25 and sets up the multiplication. However, contains a critical arithmetic mistake: 0.25 × 120 = 30, not 25. This computational error completely invalidates the final answer. The error, combined with apparent lack of verification, suggests incomplete understanding and undermines the solution's reliability despite the correct initial approach.",
		},
	},
	{
		# Score 1: Incorrect solution - fundamental formula error
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "What is the circumference of a circle with radius 10?",
			ToTField.REASONING_STEPS: [
				"Using circumference formula for a circle",
				"Circumference = π * radius",
			],
			MathField.ANSWER: "Circumference = π * 10 = 10π",
		},
		ReasoningState.OUTPUT: {
			NodeField.QUALITY: 1.0,
			ReasoningState.FEEDBACK: "Incorrect solution with fundamental formula error (1/7). Uses C = πr instead of the correct formula C = 2πr, demonstrating basic misunderstanding of circle geometry. The answer (10π) is off by a factor of 2 - the correct answer is 20π. This represents a complete disconnect from correct geometric principles and shows confusion about a fundamental mathematical concept. Such a critical conceptual error would propagate to related circle problems and indicates the need for review of basic geometry formulas.",
		},
	},
]

# Custom PRM Demos - For custom logical_rigor + step_clarity rubric (0-10 scale)
PRM_CUSTOM_DEMOS: list[dict[str, Any]] = [
	{
		# High quality reasoning - excellent logical rigor and step clarity
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "Solve for x: 2x + 5 = 13",
			ToTField.REASONING_STEPS: [
				"I need to isolate x by performing inverse operations on both sides",
				"First, subtract 5 from both sides to eliminate the constant term: 2x + 5 - 5 = 13 - 5",
				"This simplifies to: 2x = 8",
				"Next, divide both sides by 2 to isolate x: 2x / 2 = 8 / 2",
				"Therefore: x = 4",
			],
		},
		ReasoningState.OUTPUT: {
			"logical_rigor": 9.0,
			"step_clarity": 10.0,
			"feedback": "Excellent reasoning with strong logical rigor (9.0/10): systematically applies inverse operations, maintains equation balance throughout, and shows clear understanding of algebraic principles. Each step follows logically from the previous one with sound mathematical justification. Step clarity (10.0/10): each operation is explicitly stated with clear explanations of what is being done and why, making the solution easy to follow for learners.",
		},
	},
	{
		# Good quality reasoning - solid logic and clear steps
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "Find the area of a triangle with base 10 and height 6",
			ToTField.REASONING_STEPS: [
				"The area formula for a triangle is A = (1/2) * base * height",
				"Given: base = 10, height = 6",
				"Substituting into the formula: A = (1/2) * 10 * 6 = 30",
			],
		},
		ReasoningState.OUTPUT: {
			"logical_rigor": 8.0,
			"step_clarity": 7.0,
			"feedback": "Strong reasoning with good logical rigor (8.0/10): correctly identifies and applies the triangle area formula, properly substitutes values, and performs accurate calculation. The logic is sound with minor room for more explicit justification. Step clarity (7.0/10): clearly states the formula and shows substitution, though could benefit from showing the intermediate calculation (1/2 × 10 = 5, then 5 × 6 = 30) for enhanced pedagogical value.",
		},
	},
]

# Custom ORM Demos - For custom solution_quality + presentation rubric (0-10 scale)
ORM_CUSTOM_DEMOS: list[dict[str, Any]] = [
	{
		# Excellent solution - high quality and presentation
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "If a train travels 180 miles in 3 hours, what is its average speed?",
			ToTField.REASONING_STEPS: [
				"Average speed is calculated using: speed = distance / time",
				"Given: distance = 180 miles, time = 3 hours",
				"Calculating: speed = 180 miles / 3 hours = 60 miles/hour",
			],
			MathField.ANSWER: "The train's average speed is 60 miles per hour (mph). Using the formula speed = distance/time with 180 miles traveled in 3 hours, we get 180/3 = 60 mph.",
		},
		ReasoningState.OUTPUT: {
			"solution_quality": 10.0,
			"presentation": 9.0,
			"feedback": "Excellent solution with high quality (10.0/10): mathematically correct formula application, accurate calculation, and proper units included. The solution demonstrates complete understanding and arrives at the correct answer with sound reasoning. Presentation (9.0/10): well-organized and clearly communicated with explicit formula statement, proper substitution shown, and final answer highlighted with appropriate units. Easy to follow and understand.",
		},
	},
	{
		# Good solution - correct with solid presentation
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "What is 30% of 150?",
			ToTField.REASONING_STEPS: [
				"30% can be written as 0.30 or 30/100",
				"To find 30% of 150, multiply: 0.30 * 150",
				"0.30 * 150 = 45",
			],
			MathField.ANSWER: "30% of 150 equals 45",
		},
		ReasoningState.OUTPUT: {
			"solution_quality": 8.0,
			"presentation": 7.0,
			"feedback": "Strong solution with good quality (8.0/10): correctly converts percentage to decimal, properly applies multiplication, and achieves accurate result. The mathematical reasoning is sound throughout. Presentation (7.0/10): clearly structured with logical progression from conversion to calculation to answer. Could be enhanced by showing the conversion step more explicitly (30% = 30/100 = 0.30) or including the calculation in the final answer statement for completeness.",
		},
	},
]
